# train.py  (root of project or scripts/train.py – wherever you launch it)
# ─────────────────────────────────────────────────────────────────────────────
# FNO / FFNO autoregressive training with *separate* easy vs. hard
# validation + test splits.
# -----------------------------------------------------------------------------
import os, sys, yaml, logging, random, argparse
import numpy as np
import torch, torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import wandb

# local imports ---------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.data_loader import get_data_loaders
from utils.util   import (update_input_for_next_timestep,
                          masked_mse_loss_autoregressive,
                          evaluate_fno_ffno)
from utils.metrics import compute_metrics
from models import FNO, FFNO

# ───────────────────────── CLI flags (easy, hard, paths) ─────────────────────
cli = argparse.ArgumentParser()
cli.add_argument("--easy-train", type=int, required=True)
cli.add_argument("--hard-train", type=int, required=True)
cli.add_argument("--data_path",  nargs=2, required=True,
                 help="<easy.npy> <hard.npy> (order matters)")
args,_ = cli.parse_known_args()

EASY_TRAIN, HARD_TRAIN = args.easy_train, args.hard_train
EASY_PATH,  HARD_PATH  = args.data_path

# ─────────────────────────── configuration & logging ─────────────────────────
with open("config/config.yaml") as f: base_cfg = yaml.safe_load(f)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Device: %s", device)

wandb.init(project=base_cfg["wandb"]["project"],
           entity =base_cfg["wandb"]["entity"],
           name   =f"{base_cfg['model']['model_type']}_mix_{EASY_TRAIN}E_{HARD_TRAIN}H",
           config =base_cfg)

# ───────────────────────────── DataLoaders ───────────────────────────────────
(train_loader,
 val_easy_loader, val_hard_loader,
 test_easy_loader, test_hard_loader) = get_data_loaders(
     config      = base_cfg,
     easy_train  = EASY_TRAIN,
     hard_train  = HARD_TRAIN,
     easy_path   = EASY_PATH,
     hard_path   = HARD_PATH,
 )

# ─────────────────────────────── model init ──────────────────────────────────
m_cfg = base_cfg["model"]
if m_cfg["model_type"] == "FNO":
    model = FNO( input_dim = m_cfg["input_dim"],
                 output_dim= m_cfg["output_dim"],
                 modes1    = m_cfg["modes"],
                 modes2    = m_cfg["modes"],
                 width     = m_cfg["width"],
                 n_layers  = m_cfg["n_layers"],
                 retrain_fno = m_cfg["retrain_fno"] ).to(device)
elif m_cfg["model_type"] == "FFNO":
    model = FFNO(input_dim = m_cfg["input_dim"],
                 output_dim= m_cfg["output_dim"],
                 modes_x   = m_cfg["modes"],
                 modes_y   = m_cfg["modes"],
                 width     = m_cfg["width"],
                 n_layers  = m_cfg["n_layers"],
                 factor    = m_cfg["factor"],
                 n_ff_layers   = m_cfg["n_ff_layers"],
                 share_weight  = m_cfg["share_weight"],
                 ff_weight_norm= m_cfg["ff_weight_norm"],
                 layer_norm    = m_cfg["layer_norm"]
                ).to(device)
else:
    raise ValueError("Only FNO / FFNO are supported in this script.")

# ──────────────────────────── optimizer & sched ──────────────────────────────
opt_cfg = base_cfg["optimizer"]
optimizer = optim.Adam(model.parameters(),
                       lr = opt_cfg["lr"],
                       weight_decay = float(opt_cfg["weight_decay"]))
sch_cfg = base_cfg["scheduler"]
if sch_cfg["type"] == "StepLR":
    scheduler = StepLR(optimizer, step_size=sch_cfg["step_size"],
                       gamma=sch_cfg["gamma"])
elif sch_cfg["type"] == "cosineannealing":
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max = base_cfg["wandb"]["epochs"]*50,
                                  eta_min = sch_cfg.get("eta_min",0))
else:
    scheduler = None

# ───────────────────────────── training loop ─────────────────────────────────
epochs   = base_cfg["wandb"]["epochs"]
teacher_forcing_ratio = 1.0
global_step = 0

wandb.watch(model, log="all", log_freq=100)

for epoch in range(1, epochs+1):
    # ---------- training -----------------------------------------------------
    model.train()
    epoch_loss, epoch_ss = 0., 0.
    for batch in train_loader:
        inp  = batch["pixel_values"].to(device).float()      # (B,H,W,C)
        tgtS = batch["labels"].to(device).float()            # (B,T,H,W,C_out)
        optimizer.zero_grad()

        seq_preds, cur = [], inp
        step_losses = []
        for t in range(tgtS.shape[1]):
            pred_t = model(cur)                              # (B,H,W,C_out)
            seq_preds.append(pred_t)

            # step loss (relative L1)
            num = torch.nn.functional.l1_loss(pred_t, tgtS[:,t], reduction='mean')
            den = torch.nn.functional.l1_loss(tgtS[:,t],
                                              torch.zeros_like(tgtS[:,t]),
                                              reduction='mean')
            step_losses.append(num/(den+1e-8))

            # teacher forcing
            next_frame = tgtS[:,t] if random.random() < teacher_forcing_ratio else pred_t
            cur = update_input_for_next_timestep(cur, next_frame)

        preds = torch.stack(seq_preds, dim=1)
        mask  = inp[..., -1]
        loss  = masked_mse_loss_autoregressive(preds, tgtS, mask)
        single_step = torch.mean(torch.stack(step_losses))

        loss.backward()
        optimizer.step()
        if scheduler: scheduler.step()

        epoch_loss += loss.item()
        epoch_ss   += single_step.item()

        wandb.log({"train/loss_main": loss.item(),
                   "train/loss_single": single_step.item(),
                   "train/lr": optimizer.param_groups[0]['lr'],
                   "global_step": global_step}, step=global_step)
        global_step += 1

    # ---------- validation ---------------------------------------------------
    model.eval()
    with torch.no_grad():
        valE, _, _, _, _ = evaluate_fno_ffno(model, val_easy_loader, device,
                                             compute_metrics,
                                             update_input_for_next_timestep,
                                             teacher_forcing=True)
        valH, _, _, _, _ = evaluate_fno_ffno(model, val_hard_loader, device,
                                             compute_metrics,
                                             update_input_for_next_timestep,
                                             teacher_forcing=True)

    # flatten for wandb
    valE_flat = {f"val_easy/{k}":float(np.mean(v)) for k,v in valE.items()}
    valH_flat = {f"val_hard/{k}":float(np.mean(v)) for k,v in valH.items()}

    wandb.log({"epoch": epoch,
               "train/loss_epoch": epoch_loss/len(train_loader),
               "train/loss_single_epoch": epoch_ss/len(train_loader),
               **valE_flat, **valH_flat}, step=global_step)

    log.info("[epoch %d] train %.4f | val_easy %.4f | val_hard %.4f",
             epoch, epoch_loss/len(train_loader),
             valE_flat.get("val_easy/loss", 0.),
             valH_flat.get("val_hard/loss", 0.))

# ───────────────────────────── final test ------------------------------------
model.eval()
with torch.no_grad():
    testE, _, _, _, _ = evaluate_fno_ffno(model, test_easy_loader, device,
                                          compute_metrics,
                                          update_input_for_next_timestep,
                                          teacher_forcing=True)
    testH, _, _, _, _ = evaluate_fno_ffno(model, test_hard_loader, device,
                                          compute_metrics,
                                          update_input_for_next_timestep,
                                          teacher_forcing=True)

testE = {f"test_easy/{k}":float(np.mean(v)) for k,v in testE.items()}
testH = {f"test_hard/{k}":float(np.mean(v)) for k,v in testH.items()}
wandb.log({**testE, **testH})

print("=== FINAL TEST ===")
for k,v in {**testE, **testH}.items():
    print(f"{k:30s}: {v:.4f}")

wandb.finish()
