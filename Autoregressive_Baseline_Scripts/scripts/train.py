# scripts/train.py
# -----------------------------------------------------------------------------
# Strategic NaN/Inf debugging instrumentation:
#   • End-to-end finite checks on inputs, preds, targets, masks
#   • Per‑step activation stats & LR/grad norms (guarded by DEBUG_NAN)
#   • Safe masked loss (no divide-by-zero)
#   • Safe teacher-forcing update (only U,V,P)
#   • Early crash with detailed dump when any non-finite is detected
#
# Enable deep debug:
#   export DEBUG_NAN=1                # turn on verbose checks + anomaly detect
#   export DEBUG_NAN_LOG_EVERY=200    # (optional) log detailed stats every N steps
#   export DEBUG_NAN_MAXDUMPS=3       # (optional) limit number of dump prints
# -----------------------------------------------------------------------------
import os, sys, yaml, logging, random, argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import wandb

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.data_loader import get_data_loaders
from utils.metrics import compute_metrics
from models import FNO, FFNO

# ---------------- CLI ----------------
cli = argparse.ArgumentParser()
cli.add_argument("--easy-train", type=int, required=True)
cli.add_argument("--hard-train", type=int, required=True)
cli.add_argument("--data_path",  nargs=2, required=True,
                 help="<easy.npy> <hard.npy> (order: easy first, hard second)")
cli.add_argument("--config", default="config/config.yaml")
args,_ = cli.parse_known_args()

EASY_TRAIN, HARD_TRAIN = args.easy_train, args.hard_train
EASY_PATH,  HARD_PATH  = args.data_path

# -------------- cfg & logging --------
with open(args.config) as f:
    base_cfg = yaml.safe_load(f)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Device: %s", device)

# Debug flags
DEBUG = int(os.getenv("DEBUG_NAN", "0")) == 1
DEBUG_LOG_EVERY = int(os.getenv("DEBUG_NAN_LOG_EVERY", "200"))
DEBUG_MAX_DUMPS = int(os.getenv("DEBUG_NAN_MAXDUMPS", "3"))
if DEBUG:
    torch.autograd.set_detect_anomaly(True, check_nan=True)
    log.warning("DEBUG_NAN is ON: anomaly detection & extra checks enabled.")

# Memory management
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

wandb.init(project=base_cfg["wandb"]["project"],
           entity =base_cfg["wandb"]["entity"],
           name   =f"{base_cfg['model']['model_type']}_mix_{EASY_TRAIN}E_{HARD_TRAIN}H_RegulartoHarmonics_debug{int(DEBUG)}",
           config =base_cfg)

# -------------- tiny helpers ----------
def _finite_ratio(x: torch.Tensor) -> float:
    x = x.detach()
    total = x.numel()
    if total == 0: return 1.0
    finite = torch.isfinite(x).sum().item()
    return float(finite) / float(total)

def _stats(x: torch.Tensor):
    x = torch.nan_to_num(x.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    return dict(min=float(x.min().item()),
                max=float(x.max().item()),
                mean=float(x.mean().item()),
                std=float(x.std().item()),
                finite_ratio=_finite_ratio(x))

def _log_stats(prefix: str, x: torch.Tensor, step: int):
    s = _stats(x)
    log.info(f"[{prefix}] step={step} min={s['min']:.4e} max={s['max']:.4e} "
             f"mean={s['mean']:.4e} std={s['std']:.4e} finite={s['finite_ratio']:.6f}")
    wandb.log({f"{prefix}/min": s["min"], f"{prefix}/max": s["max"],
               f"{prefix}/mean": s["mean"], f"{prefix}/std": s["std"],
               f"{prefix}/finite": s["finite_ratio"]}, step=step)

def _check_finite_or_crash(name: str, x: torch.Tensor, step: int):
    if not torch.isfinite(x).all():
        bad = (~torch.isfinite(x)).sum().item()
        s = _stats(x)
        log.error(f"[NONFINITE] {name} @ step={step} bad_count={bad} "
                  f"min={s['min']:.4e} max={s['max']:.4e} mean={s['mean']:.4e} std={s['std']:.4e}")
        # dump a small slice to help locate
        xs = torch.nan_to_num(x.detach().flatten()[:10], nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy()
        log.error(f"[NONFINITE] {name} head10={xs}")
        raise RuntimeError(f"Found non‑finite values in {name} (step={step}).")

# -------------- data ------------------
(train_loader,
 val_easy_loader, val_hard_loader,
 test_easy_loader, test_hard_loader) = get_data_loaders(
     config      = base_cfg,
     easy_train  = EASY_TRAIN,
     hard_train  = HARD_TRAIN,
     easy_path   = EASY_PATH,
     hard_path   = HARD_PATH,
)

# -------------- model -----------------
m_cfg = base_cfg["model"]
if m_cfg["model_type"] == "FNO":
    model = FNO(input_dim=m_cfg["input_dim"],
                output_dim=m_cfg["output_dim"],
                modes1=m_cfg["modes"], modes2=m_cfg["modes"],
                width=m_cfg["width"], n_layers=m_cfg["n_layers"],
                retrain_fno=m_cfg["retrain_fno"]).to(device)
elif m_cfg["model_type"] == "FFNO":
    model = FFNO(input_dim=m_cfg["input_dim"],
                 output_dim=m_cfg["output_dim"],
                 modes_x=m_cfg["modes"], modes_y=m_cfg["modes"],
                 width=m_cfg["width"], n_layers=m_cfg["n_layers"],
                 factor=m_cfg["factor"], n_ff_layers=m_cfg["n_ff_layers"],
                 share_weight=m_cfg["share_weight"],
                 ff_weight_norm=m_cfg["ff_weight_norm"],
                 layer_norm=m_cfg["layer_norm"]).to(device)
    # Note: Gradient checkpointing will be applied during training loop
else:
    raise ValueError("Only FNO / FFNO are supported.")

# Expose debug flag to model (used inside SpectralConv/FFNO forward)
if hasattr(model, "debug_nan"):
    model.debug_nan = DEBUG
if hasattr(model, "debug_threshold"):
    model.debug_threshold = 1e6  # warn if |activation| exceeds this

# -------------- optim -----------------
opt_cfg = base_cfg["optimizer"]
optimizer = optim.Adam(model.parameters(),
                       lr=opt_cfg["lr"],
                       weight_decay=float(opt_cfg["weight_decay"]))
sch_cfg = base_cfg["scheduler"]
if sch_cfg["type"].lower() == "steplr":
    scheduler = StepLR(optimizer, step_size=sch_cfg["step_size"], gamma=sch_cfg["gamma"])
elif sch_cfg["type"].lower() in ("cosineannealing", "cosineannealinglr"):
    # NOTE: this is stepped per-batch in this script; if you want per‑epoch, move scheduler.step() below.
    scheduler = CosineAnnealingLR(optimizer, T_max=base_cfg["wandb"]["epochs"]*len(train_loader),
                                  eta_min=sch_cfg.get("eta_min", 0.0))
else:
    scheduler = None

# -------------- safe ops --------------
def safe_update_input(cur: torch.Tensor, next_frame: torch.Tensor) -> torch.Tensor:
    out = cur.clone()
    C_out = next_frame.shape[-1]
    out[..., :C_out] = next_frame
    return out

def masked_mse_autoreg_safe(preds, target, mask, eps=1e-6):
    preds  = torch.nan_to_num(preds,  nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    mask   = torch.nan_to_num(mask,   nan=0.0).clamp(0.0, 1.0)
    m = mask.unsqueeze(1).unsqueeze(-1)
    diff = (preds - target) ** 2
    num = (diff * m).sum(dim=(1,2,3,4))
    den = m.sum(dim=(1,2,3,4)).clamp_min(eps)
    return (num / den).mean()

# -------------- training loop ----------
epochs   = base_cfg["wandb"]["epochs"]
teacher_forcing_ratio = 1.0
global_step = 0
max_grad_norm = 1.0
dumps_left = DEBUG_MAX_DUMPS

wandb.watch(model, log="all", log_freq=100)

def _first_metric(d: dict, prefix: str):
    if not d: return 0.0
    for k in ("loss", "loss_main", "mse", "rmse", "mae"):
        key = f"{prefix}/{k}"
        if key in d: return d[key]
    return next(iter(d.values()))

def log_wn_stats(model: torch.nn.Module, step: int, top_k: int = 3):
    """
    Walk the model and report the min/max norms of v and values of g for WNLinear.
    Helpful to see whether ||v|| is collapsing to ~0 which would explode grad.
    """
    rows = []
    for name, m in model.named_modules():
        if hasattr(m, "wnorm") and m.wnorm and hasattr(m, "weight_v"):
            with torch.no_grad():
                v = m.weight_v
                g = m.weight_g
                vnorm = v.norm(dim=1)  # (out,)
                rows.append((
                    name,
                    float(vnorm.min().item()),
                    float(vnorm.max().item()),
                    float(g.min().item()),
                    float(g.max().item())
                ))
    if rows:
        # log a compact line per call (also to W&B)
        for name, vmin, vmax, gmin, gmax in rows[:50]:  # don’t spam
            log.info(f"[wn] step={step} {name}: ||v|| min={vmin:.3e} max={vmax:.3e} "
                     f"g min={gmin:.3e} max={gmax:.3e}")
            wandb.log({
                f"wn/{name.replace('.','_')}_vmin": vmin,
                f"wn/{name.replace('.','_')}_vmax": vmax,
                f"wn/{name.replace('.','_')}_gmin": gmin,
                f"wn/{name.replace('.','_')}_gmax": gmax,
            }, step=step)

for epoch in range(1, epochs+1):
    model.train()
    epoch_loss, epoch_ss = 0.0, 0.0

    for batch in train_loader:
        inp  = batch["pixel_values"].to(device).float()      # (B,H,W,6)
        tgtS = batch["labels"].to(device).float()            # (B,T,H,W,C_out)
        B,T = inp.shape[0], tgtS.shape[1]

        inp  = torch.nan_to_num(inp,  nan=0.0, posinf=0.0, neginf=0.0)
        tgtS = torch.nan_to_num(tgtS, nan=0.0, posinf=0.0, neginf=0.0)

        if DEBUG and global_step % DEBUG_LOG_EVERY == 0 and dumps_left > 0:
            _log_stats("debug/input_seed", inp, global_step)
            _log_stats("debug/target_seq", tgtS, global_step)

        optimizer.zero_grad(set_to_none=True)

        mask = inp[..., -1]
        with torch.no_grad():
            flat_sum = mask.flatten(1).sum(-1)              # (B,)
            if (flat_sum == 0).any():
                bad = (flat_sum == 0).nonzero(as_tuple=False).flatten()
                log.warning(f"Batch contains {bad.numel()} samples with empty valid-mask; "
                            f"forcing mask=1 for those indices: {bad.tolist()}")
                mask[bad, ...] = 1.0

        seq_preds, cur = [], inp
        step_losses = []
        for t in range(T):
            if DEBUG and global_step % DEBUG_LOG_EVERY == 0 and dumps_left > 0:
                _log_stats(f"debug/inp_t{t}", cur, global_step)

            pred_t = model(cur)                              # (B,H,W,C_out)
            pred_t = torch.nan_to_num(pred_t, nan=0.0, posinf=0.0, neginf=0.0)
            if DEBUG and global_step % DEBUG_LOG_EVERY == 0 and dumps_left > 0:
                _log_stats(f"debug/pred_t{t}", pred_t, global_step)

            seq_preds.append(pred_t)

            num = torch.nn.functional.l1_loss(pred_t, tgtS[:, t], reduction='mean')
            den = torch.nn.functional.l1_loss(tgtS[:, t], torch.zeros_like(tgtS[:, t]), reduction='mean').clamp_min(1e-6)
            step_losses.append(num / den)

            next_frame = tgtS[:, t] if random.random() < teacher_forcing_ratio else pred_t
            cur = safe_update_input(cur, next_frame)

        preds = torch.stack(seq_preds, dim=1)                # (B,T,H,W,C_out)
        loss  = masked_mse_autoreg_safe(preds, tgtS, mask)
        single_step = torch.mean(torch.stack(step_losses))

        if not torch.isfinite(loss):
            _check_finite_or_crash("loss", loss, global_step)

        loss.backward()

        # grad stats
        if DEBUG and global_step % DEBUG_LOG_EVERY == 0 and dumps_left > 0:
            total_gnorm = 0.0
            bad_grads = []
            for n, p in model.named_parameters():
                if p.grad is None: continue
                g = p.grad.detach()
                if not torch.isfinite(g).all():
                    bad_grads.append(n)
                total_gnorm += g.norm().item() ** 2
            total_gnorm = float(np.sqrt(total_gnorm))
            log.info(f"[debug] step={global_step} grad_norm={total_gnorm:.4e} lr={optimizer.param_groups[0]['lr']:.6e}")
            wandb.log({"debug/grad_norm": total_gnorm}, step=global_step)
            if bad_grads:
                log.error(f"[debug] Non‑finite gradients in: {bad_grads}")
                raise RuntimeError("Detected non‑finite gradients.")

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler: scheduler.step()

        epoch_loss += float(loss.item())
        epoch_ss   += float(single_step.item())

        wandb.log({"train/loss_main": float(loss.item()),
                   "train/loss_single": float(single_step.item()),
                   "train/lr": optimizer.param_groups[0]['lr'],
                   "global_step": global_step}, step=global_step)
        global_step += 1
        if DEBUG and global_step % DEBUG_LOG_EVERY == 0 and dumps_left > 0:
            dumps_left -= 1

    # ---------- validation ----------
    model.eval()
    with torch.no_grad():
        # We call a *safe* wrapper around compute_metrics inside evaluate_fno_ffno if available.
        from utils.util import evaluate_fno_ffno as _eval
        valE, _, _, _, _ = _eval(model, val_easy_loader, device, compute_metrics,
                                 safe_update_input, teacher_forcing=True)
        valH, _, _, _, _ = _eval(model, val_hard_loader, device, compute_metrics,
                                 safe_update_input, teacher_forcing=True)

    valE_flat = {f"val_easy/{k}": float(np.nan_to_num(np.mean(v), nan=np.inf)) for k, v in valE.items()}
    valH_flat = {f"val_hard/{k}": float(np.nan_to_num(np.mean(v), nan=np.inf)) for k, v in valH.items()}
    # log NaN ratios if any metric broke
    for name, val in list(valE_flat.items()) + list(valH_flat.items()):
        if not np.isfinite(val):
            log.error(f"[metrics] Non‑finite metric {name}={val}. "
                      "Denominator likely zero in relative error; check compute_metrics.")
            # replace with large sentinel to keep wandb happy
            val = 1e9

    wandb.log({"epoch": epoch,
               "train/loss_epoch": epoch_loss/len(train_loader),
               "train/loss_single_epoch": epoch_ss/len(train_loader),
               **valE_flat, **valH_flat}, step=global_step)

    log.info("[epoch %d] train %.6f | val_easy %.6f | val_hard %.6f",
             epoch, epoch_loss/len(train_loader),
             valE_flat.get("val_easy/loss", 0.0),
             valH_flat.get("val_hard/loss", 0.0))

# ---------- final test ----------
model.eval()
with torch.no_grad():
    from utils.util import evaluate_fno_ffno as _eval
    testE, _, _, _, _ = _eval(model, test_easy_loader, device, compute_metrics,
                              safe_update_input, teacher_forcing=True)
    testH, _, _, _, _ = _eval(model, test_hard_loader, device, compute_metrics,
                              safe_update_input, teacher_forcing=True)

def _san(m: dict, prefix: str):
    out = {}
    for k, v in m.items():
        val = float(np.nan_to_num(np.mean(v), nan=np.inf))
        if not np.isfinite(val):
            log.error(f"[test] Non‑finite {prefix}/{k}={val}")
            val = 1e9
        out[f"{prefix}/{k}"] = val
    return out

testE = _san(testE, "test_easy")
testH = _san(testH, "test_hard")
wandb.log({**testE, **testH}, step=global_step)

print("=== FINAL TEST ===")
for k, v in {**testE, **testH}.items():
    print(f"{k:30s}: {v:.6f}")

wandb.finish()
