import sys
import torch
import yaml
import logging
from utils.data_loader import get_data_loaders
import wandb
from models import FFNO, FNO, ScOT, ScOTConfig
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import os
import random
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.util import update_input_for_next_timestep, masked_mse_loss_autoregressive, scot_autoregressive_loss, evaluate_scOT, scot_autoregressive_loss, evaluate_fno_ffno, plot_autoregressive_sequence
from utils.metrics import compute_metrics
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration from the YAML file
with open("config/config.yaml", "r") as f:
    base_config = yaml.safe_load(f)
model_type = base_config["model"]["model_type"]
wandb_run_name = f"{model_type}_autoregressive"  # Example: "FNO_autoregressive" or "FFNO_autoregressive"
wandb.init(
    project=base_config["wandb"]["project"],
    entity=base_config["wandb"]["entity"],
    name=wandb_run_name,
    config=base_config["wandb"]  # This logs the WandB related hyperparameters
)

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
RANK = 0
MODEL_MAP = {
    "T": {
        "num_heads": [3, 6, 12, 24],
        "skip_connections": [2, 2, 2, 0],
        "window_size": 16,
        "patch_size": 4,
        "mlp_ratio": 4.0,
        "depths": [4, 4, 4, 4],
        "embed_dim": 48,
    },
    "S": {
        "num_heads": [3, 6, 12, 24],
        "skip_connections": [2, 2, 2, 0],
        "window_size": 16,
        "patch_size": 4,
        "mlp_ratio": 4.0,
        "depths": [8, 8, 8, 8],
        "embed_dim": 48,
    },
    "B": {
        "num_heads": [3, 6, 12, 24],
        "skip_connections": [2, 2, 2, 0],
        "window_size": 16,
        "patch_size": 4,
        "mlp_ratio": 4.0,
        "depths": [8, 8, 8, 8],
        "embed_dim": 96,
    },
    "L": {
        "num_heads": [3, 6, 12, 24],
        "skip_connections": [2, 2, 2, 0],
        "window_size": 16,
        "patch_size": 4,
        "mlp_ratio": 4.0,
        "depths": [8, 8, 8, 8],
        "embed_dim": 192,
    },
}

# Get the training and test DataLoaders using the configuration
train_loader, val_loader, test_loader = get_data_loaders(base_config)
logger.info(f"Train dataset size: {len(train_loader.dataset)}, Test dataset size: {len(test_loader.dataset)}")

if MODEL_MAP and (
        type(base_config["model"]["model_name"]) == str and base_config["model"]["model_name"] in MODEL_MAP.keys()
    ):
        final_config = {**base_config, **MODEL_MAP[base_config["model"]["model_name"]]}
        if RANK == 0 or RANK == -1:
            wandb.config.update(MODEL_MAP[base_config["model"]["model_name"]], allow_val_change=True)

model_type = base_config["model"]["model_type"]
if model_type == "FFNO":
    model = FFNO(
        input_dim=base_config["model"]["input_dim"],
        output_dim=base_config["model"]["output_dim"],
        modes_x=base_config["model"]["modes"],
        modes_y=base_config["model"]["modes"],
        width=base_config["model"]["width"],
        n_layers=base_config["model"]["n_layers"],
        factor=base_config["model"]["factor"],
        n_ff_layers=base_config["model"]["n_ff_layers"],
        share_weight=base_config["model"]["share_weight"],
        ff_weight_norm=base_config["model"]["ff_weight_norm"],
        layer_norm=base_config["model"]["layer_norm"]
    ).to(device)
elif model_type == "FNO":
    model = FNO(
        input_dim=base_config["model"]["input_dim"],
        output_dim=base_config["model"]["output_dim"],
        modes1=base_config["model"]["modes"],
        modes2=base_config["model"]["modes"],
        width=base_config["model"]["width"],
        n_layers=base_config["model"]["n_layers"],
        retrain_fno=base_config["model"]["retrain_fno"],
    ).to(device)
elif model_type == "scOT":
    model_config = ScOTConfig(
            image_size=128,
            patch_size=final_config["patch_size"],
            num_channels=base_config["model"]["input_dim"],
            num_out_channels= base_config["model"]["output_dim"],
            embed_dim=final_config["embed_dim"],
            depths=final_config["depths"],
            num_heads=final_config["num_heads"],
            skip_connections=final_config["skip_connections"],
            window_size=final_config["window_size"],
            mlp_ratio=final_config["mlp_ratio"],
            qkv_bias=True,
            hidden_dropout_prob=0.0,  # default
            attention_probs_dropout_prob=0.0,  # default
            drop_path_rate=0.0,
            hidden_act="gelu",
            use_absolute_embeddings=False,
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            p=1,
            channel_slice_list_normalized_loss= list(range(base_config["model"]["output_dim"])),
            residual_model="convnext",
            use_conditioning=True,
            learn_residual=False,
        )
    model = ScOT(model_config).to(device)
else:
    raise ValueError(f"Unknown model type: {model_type}")

if base_config["optimizer"]["type"] == "Adam":
    optimizer = optim.Adam(model.parameters(), 
                           lr=base_config["optimizer"]["lr"], 
                           weight_decay=float(base_config["optimizer"]["weight_decay"]))
else:
    # Add other optimizer options as needed
    raise ValueError("Optimizer type not supported!")
T_max = base_config["wandb"]["epochs"] * 50
# Set up scheduler from config
if base_config["scheduler"]["type"] == "StepLR":
    scheduler = StepLR(optimizer, 
                       step_size=base_config["scheduler"]["step_size"], 
                       gamma=base_config["scheduler"]["gamma"])
elif base_config["scheduler"]["type"] == "cosineannealing":
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=T_max,
        eta_min=base_config["scheduler"].get("eta_min", 0)
    )

else:
    # Add other scheduler options as needed
    raise ValueError("Scheduler type not supported!")

epochs = base_config["wandb"]["epochs"]
output_folder = "./output_images"
os.makedirs(output_folder, exist_ok=True)
teacher_forcing_ratio = 1.0
wandb.watch(model, log="all", log_freq=100)
global_step = 0
for epoch in range(1, epochs + 1):
    model.train()
    train_loss_epoch = 0.0
    train_loss_single_step_epoch = 0.0
    
    for batch_idx, batch in enumerate(train_loader):
        initial_inputs = batch["pixel_values"].to(device).float()   # shape (B, H, W, C)
        target_sequences = batch["labels"].to(device).float()       # shape (B, T, H, W, C)
        optimizer.zero_grad()
        
        sequence_preds = []
        current_input = initial_inputs
        T = target_sequences.shape[1]
        single_step_losses = []
        
        for t in range(T):
            # Predict next timestep
            if base_config["model"]["model_type"] in ["FNO", "FFNO"]:
                # FNO/FFNO style
                pred_t = model(current_input)  # (B, H, W, C)
            else:
                # scOT style
                times = torch.full(
                    (current_input.shape[0], 1),t,device=device,dtype=current_input.dtype)
                scot_out = model(current_input, time=times)
                pred_t = scot_out.output  # also (B, H, W, C) by design
                
            sequence_preds.append(pred_t)

            numerator = torch.nn.functional.l1_loss(pred_t, target_sequences[:, t], reduction='mean')
            # For “relative” we compare to the L1 of ground truth from 0
            denominator = torch.nn.functional.l1_loss(
                target_sequences[:, t],
                torch.zeros_like(target_sequences[:, t]),
                reduction='mean'
            )
            single_step_loss_t = numerator / (denominator + 1e-8)
            single_step_losses.append(single_step_loss_t)
            
            # Teacher forcing
            if random.random() < teacher_forcing_ratio:
                next_data = target_sequences[:, t]  # (B, H, W, C)
            else:
                next_data = pred_t
            
            current_input = update_input_for_next_timestep(current_input, next_data)
        
        # Stack predictions => (B, T, H, W, C)
        sequence_preds = torch.stack(sequence_preds, dim=1)
        if base_config["model"]["model_type"] == "FNO" or base_config["model"]["model_type"] == "FFNO":
            mask = initial_inputs[..., -1]  # (B, H, W)
            loss = masked_mse_loss_autoregressive(sequence_preds, target_sequences, mask)
        else:
            mask = initial_inputs[:, -1, :, :]
            channel_slices = model_config.channel_slice_list_normalized_loss
            p=1
            loss = scot_autoregressive_loss(sequence_preds,target_sequences,p=p,channel_slice_list_normalized_loss=channel_slices,epsilon=1e-8)
        single_step_loss = torch.mean(torch.stack(single_step_losses))
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        train_loss_epoch += loss.item()
        train_loss_single_step_epoch += single_step_loss.item()
        current_lr = (scheduler.get_last_lr()[0]
                      if scheduler is not None
                      else optimizer.param_groups[0]['lr'])

        wandb.log(
            {
                "train/loss_main": loss.item(),
                "train/loss_single_step": single_step_loss.item(),
                "train/learning_rate": current_lr,
                "train/epoch": epoch,
                "train/global_step": global_step
            },
            step=global_step
        )
        global_step += 1
    
    avg_train_loss = train_loss_epoch / len(train_loader)
    avg_train_loss_single_step = train_loss_single_step_epoch / len(train_loader)
    
    model.eval()
    with torch.no_grad():
        model_type = base_config["model"]["model_type"]
        if model_type in ["FNO", "FFNO"]:
            val_metrics, pred_1sim, target_1sim, N, T = evaluate_fno_ffno(model, val_loader, device, compute_metrics, update_input_for_next_timestep, teacher_forcing=True)
        else:
            val_metrics = evaluate_scOT(
                model, val_loader, device, compute_metrics, update_input_for_next_timestep
            )

    # Convert validation metrics to scalars
    val_metrics_scalar = {}
    for k, v in val_metrics.items():
        if isinstance(v, np.ndarray):
            val_metrics_scalar[k] = float(np.mean(v))
        elif torch.is_tensor(v):
            val_metrics_scalar[k] = float(v.mean().item()) if v.numel() > 1 else float(v.item())
        else:
            val_metrics_scalar[k] = v

    # Log epoch metrics (including validation)
    wandb.log({
    "epoch": epoch,
    "train/loss_epoch": avg_train_loss,
    "train/loss_epoch_single_step": avg_train_loss_single_step,
    **{f"val/{k}": v for k, v in val_metrics_scalar.items() if k != "single_step_loss"}
}, step=global_step)

    # Print summary
    metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in val_metrics_scalar.items())
    print(f"[Epoch {epoch}] train_loss={avg_train_loss:.4f} | val_metrics: {metrics_str}")

    if epoch == 1 or (epoch % 50) == 0:
        plot_autoregressive_sequence(
            prediction=pred_1sim,
            truth=target_1sim,
            sim_idx=0,
            epoch=epoch,
            save_dir="plots",
            output_dim=3,
            channel_names=["Ch0", "Ch1", "Ch2"]
        )

# -------------------------
# Final Test Evaluation (after all epochs)
# -------------------------
model.eval()
with torch.no_grad():
    model_type = base_config["model"]["model_type"]
    if model_type in ["FNO", "FFNO"]:
        test_metrics, test_pred_1sim, test_target_1sim, N, T = evaluate_fno_ffno(
            model, test_loader, device, compute_metrics, update_input_for_next_timestep, teacher_forcing=False
        )
    else:
        test_metrics = evaluate_scOT(
            model, test_loader, device, compute_metrics, update_input_for_next_timestep
        )

test_metrics_scalar = {}
for k, v in test_metrics.items():
    if isinstance(v, np.ndarray):
        test_metrics_scalar[k] = float(np.mean(v))
    elif torch.is_tensor(v):
        test_metrics_scalar[k] = float(v.mean().item()) if v.numel() > 1 else float(v.item())
    else:
        test_metrics_scalar[k] = v

wandb.log({
    **{f"test/{k}": v for k, v in test_metrics_scalar.items()}
}, step=global_step)

test_str = " | ".join(f"{k}={v:.4f}" for k, v in test_metrics_scalar.items())
print(f"[Final Test] {test_str}")

wandb.finish()
