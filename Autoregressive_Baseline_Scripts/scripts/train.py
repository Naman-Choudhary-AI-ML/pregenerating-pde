import sys
import torch
import yaml
import logging
from utils.data_loader import get_data_loaders
import wandb
from models import FFNO, FNO, ScOT, ScOTConfig
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
import random
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.util import update_input_for_next_timestep, masked_mse_loss_autoregressive, scot_autoregressive_loss, evaluate_scOT, scot_autoregressive_loss, evaluate_fno_ffno
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
train_loader, test_loader = get_data_loaders(base_config)
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

# Set up scheduler from config
if base_config["scheduler"]["type"] == "StepLR":
    scheduler = StepLR(optimizer, 
                       step_size=base_config["scheduler"]["step_size"], 
                       gamma=base_config["scheduler"]["gamma"])
else:
    # Add other scheduler options as needed
    raise ValueError("Scheduler type not supported!")

epochs = base_config["wandb"]["epochs"]
output_folder = "./output_images"
os.makedirs(output_folder, exist_ok=True)
teacher_forcing_ratio = 0.0
wandb.watch(model, log="all", log_freq=100)
global_step = 0
for epoch in range(1, epochs + 1):
    model.train()
    train_loss_epoch = 0.0
    
    for batch_idx, batch in enumerate(train_loader):
        initial_inputs = batch["pixel_values"].to(device).float()   # shape (B, H, W, C)
        target_sequences = batch["labels"].to(device).float()       # shape (B, T, H, W, C)
        optimizer.zero_grad()
        
        sequence_preds = []
        current_input = initial_inputs
        T = target_sequences.shape[1]
        
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
            p=2
            loss = scot_autoregressive_loss(sequence_preds,target_sequences,p=p,channel_slice_list_normalized_loss=channel_slices,epsilon=1e-8)
        
        loss.backward()
        optimizer.step()
        
        train_loss_epoch += loss.item()
        wandb.log({"train/loss_step": loss.item()}, step=global_step)
        global_step += 1
    
    avg_train_loss = train_loss_epoch / len(train_loader)
    # Conditional Evaluate
    model_type = base_config["model"]["model_type"]
    if model_type in ["FNO", "FFNO"]:
        val_metrics = evaluate_fno_ffno(
            model, test_loader, device, compute_metrics, update_input_for_next_timestep
        )
    else:
        # scOT uses its own evaluate function
        val_metrics = evaluate_scOT(
            model, test_loader, device, compute_metrics, update_input_for_next_timestep
        )
    
    val_metrics_scalar = {}
    for k, v in val_metrics.items():
        if isinstance(v, np.ndarray):
            val_metrics_scalar[k] = float(np.mean(v))
        elif torch.is_tensor(v):
            val_metrics_scalar[k] = float(v.mean().item()) if v.numel() > 1 else float(v.item())
        else:
            val_metrics_scalar[k] = v

    # Log evaluation metrics along with epoch and average training loss
    wandb.log({
        "epoch": epoch,
        "train/loss_epoch": avg_train_loss,
        **{f"val/{k}": v for k, v in val_metrics_scalar.items()}
    }, step=epoch)

    # Format and print evaluation metrics in a human-readable string
    metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in val_metrics_scalar.items())
    print(f"[Epoch {epoch}] train_loss={avg_train_loss:.4f} | val_metrics: {metrics_str}")


wandb.finish()
