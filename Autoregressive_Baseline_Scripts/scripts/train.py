import sys
import torch
import yaml
import logging
from utils.data_loader import get_data_loaders
import wandb
from models import FFNO, FNO
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.util import autoregressive_rollout, update_input_for_next_timestep, masked_mse_loss_autoregressive, plot_autoregressive_sequence
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration from the YAML file
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
model_type = config["model"]["model_type"]
wandb_run_name = f"{model_type}_autoregressive"  # Example: "FNO_autoregressive" or "FFNO_autoregressive"
wandb.init(
    project=config["wandb"]["project"],
    entity=config["wandb"]["entity"],
    name=wandb_run_name,
    config=config["wandb"]  # This logs the WandB related hyperparameters
)

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Get the training and test DataLoaders using the configuration
train_loader, test_loader = get_data_loaders(config)
logger.info(f"Train dataset size: {len(train_loader.dataset)}, Test dataset size: {len(test_loader.dataset)}")

model_type = config["model"]["model_type"]
if model_type == "FFNO":
    model = FFNO(
        input_dim=config["model"]["input_dim"],
        output_dim=config["model"]["output_dim"],
        modes_x=config["model"]["modes"],
        modes_y=config["model"]["modes"],
        width=config["model"]["width"],
        n_layers=config["model"]["n_layers"],
        factor=config["model"]["factor"],
        n_ff_layers=config["model"]["n_ff_layers"],
        share_weight=config["model"]["share_weight"],
        ff_weight_norm=config["model"]["ff_weight_norm"],
        layer_norm=config["model"]["layer_norm"]
    ).to(device)
elif model_type == "FNO":
    model = FNO(
        input_dim=config["model"]["input_dim"],
        output_dim=config["model"]["output_dim"],
        modes1=config["model"]["modes"],
        modes2=config["model"]["modes"],
        width=config["model"]["width"],
        n_layers=config["model"]["n_layers"],
        retrain_fno=config["model"]["retrain_fno"],
    ).to(device)
else:
    raise ValueError(f"Unknown model type: {model_type}")

if config["optimizer"]["type"] == "Adam":
    optimizer = optim.Adam(model.parameters(), 
                           lr=config["optimizer"]["lr"], 
                           weight_decay=float(config["optimizer"]["weight_decay"]))
else:
    # Add other optimizer options as needed
    raise ValueError("Optimizer type not supported!")

# Set up scheduler from config
if config["scheduler"]["type"] == "StepLR":
    scheduler = StepLR(optimizer, 
                       step_size=config["scheduler"]["step_size"], 
                       gamma=config["scheduler"]["gamma"])
else:
    # Add other scheduler options as needed
    raise ValueError("Scheduler type not supported!")

epochs = config["wandb"]["epochs"]
output_folder = "./output_images"
os.makedirs(output_folder, exist_ok=True)
teacher_forcing_ratio = 0.0

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    for batch_idx, (initial_inputs, target_sequences) in enumerate(train_loader):
        # initial_inputs: (B, H, W, C)
        # target_sequences: (B, T, H, W, C)
        initial_inputs = initial_inputs.to(device)
        target_sequences = target_sequences.to(device)
        optimizer.zero_grad()
        
        # We'll accumulate predictions in a list.
        sequence_preds = []  
        current_input = initial_inputs  # shape (B, H, W, C)
        T = target_sequences.shape[1]     # number of timesteps
        
        for t in range(T):
            # Predict the next timestep; output shape: (B, H, W, C)
            pred_t = model(current_input)
            sequence_preds.append(pred_t)
            
            # Teacher forcing: decide whether to use ground truth or prediction for next input.
            if random.random() < teacher_forcing_ratio:
                next_data = target_sequences[:, t, ...]  # shape: (B, H, W, C)
            else:
                next_data = pred_t
            
            # Update only the first teacher_channels (assumed to be physical fields)
            current_input = update_input_for_next_timestep(current_input, next_data)
        
        # Stack all predictions: shape becomes (B, T, H, W, C)
        sequence_preds = torch.stack(sequence_preds, dim=1)
        
        # Use the mask from initial_inputs (shape: (B, H, W)) for the loss.
        mask = initial_inputs[..., -1]  # (B, H, W)
        loss = masked_mse_loss_autoregressive(sequence_preds, target_sequences, mask)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Evaluation Loop (Option B)
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            initial_inputs, target_sequences = batch
            initial_inputs = initial_inputs.to(device)
            target_sequences = target_sequences.to(device)
            sequence_preds = []
            current_input = initial_inputs
            T = target_sequences.shape[1]
            for t in range(T):
                pred_t = model(current_input)
                sequence_preds.append(pred_t)
                # During evaluation, no teacher forcing: always use the model's prediction.
                current_input = update_input_for_next_timestep(current_input, pred_t)
            sequence_preds = torch.stack(sequence_preds, dim=1)
            mask = initial_inputs[..., -1]
            loss = masked_mse_loss_autoregressive(sequence_preds, target_sequences, mask)
            test_loss += loss.item()
        test_loss /= len(test_loader)
    
    scheduler.step()
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "learning_rate": scheduler.get_last_lr()[0]
    })
    print(f"Epoch {epoch}: train_loss = {train_loss}, test_loss = {test_loss}")
    
    # Visualization
    if (epoch - 1) % 50 == 0:
        model.eval()
        inputs, target_seq = next(iter(test_loader))
        inputs = inputs.to(device)
        target_seq = target_seq.to(device)
        sample_input = inputs[0].unsqueeze(0)       # (1, H, W, C)
        sample_target = target_seq[0].unsqueeze(0)    # (1, T, H, W, C)
        sample_preds = autoregressive_rollout(model, sample_input, sample_target, teacher_forcing_ratio=0.0)
        sample_preds_np = sample_preds.detach().cpu().numpy()[0]  # (T, H, W, C)
        sample_target_np = sample_target.detach().cpu().numpy()[0]  # (T, H, W, C)
        print("Prediction min/max:", sample_preds_np.min(), sample_preds_np.max())
        print("Ground truth min/max:", sample_target_np.min(), sample_target_np.max())
        # Optionally, you can also pass in a mask_array if desired.
        plot_autoregressive_sequence(sample_preds_np, sample_target_np, sim_idx=0, epoch=epoch)

wandb.finish()