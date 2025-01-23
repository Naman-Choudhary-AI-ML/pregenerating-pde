#!pip install einops

import torch.nn as nn
import torch
import einops

from einops import rearrange
from einops.layers.torch import Rearrange

import random
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

from torch.optim import AdamW
from the_well.data import WellDataset  # For loading your dataset
from einops import rearrange          # Ensures tensor reshaping
import wandb
import os


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Applies following operations:
# 1. LayerNorm
# 2. nn.Linear (dim --> hidden_dim)
# 3. GELU
# 4. Dropout
# 5. nn.Linear (hidden_dim --> dim)
# 6. Dropout

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Takes a sequence of embedding of dimension dim
# 1. Applies LayerNorm
# 2. Applies linear layer dim -> 3x inner_dim
#                                NOTE: inner_dim = dim_head x heads
# 3. Applies attention
# 4. Projects inner -> dim

class AttentionBlock(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# Takes sequence of embeddings of dimension dim
# 1. Applies depth times:
#    a) Attention block: dim->dim (in the laast dimension)
#    b) MLP block:       dim->dim (in the laast dimension)
# 2. Applies LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                AttentionBlock(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# Takes an image of size (n, c, h, w)
# Finds patch sizes (p_h, p_w) & number of patches (n_h, n_w)
# NOTE: It must hold that h%p_h == 0

# 1. Applies to_patch_embedding :
#     a. (n, c, p_h*p1, p_w*p2) -> (n, n_h*n_w, p_h*p_w*c)
#     b. LayerNorm
#     c. Linear embedding p_h*p_w*c -> dim
#     d. LayerNorm
# 2. Add positional embedding
# 3. Apply Transformer Block
# 4. Depatchify

class ViT(nn.Module):
    def __init__(self,
                image_size,
                patch_size,
                dim,
                depth,
                heads,
                mlp_dim = 256,
                channels = 1,
                dim_head = 32,
                emb_dropout = 0.,):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.patch_to_image = nn.Sequential(
            nn.Linear(dim, patch_dim),
            nn.LayerNorm(patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_height, p2 = patch_width, h = image_height // patch_height)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerBlock(dim, depth, heads, dim_head, mlp_dim)

        self.conv_last = torch.nn.Conv2d(in_channels = channels,
                                          out_channels= channels,
                                          kernel_size = 3,
                                          padding     = 1)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        _, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.patch_to_image(x)
        x = self.conv_last(x)
        return x


    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters: {nparams})')

        return nparams
    
########################
# Dataloader definition:
########################
class TurbulentRadiativeDatasetSingleStep(Dataset):
    def __init__(self, dataset, which="train"):
        """
        Dataset for Turbulent Radiative Layer - 2D.
        Returns 1 input timestep and 1 output timestep.
        
        Args:
        - dataset: An instance of WellDataset (already loaded with input/output fields).
        - which: 'train', 'val', or 'test' to specify the data split.
        """
        self.dataset = dataset  # WellDataset object
        self.which = which

        # Length of the dataset
        self.length = len(dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        Retrieves input and output for the given index.
        - Input: First timestep as [Lx, Ly, F].
        - Output: Next timestep as [Lx, Ly, F].
        """
        item = self.dataset[index]
        input_data = torch.tensor(item["input_fields"][0]).type(torch.float32)  # Shape: [Lx, Ly, F]
        output_data = torch.tensor(item["output_fields"][0]).type(torch.float32)  # Shape: [Lx, Ly, F]

        # Flatten spatial fields for ViT (channels = F)
        input_data = rearrange(input_data, "Lx Ly F -> F Lx Ly")
        output_data = rearrange(output_data, "Lx Ly F -> F Lx Ly")

        return input_data, output_data

# Instantiate WellDataset with n_steps_input=1, n_steps_output=1
train_dataset = WellDataset(
    well_base_path="/home/namancho/datasets/datasets",
    well_dataset_name="turbulent_radiative_layer_2D",
    well_split_name="train",
    n_steps_input=1,  # Single timestep input
    n_steps_output=1,  # Single timestep output
    use_normalization=False,
)

val_dataset = WellDataset(
    well_base_path="/home/namancho/datasets/datasets",
    well_dataset_name="turbulent_radiative_layer_2D",
    well_split_name="valid",
    n_steps_input=1,
    n_steps_output=1,
    use_normalization=False,
)

# Wrap in PyTorch Dataset
train_data = TurbulentRadiativeDatasetSingleStep(train_dataset, which="train")
val_data = TurbulentRadiativeDatasetSingleStep(val_dataset, which="val")

# Define DataLoaders
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

image_size = (128, 384)  # Spatial dimensions of each timestep
patch_size = (16, 16)    # Size of patches
dim = 128                # Dimensionality of patch embeddings
depth = 4                # Number of transformer blocks
heads = 4                # Number of attention heads
dim_head = 32            # Dimensionality of each attention head
emb_dropout = 0.0
channels = 4             # Number of fields (density, pressure, velocity_x, velocity_y)
learning_rate = 0.001
epochs = 200

model = ViT(
    image_size=image_size,
    patch_size=patch_size,
    dim=dim,
    depth=depth,
    heads=heads,
    mlp_dim=256,
    channels=channels,
    dim_head=dim_head,
    emb_dropout=emb_dropout,
).to(device)

model.print_size()

import os

# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=200, eta_min=10**-6)

# Define loss function
criterion = nn.MSELoss()

# Initialize WandB
wandb.init(
    project="GeoFNO1",
    entity="namancho",
    name="ViT_TurbulentRadiative_1",
    config={
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": 16,
        "image_size": image_size,
        "patch_size": patch_size,
        "dim": dim,
        "depth": depth,
        "heads": heads,
        "dim_head": dim_head,
    }
)

freq_print = 1
freq_plot = 50  # Save plots every 50 epochs
plot_dir = "./plots"  # Directory to save plots
os.makedirs(plot_dir, exist_ok=True)

# Training loop
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for step, (input_batch, output_batch) in enumerate(train_loader):
        input_batch, output_batch = input_batch.to(device), output_batch.to(device)

        # Forward pass
        optimizer.zero_grad()
        output_pred_batch = model(input_batch)
        loss = criterion(output_pred_batch, output_batch)

        # Backward pass
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    scheduler.step()

    # Evaluation
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for input_batch, output_batch in val_loader:
            input_batch, output_batch = input_batch.to(device), output_batch.to(device)

            # Forward pass
            output_pred_batch = model(input_batch)
            loss = criterion(output_pred_batch, output_batch)
            test_loss += loss.item()

        test_loss /= len(val_loader)
    # Print epoch loss
    print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")

    # WandB Logging
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "learning_rate": scheduler.get_last_lr()[0]
    })

    # Save plots every freq_plot epochs
    if epoch % freq_plot == 0 or epoch == epochs - 1:
        input_sample = input_batch[0].cpu().numpy()  # First input sample in batch
        output_sample = output_batch[0].cpu().numpy()  # First ground truth sample
        pred_sample = output_pred_batch[0].cpu().detach().numpy()  # First prediction

        # Define the field names in the correct order
        field_names = ["Density", "Pressure", "Velocity_x", "Velocity_y"]

        # Get number of fields (channels)
        num_fields = input_sample.shape[0]

        # Create a figure with num_fields rows and 3 columns
        fig, axes = plt.subplots(num_fields, 3, figsize=(15, 4 * num_fields))

        for field in range(num_fields):
            # Color range for Input (independent)
            input_vmin = np.min(input_sample[field])
            input_vmax = np.max(input_sample[field])

            # Color range for Ground Truth and Prediction (shared)
            # shared_vmin = min(np.min(output_sample[field]))
            # shared_vmax = max(np.max(output_sample[field]))
            shared_vmin = input_vmin
            shared_vmax = input_vmax

            # Plot Input
            im0 = axes[field, 0].imshow(
                input_sample[field], cmap="RdBu_r", interpolation="none", vmin=input_vmin, vmax=input_vmax
            )
            axes[field, 0].set_title(f"Input ({field_names[field]})")
            axes[field, 0].set_xticks([])
            axes[field, 0].set_yticks([])
            # Add color bar for Input
            cbar_input = fig.colorbar(im0, ax=axes[field, 0], orientation="horizontal", fraction=0.05, pad=0.1)
            cbar_input.set_label("Input Color Scale")

            # Plot Ground Truth
            im1 = axes[field, 1].imshow(
                output_sample[field], cmap="RdBu_r", interpolation="none", vmin=shared_vmin, vmax=shared_vmax
            )
            axes[field, 1].set_title(f"Ground Truth ({field_names[field]})")
            axes[field, 1].set_xticks([])
            axes[field, 1].set_yticks([])
            # Add color bar for Ground Truth
            cbar_gt = fig.colorbar(im1, ax=axes[field, 1], orientation="horizontal", fraction=0.05, pad=0.1)
            cbar_gt.set_label("Ground Truth Color Scale")

            # Plot Prediction
            im2 = axes[field, 2].imshow(
                pred_sample[field], cmap="RdBu_r", interpolation="none", vmin=shared_vmin, vmax=shared_vmax
            )
            axes[field, 2].set_title(f"Prediction ({field_names[field]})")
            axes[field, 2].set_xticks([])
            axes[field, 2].set_yticks([])
            # Add color bar for Prediction
            cbar_pred = fig.colorbar(im2, ax=axes[field, 2], orientation="horizontal", fraction=0.05, pad=0.1)
            cbar_pred.set_label("Prediction Color Scale")

        # plt.tight_layout()

        # Save plot to file
        plot_path = os.path.join(plot_dir, f"epoch_{epoch}.png")
        plt.savefig(plot_path)
        plt.close()



