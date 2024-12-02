import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb
from torch.optim.lr_scheduler import StepLR
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from netCDF4 import Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# dataset with shape (1177, 21, 128, 128, 2)
path = '/home/namancho/datasets/NS-SL-Openfoam/openfoam.npy'


# # Load the .nc file
# with Dataset(path, mode='r') as nc:
#     data = nc.variables['velocity'][:]  # Assuming 'velocity' is the data key
data = np.load(path)  # Replace 'data.npy' with your actual data file
print("SHAPE", data.shape)

dataset_name = os.path.basename(os.path.dirname(path))  # This gives the folder name like 'NS-PwC'
output_folder = dataset_name
# output_folder = "NS-Sines-Poseidon-2-trial"

# Create the directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Use time step 0 as input and time step 1 as target
input_data = data[:, 0, :, :, :]  # Shape: (1177, 128, 128, 2)
target_data = data[:, 1, :, :, :]  # Shape: (1177, 128, 128, 2)

print("input data, target data", input_data.shape, target_data.shape)

# Transpose to shape (1177, 128, 128, 2)
# input_data = np.transpose(input_data, (0, 2, 3, 1))
# target_data = np.transpose(target_data, (0, 2, 3, 1))

print("input data, target data after transpose", input_data.shape, target_data.shape)

# Create x and y coordinate grids and expand to match the batch size
x_grid = np.linspace(0, 1, input_data.shape[1])
y_grid = np.linspace(0, 1, input_data.shape[2])
xv, yv = np.meshgrid(x_grid, y_grid, indexing='ij')

# Expand and tile xv and yv to match the input_data batch size
xv = np.tile(xv[None, :, :, None], (input_data.shape[0], 1, 1, 1))  # Shape: (1177, 128, 128, 1)
yv = np.tile(yv[None, :, :, None], (input_data.shape[0], 1, 1, 1))  # Shape: (1177, 128, 128, 1)

# Concatenate input data with x and y coordinates
input_with_coords = np.concatenate([input_data, xv, yv], axis=-1)  # Shape: (1177, 128, 128, 4)

# Convert numpy arrays to PyTorch tensors
input_tensor = torch.from_numpy(input_with_coords).float().to(device)  # Shape: (1177, 128, 128, 4)
target_tensor = torch.from_numpy(target_data).float().to(device)        # Shape: (1177, 128, 128, 2)

# Assuming 80% train and 20% test split
train_size = int(0.8 * len(input_tensor))
test_size = len(input_tensor) - train_size

# Manually slice the dataset
train_input, test_input = input_tensor[:train_size], input_tensor[train_size:]
train_target, test_target = target_tensor[:train_size], target_tensor[train_size:]

# Create separate TensorDatasets
train_dataset = TensorDataset(train_input, train_target)
test_dataset = TensorDataset(test_input, test_target)

# Create DataLoader objects
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



################################################################
#  2d fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, fno_architecture, device=None, padding_frac=1 / 4):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.modes1 = fno_architecture["modes"]
        self.modes2 = fno_architecture["modes"]
        self.width = fno_architecture["width"]
        self.n_layers = fno_architecture["n_layers"]
        self.retrain_fno = fno_architecture["retrain_fno"]

        torch.manual_seed(self.retrain_fno)
        # self.padding = 9 # pad the domain if input is non-periodic
        self.padding_frac = padding_frac
        self.fc0 = nn.Linear(4, self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv_list = nn.ModuleList(
            [nn.Conv2d(self.width, self.width, 1) for _ in range(self.n_layers)])
        self.spectral_list = nn.ModuleList(
            [SpectralConv2d(self.width, self.width, self.modes1, self.modes2) for _ in range(self.n_layers)])

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 2)

        self.to(device)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1_padding = int(round(x.shape[-1] * self.padding_frac))
        x2_padding = int(round(x.shape[-2] * self.padding_frac))
        x = F.pad(x, [0, x1_padding, 0, x2_padding])

        for k, (s, c) in enumerate(zip(self.spectral_list, self.conv_list)):

            x1 = s(x)
            x2 = c(x)
            x = x1 + x2
            if k != self.n_layers - 1:
                x = F.gelu(x)
        x = x[..., :-x1_padding, :-x2_padding]

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

# Define L2 Loss
criterion = nn.MSELoss()  # Equivalent to L2 loss

#Wandb Logging
#######################################################
# Initialize wandb
wandb.init(
    project="GeoFNO1",  # Your personal project name
    entity="namancho",  # Replace with your WandB username
    name=f"FNO_{dataset_name}_{1}",  # Optional, gives each run a unique name
    config={  # Optional configuration logging
        "learning_rate": 0.001,
        "epochs": 401,
        "batch_size": 32,
        "modes": 12,
        "width": 32,
        "output_dim": 2,
        "n_layers": 4,
        "scheduler_step_size": 100,
        "scheduler_gamma": 0.5
    }
)

# Model architecture and optimizer setup

fno_architecture = {
    "modes": wandb.config.modes,
    "width": wandb.config.width,
    "n_layers": wandb.config.n_layers,
    "retrain_fno": 42,
}
model = FNO2d(fno_architecture, device=device).to(device)

optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=wandb.config.scheduler_step_size, gamma=wandb.config.scheduler_gamma)

# Training loop
epochs = wandb.config.epochs
# output_folder = "./output_images"
# os.makedirs(output_folder, exist_ok=True)

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)  # L2 loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Evaluate
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, targets)  # L2 loss
            test_loss += loss.item()

        test_loss /= len(test_loader)

    # Scheduler step
    scheduler.step()

    # Log metrics to wandb
    wandb.log({
        "epoch": epoch-1,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "learning_rate": scheduler.get_last_lr()[0]  # Log current learning rate
    })
    print(f"Output:", epoch, train_loss, test_loss)

    # Save prediction plots every 50 epochs
    if (epoch-1) % 50 == 0:
        model.eval()

        # Take the first batch of inputs and targets from the evaluation loop
        inputs, targets = next(iter(test_loader))
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        truth = targets[0].squeeze().detach().cpu().numpy()  # Ground truth for the first test case
        pred = outputs[0].squeeze().detach().cpu().numpy()  # Model prediction for the first test case

        fig, ax = plt.subplots(2, 2, figsize=(14, 10))  # 2x2 grid for horizontal and vertical velocity
        channels = ["Horizontal Velocity (u)", "Vertical Velocity (v)"]
        cmap = "gist_ncar"  # Colormap as per your previous setup

        for ch in range(2):  # Loop over horizontal and vertical velocity channels
            vmin = min(truth[:, :, ch].min(), pred[:, :, ch].min())
            vmax = max(truth[:, :, ch].max(), pred[:, :, ch].max())

            # Plot ground truth
            im1 = ax[ch, 0].imshow(truth[:, :, ch], cmap=cmap, vmin=vmin, vmax=vmax)
            ax[ch, 0].set_title(f"Ground Truth - {channels[ch]}")
            ax[ch, 0].axis("off")
            fig.colorbar(im1, ax=ax[ch, 0], orientation="vertical", fraction=0.046, pad=0.04)

            # Plot prediction
            im2 = ax[ch, 1].imshow(pred[:, :, ch], cmap=cmap, vmin=vmin, vmax=vmax)
            ax[ch, 1].set_title(f"Prediction - {channels[ch]}")
            ax[ch, 1].axis("off")
            fig.colorbar(im2, ax=ax[ch, 1], orientation="vertical", fraction=0.046, pad=0.04)

        # Save and log the plot
        plt.tight_layout()
        plot_path = os.path.join(output_folder, f"epoch_{epoch-1}.png")
        plt.savefig(plot_path)
        if (epoch-1)%200 == 0:
            wandb.log({"Prediction Plots": wandb.Image(plot_path)})
        plt.close(fig)

wandb.finish()
