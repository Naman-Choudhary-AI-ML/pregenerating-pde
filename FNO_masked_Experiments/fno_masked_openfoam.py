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
from utils import plot_mesh_centers, load_coordinates, masked_mse_loss, plot_predictions_with_centers


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


path = '/home/vhsingh/Geo-UPSplus/FNO_masked_Experiments/CE-KH_Openfoam_Irregular/results.npy'

dataset_name = os.path.basename(os.path.dirname(path))  # This gives the folder name like 'NS-PwC'
output_folder = dataset_name
# output_folder = "NS-Sines-Poseidon-2-trial"

# Create the directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# dataset with shape (1177, 21, 128, 128, 2)

C_input = "/home/vhsingh/Geo-UPSplus/FNO_masked_Experiments/CE-KH_Openfoam_Irregular/C"
C_output = output_folder + "/C_updated"

plot_mesh_centers(C_input, C_output)
# # Load the .nc file
# with Dataset(path, mode='r') as nc:
#     data = nc.variables['velocity'][:]  # Assuming 'velocity' is the data key
data = np.load(path)  # Replace 'data.npy' with your actual data file
print("SHAPE", data.shape)

resolution = (128, 128)
missing_points = resolution[0] * resolution[1] - data[2]
print("CHECKKKKKKKKKKKKKKKKKKKKKKK for missing_points: ", missing_points)

hole_dummy_values = np.full((data.shape[0], data.shape[1], 64, data.shape[3]), -100, dtype=data.dtype)
data_with_holes = np.concatenate([data, hole_dummy_values], axis = 2) #New shape: (trajectories, timesteps 21, 16384, 2)
print("Data with dummy values added for hole centres:", data_with_holes.shape)

#Reshape
data_reshaped = data_with_holes.reshape(data.shape[0], data.shape[1], 128, 128, data.shape[3])
print("Reshaped data to grid format (Regular):", data_reshaped.shape)


#Load x and y coordinates from the new C file
x_coords, y_coords = load_coordinates(C_output)
print("Loaded x and y coordinates from the updated C file, " "x_coords:", x_coords.shape, "y_coords:", y_coords.shape)

#Expand and tile x and y coordinates
xv = np.tile(x_coords[None, None, :, :], (data_reshaped.shape[0], data_reshaped.shape[1], 1, 1))  #Shape: (trajectories, 21, 128, 128)
yv = np.tile(y_coords[None, None, :, :], (data_reshaped.shape[0], data_reshaped.shape[1], 1, 1))  #Shape: (trajectories, 21, 128, 128)

#Adding x and y as additional channels for FNO
data_with_coords = np.concatenate([data_reshaped, xv[..., None], yv[..., None]], axis=-1) #new shape: (trajectories, 21, 128, 128, 4)
print("Data with additional x and y channels:", data_with_coords.shape)

#Create masking channel
mask = np.ones((data_reshaped.shape[0], data_reshaped.shape[1], 128, 128), dtype=data_reshaped.dtype)
mask[:, :, -1, -64:] = 0 #Last 8x8 block corresponds to the hole
data_with_mask = np.concatenate([data_with_coords, mask[..., None]], axis = -1) #add mask as the last channel
print("Data with masking channel:", data_with_mask.shape)

data_tensor = torch.from_numpy(data_with_mask).float().to(device)
print("Final data tensor shape:", data_tensor.shape)

input_data = data_tensor[:, 0, :, :, :] #shape: (trajectories, 128, 128, 5), we are choosing timestep 0 as input
target_data = data_tensor[:, 1, :, :, :2] #shape: (trajectories, 128, 128, 2), we are choosing timestep 1 as target

print("input data, target data", input_data.shape, target_data.shape)

train_size = int(0.8 * len(input_data))
test_size = len(input_data) - train_size

train_input, test_input = input_data[:train_size], input_data[train_size:]
train_target, test_target = target_data[:train_size], target_data[train_size:]

print("train input, train target", train_input.shape, train_target.shape)
print("test input, test target", test_input.shape, test_target.shape)

train_dataset = TensorDataset(train_input, train_target)
test_dataset = TensorDataset(test_input, test_target)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("train loader, test loader", len(train_loader), len(test_loader))
print("Train DataLoader and Test DataLoader created")

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

        input: the solution of the coefficient function and locations (ux(x, y), uy(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=4)
        output: the solution
        output shape: (batchsize, x=s, y=s, c=2) ux and uy
        """
        self.modes1 = fno_architecture["modes"]
        self.modes2 = fno_architecture["modes"]
        self.width = fno_architecture["width"]
        self.n_layers = fno_architecture["n_layers"]
        self.retrain_fno = fno_architecture["retrain_fno"]

        torch.manual_seed(self.retrain_fno)
        # self.padding = 9 # pad the domain if input is non-periodic
        self.padding_frac = padding_frac
        self.fc0 = nn.Linear(4, self.width)  # input channel is 4: (ux(x, y), uy(x, y), x, y)

        self.conv_list = nn.ModuleList(
            [nn.Conv2d(self.width, self.width, 1) for _ in range(self.n_layers)])
        self.spectral_list = nn.ModuleList(
            [SpectralConv2d(self.width, self.width, self.modes1, self.modes2) for _ in range(self.n_layers)])

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 2)

        self.to(device)

    def forward(self, x):
        """
        Forward pass of the FNO model.
        Args:
            x: Input tensor with shape (batchsize, x, y, c=5).
               Last channel is the mask.
        Returns:
            Tensor with shape (batchsize, x, y, 2) (ux, uy).
        """
        #separate mask channel and input
        mask = x[..., -1] #shape: (batchsize, x, y)
        velocity_channels = x[..., :2] #shape: (batchsize, x, y, c=2) ux uy channels
        xy_channels = x[..., 2:4] #shape: (batchsize, x, y, c=2) x y channels

        #applying mask to the velocity channels
        velocity_channels = velocity_channels * mask.unsqueeze(-1) #shape: (batchsize, x, y, c=2)
        
        #recombine the channels
        x = torch.cat([velocity_channels, xy_channels], dim=-1) #shape: (batchsize, x, y, c=4)
        
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

        #applying mask to the output to ensure no values are predicted in the hole region
        x = x * mask.unsqueeze(-1) #shape: (batchsize, x, y, 2)
        # print("Output of model shape:", x.shape)
        return x


#Wandb Logging
#######################################################
# Initialize wandb
wandb.init(
    project="Research",  # Your personal project name
    entity="ved100-carnegie-mellon-university",  # Replace with your WandB username
    name=f"FNO_masked_{dataset_name}_{200}_BL",  # Optional, gives each run a unique name
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
        #extracting mask from input
        mask = inputs[..., -1]

        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        #compute masked loss
        loss = masked_mse_loss(outputs, targets, mask)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Evaluate
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            mask = inputs[..., -1]
            inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
            outputs = model(inputs)

            loss = masked_mse_loss(outputs, targets, mask)
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
        inputs, targets = next(iter(test_loader))
        mask = inputs[..., -1].to(device)
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        plot_predictions_with_centers(outputs, targets, mask, C_output, output_folder, epoch-1)
        
        # if (epoch-1)%200 == 0:
        #     wandb.log({"Prediction Plots": wandb.Image(plot_path)})

wandb.finish()
