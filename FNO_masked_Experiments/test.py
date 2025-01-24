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
from utils import plot_mesh_centers, load_coordinates, masked_mse_loss, plot_predictions_with_centers, reorder_data_to_row_by_row


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


path = '/home/namancho/datasets/NS-Gauss-Irr-Openfoam/openfoam800.npy'

dataset_name = os.path.basename(os.path.dirname(path))  # This gives the folder name like 'NS-PwC'
output_folder = dataset_name
# output_folder = "NS-Sines-Poseidon-2-trial"

# Create the directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# dataset with shape (1177, 21, 128, 128, 2)

C_input = "/home/namancho/Geo-UPSplus/FNO_masked_Experiments/C"
C_output = output_folder + "/C_updated"



plot_mesh_centers(C_input, C_output)
# # Load the .nc file
# with Dataset(path, mode='r') as nc:
#     data = nc.variables['velocity'][:]  # Assuming 'velocity' is the data key
data = np.load(path)  # Replace 'data.npy' with your actual data file
print("SHAPE", data.shape)


hole_dummy_values = np.full((data.shape[0], data.shape[1], 64, data.shape[3]), 0, dtype=data.dtype)
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

data_numpy = data_tensor.cpu().numpy()
# Extract x and y coordinates from the data channels
x_coords = data_numpy[0, 0, :, :, 2]
y_coords = data_numpy[0, 0, :, :, 3]

print("xcoords shape:", x_coords.shape)
print("ycoords", y_coords.shape)
print("Flattened x_coords size:", x_coords.flatten().size)  # Expect 16384
print("Flattened y_coords size:", y_coords.flatten().size)  # Expect 16384
print("data_numpy shape:", data_numpy.shape)  # Expect (200, 21, 128, 128, channels)

# Reorder the data
data_reordered = reorder_data_to_row_by_row(data_numpy, x_coords, y_coords, save_path="/home/namancho/datasets/NS-Gauss-Irr-Openfoam/reordered_data.npy")

reloaded_data = np.load("/home/namancho/datasets/NS-Gauss-Irr-Openfoam/reordered_data.npy")
print("Reordered data shape:", reloaded_data.shape)

data_sliced = data_reordered[..., [0, 1, 4]] #shape: 200, 21, 128, 128, 3

sliced_data_path = '/home/namancho/datasets/NS-Gauss-Irr-Openfoam/sliced_data1200.npy'
np.save(sliced_data_path, data_sliced)

print(f"Sliced dataset saved to {sliced_data_path} and shape is {data_sliced.shape}")

resolution = 128
# Generate grid coordinates for plotting
x_coords = np.linspace(0, 1, resolution)  # Normalized x-coordinates
y_coords = np.linspace(0, 1, resolution)  # Normalized y-coordinates
xv, yv = np.meshgrid(x_coords, y_coords)

# Extract data for trajectory 0 and timestep 0
trajectory_idx = 0
timestep_idx = 0
horizontal_velocity = data_reordered[trajectory_idx, timestep_idx, :, :, 0]
vertical_velocity = data_reordered[trajectory_idx, timestep_idx, :, :, 1]
mask = data_reordered[:, :, :, :, 4]
mask_valid = mask[trajectory_idx, timestep_idx, :, :]

# Flatten arrays and apply mask
xv_flat = xv.flatten()
yv_flat = yv.flatten()
valid_indices = mask_valid.flatten() == 1
xv_valid = xv_flat[valid_indices]
yv_valid = yv_flat[valid_indices]
horizontal_valid = horizontal_velocity.flatten()[valid_indices]
vertical_valid = vertical_velocity.flatten()[valid_indices]

# Calculate color scale range based on valid values
vmin_h = horizontal_valid.min()
vmax_h = horizontal_valid.max()
vmin_v = vertical_valid.min()
vmax_v = vertical_valid.max()

# Plot horizontal and vertical velocity
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
cmap = "gist_ncar"

# Horizontal velocity
sc1 = ax[0].scatter(xv_valid, yv_valid, c=horizontal_valid, cmap=cmap, vmin=vmin_h, vmax=vmax_h, s=10)
ax[0].set_title("Horizontal Velocity (u)")
ax[0].axis("off")
plt.colorbar(sc1, ax=ax[0], orientation="vertical")

# Vertical velocity
sc2 = ax[1].scatter(xv_valid, yv_valid, c=vertical_valid, cmap=cmap, vmin=vmin_v, vmax=vmax_v, s=10)
ax[1].set_title("Vertical Velocity (v)")
ax[1].axis("off")
plt.colorbar(sc2, ax=ax[1], orientation="vertical")

plt.tight_layout()
plt.show()
plot_path=os.path.join("./test2.png")
plt.savefig(plot_path)
