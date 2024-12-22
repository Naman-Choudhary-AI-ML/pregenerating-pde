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


path = '/home/namancho/datasets/NS-Gauss-Irr-Openfoam/openfoam.npy'

dataset_name = os.path.basename(os.path.dirname(path))  # This gives the folder name like 'NS-PwC'
output_folder = dataset_name
# output_folder = "NS-Sines-Poseidon-2-trial"

# Create the directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# dataset with shape (1177, 21, 128, 128, 2)

C_input = "./C"
C_output = output_folder + "/C_updated"


plot_mesh_centers(C_input, C_output)
# # Load the .nc file
# with Dataset(path, mode='r') as nc:
#     data = nc.variables['velocity'][:]  # Assuming 'velocity' is the data key
data = np.load(path)  # Replace 'data.npy' with your actual data file
print("SHAPE", data.shape)

hole_dummy_values = np.full((data.shape[0], data.shape[1], 64, data.shape[3]), -100, dtype=data.dtype)
data_with_holes = np.concatenate([data, hole_dummy_values], axis = 2) #New shape: (trajectories, timesteps 21, 16384, 2)
print("Data with dummy values added for hole centres:", data_with_holes.shape)

#Reshape
data_reshaped = data_with_holes.reshape(data.shape[0], data.shape[1], 128, 128, data.shape[3])
print("Reshaped data to grid format (Regular):", data_reshaped.shape)
print("Last 8x8 block after reshaping (raw values):")
print(data_reshaped[0, 0, 120:128, 120:128, :])  # First trajectory, first timestep


#Load x and y coordinates from the new C file
x_coords, y_coords = load_coordinates(C_output)
print("Loaded x and y coordinates from the updated C file, " "x_coords:", x_coords.shape, "y_coords:", y_coords.shape)
# Check the last 8x8 block
print("X coordinates (120:128, 120:128):")
print(x_coords[-1, -64:])

print("Y coordinates (120:128, 120:128):")
print(y_coords[-1, -64:])        

# #Expand and tile x and y coordinates
# xv = np.tile(x_coords[None, None, :, :], (data_reshaped.shape[0], data_reshaped.shape[1], 1, 1))  #Shape: (trajectories, 21, 128, 128)
# yv = np.tile(y_coords[None, None, :, :], (data_reshaped.shape[0], data_reshaped.shape[1], 1, 1))  #Shape: (trajectories, 21, 128, 128)

# #Adding x and y as additional channels for FNO
# data_with_coords = np.concatenate([data_reshaped, xv[..., None], yv[..., None]], axis=-1) #new shape: (trajectories, 21, 128, 128, 4)
# print("Data with additional x and y channels:", data_with_coords.shape)

# #Create masking channel
# mask = np.ones((data_reshaped.shape[0], data_reshaped.shape[1], 128, 128), dtype=data_reshaped.dtype)
# mask[:, :, 120:128, 120:128] = 0 #Last 8x8 block corresponds to the hole
# data_with_mask = np.concatenate([data_with_coords, mask[..., None]], axis = -1) #add mask as the last channel
# print("Data with masking channel:", data_with_mask.shape)

# data_tensor = torch.from_numpy(data_with_mask).float().to(device)
# print("Final data tensor shape:", data_tensor.shape)