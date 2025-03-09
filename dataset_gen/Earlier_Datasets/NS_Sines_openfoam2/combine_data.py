import numpy as np
import os

# Configuration
dataset_dir = "./NS_Sines_dataset"  # Directory where .npy files are stored
output_file = "/home/namancho/datasets/NS-Sines-Openfoam/NS_Sines_openfoam2.npy"  # Combined reshaped dataset output file
nx, ny = 128, 128  # Grid dimensions
channels = 2       # Number of velocity components (Ux, Uy)
timesteps = 21     # Number of timesteps

def combine_and_reshape_trajectories(dataset_dir, nx, ny, timesteps, channels):
    """
    Combine and reshape all trajectory .npy files into a single 5D dataset.
    """
    files = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".npy")])
    num_trajectories = len(files)

    # Initialize combined dataset
    combined_data = np.zeros((num_trajectories, timesteps, nx, ny, channels))

    # Load, reshape, and combine all trajectory files
    for i, file in enumerate(files):
        print(f"Processing trajectory {i + 1}/{num_trajectories}: {file}")
        trajectory_data = np.load(os.path.join(dataset_dir, file))  # Load individual .npy file
        reshaped_data = trajectory_data.reshape((timesteps, nx, ny, channels))  # Reshape to 4D
        combined_data[i] = reshaped_data

    return combined_data

def main():
    print("Combining and reshaping trajectory .npy files...")
    combined_data = combine_and_reshape_trajectories(dataset_dir, nx, ny, timesteps, channels)
    print(f"Combined dataset shape: {combined_data.shape}")

    print(f"Saving combined dataset to {output_file}...")
    np.save(output_file, combined_data)
    print("Dataset saved successfully!")

if __name__ == "__main__":
    main()
