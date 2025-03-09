import os
import numpy as np
import subprocess
from scipy.fftpack import dst, idst

# Configuration
num_trajectories = 800  # Number of trajectories
timesteps = 21           # Number of timesteps (0 to 1)
output_dir = "NS_Gauss_irr_dataset"  # Directory to store dataset
base_case_dir = "./"       # Base OpenFOAM case folder
channels = 2
final_output_dir = "/home/namancho/datasets/NS-Gauss-Irr-Openfoam"  # Combined reshaped dataset output file
dataset_dir = "./NS_Gauss_irr_dataset"
os.makedirs(final_output_dir,exist_ok=True)
nx, ny = 128, 128
hole_start, hole_end = 60, 68

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Step 1: Command to generate C file
def generate_cell_centers(case_dir):
    """
    Generate cell centers using OpenFOAM's postProcess utility.
    """
    subprocess.run(["blockMesh"], cwd=case_dir)
    subprocess.run(["postProcess", "-func", "writeCellCentres"], cwd=case_dir)

def parse_c_file(file_path):
    """
    Parse the openfoam C file to get correct x, y coordinates of cell centres used by openfoam for solving
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    start_idx = None
    num_centres = None
    for i, line in enumerate(lines):
        if "internalField" in line:
            num_centres = int(lines[i + 1].strip())
            start_idx = i + 3
            break

    if start_idx is None or num_centres is None:
        raise ValueError("Could not find 'internalField' or num of cell centres in the file")

    coordinates = []
    for line in lines[start_idx : start_idx + num_centres]:
        if "(" in line and ")" in line:
            x, y, _ = map(float, line.strip("()\n").split())
            coordinates.append([x, y])
    return np.array(coordinates)

def generate_gaussian_velocity(cell_centers, num_gaussians=100, nx=128, ny=128, max_velocity=1.5):
    """
    Generate initial velocity fields for the NS-Gauss dataset using vorticity.

    Parameters:
        cell_centers: np.ndarray - (16320, 2) array of x, y coordinates of valid cells.
        num_gaussians: int - Number of Gaussian components in vorticity.
        max_velocity: float - Maximum allowable velocity magnitude (to scale).
        nx, ny: int - Dimensions of the pseudo-regular grid.

    Returns:
        U: np.ndarray - (16320, 3) array of velocity vectors (Ux, Uy, Uz).
    """
    # Step 1: Create a pseudo-regular grid
    omega_grid = np.zeros((nx, ny))
    grid_map = np.full((nx, ny), -1)  # Map of indices for irregular cells (-1 = hole)
    valid_indices = []
    for idx, (x, y) in enumerate(cell_centers):
        i = int(np.floor(x * nx))
        j = int(np.floor(y * ny))
        omega_grid[i, j] = 0  # Initialize vorticity
        grid_map[i, j] = idx
        valid_indices.append((i, j))

    # Step 2: Add Gaussian components to vorticity
    for _ in range(num_gaussians):
        alpha = np.random.uniform(-1, 1)  # Amplitude
        sigma = np.random.uniform(0.01, 0.1)  # Width
        x_i = np.random.uniform(0, 1)  # Center x
        y_i = np.random.uniform(0, 1)  # Center y
        for i, j in valid_indices:
            x = i / nx
            y = j / ny
            omega_grid[i, j] += alpha / sigma * np.exp(-((x - x_i)**2 + (y - y_i)**2) / (2 * sigma**2))

    # Step 3: Solve for stream function (ψ) using DST
    omega_dst = dst(dst(omega_grid, type=1, axis=0), type=1, axis=1)
    kx = np.fft.fftfreq(nx, d=1 / nx)**2
    ky = np.fft.fftfreq(ny, d=1 / ny)**2
    denominator = kx[:, None] + ky[None, :]
    denominator[0, 0] = 1  # Prevent division by zero
    psi_dst = omega_dst / denominator
    psi_grid = idst(idst(psi_dst, type=1, axis=0), type=1, axis=1)

    # Step 4: Extract ψ values for irregular cells and compute velocities
    U = np.zeros((len(cell_centers), 3))  # Initialize velocity field (Ux, Uy, Uz)
    for i, j in valid_indices:
        idx = grid_map[i, j]
        u_x = np.gradient(psi_grid, axis=1)[i, j]  # ∂ψ/∂y
        u_y = -np.gradient(psi_grid, axis=0)[i, j]  # -∂ψ/∂x
        U[idx, 0] = u_x
        U[idx, 1] = u_y
    # Step 5: Normalize velocities based on magnitude
    velocity_magnitude = np.sqrt(U[:, 0]**2 + U[:, 1]**2)
    max_magnitude = velocity_magnitude.max()

    if max_magnitude > max_velocity:
        scale_factor = max_velocity / max_magnitude
        U[:, 0] *= scale_factor
        U[:, 1] *= scale_factor


    return U


def update_U_file(velocity, case_dir):
    """
    Updates the U file with the new velocity field.

    Parameters:
    - velocity: np.ndarray of shape (num_valid_cells, 3), the velocity field
    - case_dir: str, the directory of the OpenFOAM case
    """
    num_cells = len(velocity)  # Total number of valid cells
    u_file_path = os.path.join(case_dir, "0", "U")
    
    with open(u_file_path, "w") as f:
        # Write FoamFile header
        f.write("FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       volVectorField;\n    object      U;\n}\n")
        f.write("dimensions      [0 1 -1 0 0 0 0];\n\n")
        
        # Write internalField
        f.write("internalField   nonuniform List<vector>\n")
        f.write(f"{num_cells}\n(\n")
        for vel in velocity:
            f.write(f"({vel[0]} {vel[1]} {vel[2]})\n")
        f.write(");\n\n")
        
        # Write boundary conditions
        f.write("boundaryField\n{\n")
        f.write("    left\n    {\n        type cyclic;\n    }\n")
        f.write("    right\n    {\n        type cyclic;\n    }\n")
        f.write("    bottom\n    {\n        type cyclic;\n    }\n")
        f.write("    top\n    {\n        type cyclic;\n    }\n")
        f.write("    frontAndBack\n    {\n        type empty;\n    }\n")
        f.write("    holeWalls\n    {\n        type noSlip;\n    }\n")
        f.write("}\n")


# Function to run OpenFOAM simulation
def run_simulation(case_dir):
    subprocess.run(["foamCleanTutorials"], cwd=case_dir)
    subprocess.run(["blockMesh"], cwd=case_dir)
    subprocess.run(["icoFoam"], cwd=case_dir)

def extract_velocity_data(case_dir, timesteps, points_len):
    """
    Extract velocity data for all timesteps from the OpenFOAM case.
    """
    data = np.zeros((timesteps, points_len, 2))  # (time, points, channels)

    for i, time in enumerate(np.linspace(0, 1, timesteps)):
        time_dir = os.path.join(case_dir, f"{time:.6g}")
        if not os.path.exists(time_dir):
            raise FileNotFoundError(f"Timestep {time:.6g} missing in {case_dir}")
        
        u_file = os.path.join(time_dir, "U")
        with open(u_file, "r") as f:
            lines = f.readlines()
        
        # Locate the start of velocity data
        try:
            start_index = lines.index("(\n") + 1
            end_index = start_index + points_len
        except ValueError:
            raise ValueError(f"Could not locate velocity data in {u_file}")

        # Extract velocity values
        velocities = lines[start_index:end_index]
        for j, vel in enumerate(velocities):
            ux, uy, _ = map(float, vel.strip("()\n").split())
            data[i, j, 0] = ux
            data[i, j, 1] = uy

    return data

def combine_trajectories_flat(dataset_dir, timesteps, num_cells, channels):
    """
    Combine all trajectory .npy files into a single 4D dataset without reshaping.
    """
    files = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".npy")])
    num_trajectories = len(files)

    # Initialize combined dataset
    combined_data = np.zeros((num_trajectories, timesteps, num_cells, channels))

    # Load and combine all trajectory files
    for i, file in enumerate(files):
        print(f"Processing trajectory {i + 1}/{num_trajectories}: {file}")
        trajectory_data = np.load(os.path.join(dataset_dir, file))  # Load individual .npy file
        # print(trajectory_data.shape, combined_data[i].shape)
        combined_data[i] = trajectory_data  # Directly load into combined dataset

    return combined_data

generate_cell_centers(base_case_dir)
centres = parse_c_file("./0/C")

# Main dataset generation loop
for traj in range(num_trajectories):
    print(f"Generating trajectory {traj + 1}/{num_trajectories}...")

    # Step 1: Generate Gaussian Velocity Field (with hole)
    velocity = generate_gaussian_velocity(centres, num_gaussians=100, nx=128, ny=128)

    # Step 2: Update U File
    update_U_file(velocity, base_case_dir)

    # Step 3: Run Simulation
    run_simulation(base_case_dir)

    # Step 4: Extract Velocity Data
    trajectory_data = extract_velocity_data(base_case_dir, timesteps, len(centres))

    # Step 5: Save Trajectory Data
    np.save(os.path.join(output_dir, f"trajectory_{traj:04d}.npy"), trajectory_data)


print(f"Dataset generation complete. Data saved in {output_dir}")

# Combine individual trajectory files into a single dataset
print("Combining and reshaping trajectory .npy files...")
combined_data = combine_trajectories_flat(dataset_dir, timesteps, len(centres), channels)
print(f"Combined dataset shape: {combined_data.shape}")

# Save the combined dataset
combined_output_file = os.path.join(final_output_dir, "openfoam800.npy")
print(f"Saving combined dataset to {combined_output_file}...")
np.save(combined_output_file, combined_data)
print("Combined dataset saved successfully!")

# Remove individual .npy files
import glob
for npy_file in glob.glob(os.path.join(dataset_dir, "*.npy")):
    os.remove(npy_file)
print("Individual .npy files deleted.")

