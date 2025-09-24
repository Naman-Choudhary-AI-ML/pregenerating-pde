import os
import numpy as np
import subprocess
from scipy.fftpack import dst, idst

# Configuration
num_trajectories = 1177  # Number of trajectories
timesteps = 21           # Number of timesteps (0 to 1)
output_dir = "NS_BB_dataset"  # Directory to store dataset
base_case_dir = "./"       # Base OpenFOAM case folder
channels = 2
final_output_dir = "/home/namancho/datasets/NS-BB-Openfoam"  # Combined reshaped dataset output file
dataset_dir = "./NS_BB_dataset"
os.makedirs(final_output_dir,exist_ok=True)

# Create output directory
os.makedirs(output_dir, exist_ok=True)

def extract_points(points_path):
    """
    Extract points from the points file.
    """
    with open(points_path, "r") as file:
        lines = file.readlines()

    # Extract the number of points and their data
    num_points = int(lines[19].strip())  # Line where the number of points is stored (adjust if needed)
    start_idx = 21  # Line index where point data begins (adjust if needed)
    points_data = [
        list(map(float, line.strip("()\n").split()))
        for line in lines[start_idx:start_idx + num_points]
        if len(line.strip("()\n").split()) == 3
    ]

    # Convert to NumPy array
    points = np.array(points_data)

    # Filter points for z = 0
    filtered_points = points[points[:, 2] == 0][:, :2]  # Only x and y coordinates where z = 0

    return filtered_points


def compute_cell_centers(points):
    """
    Compute cell centers for a structured 2D grid.
    """
    # Reshape points into a grid
    num_points_side = int(np.sqrt(len(points)))  # Assumes square grid
    if num_points_side * num_points_side != len(points):
        raise ValueError("Mesh points do not form a square grid!")

    grid_points = points.reshape((num_points_side, num_points_side, 2))
    
    # Compute centers by averaging adjacent points
    cell_centers = []
    for i in range(num_points_side - 1):
        for j in range(num_points_side - 1):
            center_x = (grid_points[i, j, 0] + grid_points[i + 1, j, 0] +
                        grid_points[i, j + 1, 0] + grid_points[i + 1, j + 1, 0]) / 4
            center_y = (grid_points[i, j, 1] + grid_points[i + 1, j, 1] +
                        grid_points[i, j + 1, 1] + grid_points[i + 1, j + 1, 1]) / 4
            cell_centers.append([center_x, center_y])

    return np.array(cell_centers)

def generate_brownian_bridge_velocity(nx, ny, max_velocity=1.0):
    """
    Generate Brownian Bridge velocity fields for a structured 2D grid.

    Parameters:
        nx, ny: int - Grid dimensions (number of points in x and y directions).
        max_velocity: float - Maximum allowable velocity magnitude (for stability).
    
    Returns:
        U: np.ndarray - Velocity field in OpenFOAM-compatible format.
    """
    # Grid setup
    x = np.linspace(0, 2 * np.pi, nx)  # 2π periodic domain
    y = np.linspace(0, 2 * np.pi, ny)
    X, Y = np.meshgrid(x, y)

    # Initialize Fourier space variables
    kx = np.fft.fftfreq(nx, d=1 / nx) * 2 * np.pi  # Fourier modes in x
    ky = np.fft.fftfreq(ny, d=1 / ny) * 2 * np.pi  # Fourier modes in y
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2
    K2[K2 == 0] = 1  # Avoid division by zero for the DC component

    # Initialize velocity in Fourier space
    alpha = np.random.uniform(-1, 1, (2, 2, 2))  # Coefficients for sin/cos combinations
    W_hat = np.zeros((nx, ny), dtype=complex)  # Fourier space vorticity

    # Fill Fourier modes
    for m in range(2):  # sin or cos for x
        for n in range(2):  # sin or cos for y
            for l in range(2):  # sin or cos for z
                sci = np.sin if m == 0 else np.cos
                scj = np.sin if n == 0 else np.cos
                scl = np.sin if l == 0 else np.cos
                W_hat += (
                    alpha[m, n, l]
                    / np.sqrt(K2**(3 / 2))
                    * sci(X)
                    * scj(Y)
                    * scl(X + Y)  # Example combination
                )

    # Inverse Fourier Transform to get the velocity in real space
    u_x = np.fft.ifft2(-1j * KY * W_hat).real
    u_y = np.fft.ifft2(1j * KX * W_hat).real

    # Normalize to the desired statistics
    target_mean_std = [(1.203958e-13, 0.148372), (9.570576e-12, 0.130903)]  # (mean, std) for (u_x, u_y)
    for i, (data, (target_mean, target_std)) in enumerate(zip([u_x, u_y], target_mean_std)):
        data = (data - np.mean(data)) / np.std(data) * target_std + target_mean
        if i == 0:
            u_x = data
        else:
            u_y = data

    # Flatten fields into OpenFOAM-compatible format
    U = np.zeros((nx * ny, 3))  # Ux, Uy, Uz
    U[:, 0] = u_x.flatten()
    U[:, 1] = u_y.flatten()
    return U




# Update the U file with the new velocity field
def update_U_file(nx, ny, velocity, case_dir):
    u_file_path = os.path.join(case_dir, "0", "U")
    with open(u_file_path, "w") as f:
        # Write FoamFile header
        f.write("FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       volVectorField;\n    object      U;\n}\n")
        f.write("dimensions      [0 1 -1 0 0 0 0];\n\n")
        f.write("internalField   nonuniform List<vector>\n")
        f.write(f"{nx * ny}\n(\n")
        for vel in velocity:
            f.write(f"({vel[0]} {vel[1]} {vel[2]})\n")
        f.write(");\n\n")
        # Write boundary conditions
        f.write("boundaryField\n{\n")
        f.write("    left\n    {\n        type cyclic;\n    }\n")
        f.write("    right\n    {\n        type cyclic;\n    }\n")
        f.write("    bottom\n    {\n        type cyclic;\n    }\n")
        f.write("    top\n    {\n        type cyclic;\n    }\n")
        f.write("    frontAndBack\n    {\n        type empty;\n    }\n}\n")

# Function to run OpenFOAM simulation
def run_simulation(case_dir):
    subprocess.run(["foamCleanTutorials"], cwd=case_dir)
    subprocess.run(["blockMesh"], cwd=case_dir)
    subprocess.run(["icoFoam"], cwd=case_dir)

def extract_velocity_data(case_dir, timesteps, points):
    """
    Extract velocity data for all timesteps from the OpenFOAM case.
    """
    data = np.zeros((timesteps, len(points), 2))  # (time, points, channels)

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
            end_index = start_index + len(points)
        except ValueError:
            raise ValueError(f"Could not locate velocity data in {u_file}")

        # Extract velocity values
        velocities = lines[start_index:end_index]
        for j, vel in enumerate(velocities):
            ux, uy, _ = map(float, vel.strip("()\n").split())
            data[i, j, 0] = ux
            data[i, j, 1] = uy

    return data

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

# Main dataset generation loop
for traj in range(num_trajectories):
    print(f"Generating trajectory {traj + 1}/{num_trajectories}...")

    # Step 1: Generate Sinusoidal Velocity (Fixed Grid)
    nx, ny = 128, 128
    # velocity = generate_sinusoidal_velocity(nx, ny)
    velocity = generate_brownian_bridge_velocity(nx, ny)

    # Step 2: Update U File
    update_U_file(nx, ny, velocity, base_case_dir)

    # Step 3: Run Simulation
    run_simulation(base_case_dir)

    # Step 4: Extract Points from PolyMesh
    mesh_points = extract_points(os.path.join(base_case_dir, "constant/polyMesh/points"))

    # Step 5: Compute Cell Centers for Final Dataset
    cell_centers = compute_cell_centers(mesh_points)

    # Step 6: Extract Velocity Data at Cell Centers
    trajectory_data = extract_velocity_data(base_case_dir, timesteps, cell_centers)

    # Step 7: Save Trajectory Data
    np.save(os.path.join(output_dir, f"trajectory_{traj:04d}.npy"), trajectory_data)

print(f"Dataset generation complete. Data saved in {output_dir}")

# Combine individual trajectory files into a single dataset
print("Combining and reshaping trajectory .npy files...")
combined_data = combine_and_reshape_trajectories(dataset_dir, nx, ny, timesteps, channels)
print(f"Combined dataset shape: {combined_data.shape}")

# Save the combined dataset
combined_output_file = os.path.join(final_output_dir, "openfoam.npy")
print(f"Saving combined dataset to {combined_output_file}...")
np.save(combined_output_file, combined_data)
print("Combined dataset saved successfully!")

# Remove individual .npy files
import glob
for npy_file in glob.glob(os.path.join(dataset_dir, "*.npy")):
    os.remove(npy_file)
print("Individual .npy files deleted.")

