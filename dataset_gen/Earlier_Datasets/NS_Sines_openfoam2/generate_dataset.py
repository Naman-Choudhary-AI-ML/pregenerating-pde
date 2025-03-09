import os
import numpy as np
import subprocess

# Configuration
num_trajectories = 1197  # Number of trajectories
timesteps = 21           # Number of timesteps (0 to 1)
output_dir = "NS_Sines_dataset"  # Directory to store dataset
base_case_dir = "./"       # Base OpenFOAM case folder

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

# def generate_sinusoidal_velocity(nx, ny, num_modes=10, max_velocity=1.0):
#     """
#     Generate sinusoidal velocity fields for a structured 2D grid.
#     The velocities are scaled to ensure stability during simulation.
    
#     Parameters:
#         nx, ny: int - Grid size (number of points in x and y directions).
#         num_modes: int - Number of sinusoidal modes.
#         max_velocity: float - Maximum allowable velocity magnitude (to control Courant number).
    
#     Returns:
#         U: np.ndarray - Velocity field in OpenFOAM-compatible format.
#     """
#     p = num_modes  # Number of modes
#     alpha = np.random.uniform(-1, 1, (p, p))
#     beta = np.random.uniform(0, 2 * np.pi, (p, p))
#     gamma = np.random.uniform(0, 2 * np.pi, (p, p))

#     # Generate structured grid
#     x = np.linspace(0, 1, nx)
#     y = np.linspace(0, 1, ny)
#     X, Y = np.meshgrid(x, y)

#     # Initialize velocity fields
#     u = np.zeros_like(X)
#     v = np.zeros_like(Y)

#     # Calculate velocity field
#     for i in range(p):
#         for j in range(p):
#             u += alpha[i, j] * np.sin(2 * np.pi * i * X + beta[i, j]) * np.sin(2 * np.pi * j * Y + gamma[i, j])
#             v += alpha[i, j] * np.cos(2 * np.pi * i * X + beta[i, j]) * np.cos(2 * np.pi * j * Y + gamma[i, j])

#     # Normalize the velocity to ensure stability
#     max_magnitude = np.sqrt(np.max(u**2 + v**2))
#     if max_magnitude > max_velocity:
#         scaling_factor = max_velocity / max_magnitude
#         u *= scaling_factor
#         v *= scaling_factor

#     # Flatten fields into OpenFOAM-compatible format
#     U = np.zeros((nx * ny, 3))  # Ux, Uy, Uz
#     U[:, 0] = u.flatten()
#     U[:, 1] = v.flatten()
#     return U
def generate_sinusoidal_velocity(nx, ny, num_modes=10, max_velocity=1.0):
    """
    Generate sinusoidal velocity fields for a structured 2D grid, aligning with the paper.
    
    Parameters:
        nx, ny: int - Grid size (number of points in x and y directions).
        num_modes: int - Number of sinusoidal modes (default: 10, as per the paper).
        max_velocity: float - Maximum allowable velocity magnitude (to control Courant number).
        seed: int - Random seed for reproducibility.
    
    Returns:
        U: np.ndarray - Velocity field in OpenFOAM-compatible format.
    """
    np.random.seed(42)
    
    p = num_modes
    alpha = np.random.uniform(-1, 1, (p, p))
    beta = np.random.uniform(0, 2 * np.pi, (p, p))
    gamma = np.random.uniform(0, 2 * np.pi, (p, p))

    # Structured grid
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    # Initialize velocity fields
    u = np.zeros_like(X)
    v = np.zeros_like(Y)

    # Calculate velocity fields with normalization
    for i in range(p):
        for j in range(p):
            normalization = 1 / np.sqrt(2 * np.pi * (i + j + 1))  # +1 to avoid division by zero
            u += alpha[i, j] * normalization * np.sin(2 * np.pi * i * X + beta[i, j]) * np.sin(2 * np.pi * j * Y + gamma[i, j])
            v += alpha[i, j] * normalization * np.cos(2 * np.pi * i * X + beta[i, j]) * np.cos(2 * np.pi * j * Y + gamma[i, j])

    # # Scale to maximum velocity
    # max_magnitude = np.sqrt(np.max(u**2 + v**2))
    # if max_magnitude > max_velocity:
    #     scaling_factor = max_velocity / max_magnitude
    #     u *= scaling_factor
    #     v *= scaling_factor

    # Flatten fields into OpenFOAM-compatible format
    U = np.zeros((nx * ny, 3))  # Ux, Uy, Uz
    U[:, 0] = u.flatten()
    U[:, 1] = v.flatten()
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


# Main dataset generation loop
for traj in range(num_trajectories):
    print(f"Generating trajectory {traj + 1}/{num_trajectories}...")

    # Step 1: Generate Sinusoidal Velocity (Fixed Grid)
    nx, ny = 128, 128
    velocity = generate_sinusoidal_velocity(nx, ny)

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
