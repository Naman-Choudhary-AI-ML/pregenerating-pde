import os
import glob
import numpy as np
import subprocess
from scipy.fftpack import dst, idst
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt

# ================= Configuration =================
num_trajectories = 1024     # Number of trajectories to simulate
timesteps = 21              # Total timesteps (including t=0)
# The simulation extracts 3 channels: Ux, Uy, p.
# We add a fourth channel for hole info (0: cell exists, 1: hole).
final_channels = 4

# Directories (change as needed)
output_dir = "/data/user_data/namancho/LDC_reg"                  # Where individual trajectory .npy files are saved
final_output_dir = "/data/user_data/namancho/LDC_Regular_2048"  # Where the combined dataset is saved
dataset_dir = output_dir                    # For combining, use the same directory
base_case_dir = "./"                        # Base OpenFOAM case directory

# Physical and simulation parameters
Re_min = 100
Re_max = 10000
mean_re = 5000
std_re = 2000
nu = 1.5e-5    # Kinematic viscosity (m^2/s)
L = 2.0        # Characteristic length (m)

# Final grid dimensions – this is the fixed output grid
# For both regular and irregular cases, we assume (for example) a 128×128 grid.
final_grid_shape = (128, 128)  # (n_rows, n_cols)

# Create necessary directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(final_output_dir, exist_ok=True)

# Predefine Reynolds numbers and compute corresponding U (lid velocity)
np.random.seed(42)  # reproducibility
# Generate Reynolds numbers using np.random.normal
Re_values = np.random.normal(mean_re, std_re, num_trajectories)

# Clip the values to be within [Re_min, Re_max]
Re_values = np.clip(Re_values, Re_min, Re_max)

U_values = (Re_values * nu) / L

# ================= Mesh and Simulation Utilities =================

def generate_cell_centers(case_dir):
    """
    Generate cell centers using OpenFOAM's postProcess utility.
    Calls blockMesh (if needed) and then writes the cell centers file.
    """
    subprocess.run(["blockMesh"], cwd=case_dir)
    subprocess.run(["postProcess", "-func", "writeCellCentres"], cwd=case_dir)

def parse_c_file(file_path):
    """
    Parse the OpenFOAM cell centers file (named "C") to extract (x,y) coordinates.
    Expects the file to contain an 'internalField' entry.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    start_idx = None
    num_centres = None
    for i, line in enumerate(lines):
        if "internalField" in line:
            num_centres = int(lines[i+1].strip())
            start_idx = i + 3
            break

    if start_idx is None or num_centres is None:
        raise ValueError("Could not find 'internalField' or the number of cell centres in the file")

    coordinates = []
    for line in lines[start_idx:start_idx+num_centres]:
        if "(" in line and ")" in line:
            x, y, _ = map(float, line.strip("()\n").split())
            coordinates.append([x, y])
    return np.array(coordinates)

def update_U_file(case_dir, U_value):
    """
    Update the OpenFOAM U file to set the movingWall velocity.
    Searches for the 'movingWall' section and changes its value.
    """
    u_file_path = os.path.join(case_dir, "0", "U")
    with open(u_file_path, "r") as file:
        lines = file.readlines()
    
    # Locate the movingWall section and update the uniform velocity value.
    for i, line in enumerate(lines):
        if "movingWall" in line:
            # Advance until we find the 'value' line
            while "value" not in lines[i]:
                i += 1
            lines[i] = f'        value           uniform ({U_value} 0 0);\n'
            break

    with open(u_file_path, "w") as file:
        file.writelines(lines)

def run_simulation(case_dir):
    """
    Run the OpenFOAM simulation commands.
    For regular geometry, we use the standard blockMesh routine.
    """
    subprocess.run(["foamCleanTutorials"], cwd=case_dir)
    subprocess.run(["blockMesh"], cwd=case_dir)
    subprocess.run(["icoFoam"], cwd=case_dir)

def extract_velocity_data(case_dir, timesteps, cell_centers):
    """
    Extract U (velocity) and p (pressure) data from the OpenFOAM time directories.
    Returns a NumPy array of shape (timesteps, num_cells, 3).
    """
    num_cells = len(cell_centers)
    data = np.zeros((timesteps, num_cells, 3))  # channels: Ux, Uy, p
    
    # List available time directories (assuming integer names > 0)
    available_times = sorted([int(t) for t in os.listdir(case_dir)
                              if t.isdigit() and int(t) > 0])
    if len(available_times) == 0:
        raise RuntimeError(f"No OpenFOAM time directories found in {case_dir}")

    # Choose (timesteps-1) equally spaced times (t=0 is manually set)
    selected_times = np.linspace(available_times[0], available_times[-1], timesteps - 1, dtype=int)

    # t=0: initialize with zeros (for U and p)
    data[0, :, :] = 0

    # For each selected time, read U and p files
    for idx, time in enumerate(selected_times, start=1):
        time_str = str(time)
        time_dir = os.path.join(case_dir, time_str)
        if not os.path.exists(time_dir):
            print(f"Warning: Timestep {time} missing in {case_dir}")
            continue

        # ----- Read velocity data -----
        u_file = os.path.join(time_dir, "U")
        with open(u_file, "r") as f:
            u_lines = f.readlines()
        try:
            start_index = u_lines.index("(\n") + 1
        except ValueError:
            raise ValueError(f"Could not locate velocity data in {u_file}")
        end_index = start_index + num_cells
        velocities = u_lines[start_index:end_index]
        for j, vel in enumerate(velocities):
            try:
                ux, uy, _ = map(float, vel.strip("()\n").split())
            except Exception as e:
                raise ValueError(f"Error parsing velocity on line {j}: {vel}") from e
            data[idx, j, 0] = ux
            data[idx, j, 1] = uy

        # ----- Read pressure data -----
        p_file = os.path.join(time_dir, "p")
        with open(p_file, "r") as f:
            p_lines = f.readlines()
        try:
            p_start = p_lines.index("(\n") + 1
        except ValueError:
            raise ValueError(f"Could not locate pressure data in {p_file}")
        p_end = p_start + num_cells
        pressures = p_lines[p_start:p_end]
        for j, p_val in enumerate(pressures):
            try:
                p_val_float = float(p_val.strip("()\n"))
            except Exception as e:
                raise ValueError(f"Error parsing pressure on line {j}: {p_val}") from e
            data[idx, j, 2] = p_val_float

    return data

# ================= Reshape Functions =================

def reshape_trajectory_data(sim_data, cell_centers, grid_shape, Re_value):
    """
    Reshape simulation data (timesteps, num_cells, 3) to a fixed grid of shape
    (timesteps, n_rows, n_cols, 6). The four channels are:
       - Channel 0: Ux
       - Channel 1: Uy
       - Channel 2: p
       - Channel 3: hole indicator (0 if cell exists; 1 if hole)
       - Channel 4: Reynolds number (constant)
      - Channel 5: Signed Distance Field (SDF)
    
    The mapping from cell center coordinates to grid indices is computed based on the
    bounding box of the cell centers. Each cell center (x,y) is mapped to:
    
        col = round((x - x_min) / (x_max - x_min) * (n_cols - 1))
        row = round((y - y_min) / (y_max - y_min) * (n_rows - 1))
    
    Cells that are not filled by any simulation cell (e.g. inside a hole) are left at
    zeros for Ux, Uy, p and the hole indicator is set to 1.
    """
    n_rows, n_cols = grid_shape
    T = sim_data.shape[0]
    
    # Compute domain boundaries from cell centers
    x_min, x_max = np.min(cell_centers[:, 0]), np.max(cell_centers[:, 0])
    y_min, y_max = np.min(cell_centers[:, 1]), np.max(cell_centers[:, 1])
    
    # Initialize final grid for all timesteps with zeros
    reshaped = np.zeros((T, n_rows, n_cols, 6))
    
    # Create a mask grid (hole indicator) initialized to 1 (i.e. hole)
    mask = np.ones((n_rows, n_cols))
    
    # Precompute mapping for each simulation cell (each entry in cell_centers)
    mapping = []
    for (x, y) in cell_centers:
        # Map x to column index and y to row index
        col = int(round((x - x_min) / (x_max - x_min) * (n_cols - 1)))
        row = int(round((y - y_min) / (y_max - y_min) * (n_rows - 1)))
        mapping.append((row, col))
        mask[row, col] = 0  # mark that a cell exists here

    #Compute SDF
    # outside_dist = distance_transform_edt(mask == 0)
    # inside_dist = distance_transform_edt(mask == 1)
    # sdf_field = outside_dist - inside_dist
    sdf_field = np.ones_like(mask, dtype=np.float32)  # Entire domain = fluid

    Re_min = 100
    Re_max = 10000
    Re_value_final = np.clip((Re_value - Re_min) / (Re_max - Re_min), 0.0, 1.0)

    # Assign Reynolds number to all cells in the grid
    reshaped[:, :, :, 3] = Re_value_final  # Ensure ALL cells, including holes, get Re value


    # For each timestep, fill in the data according to the mapping
    for t in range(T):
        for i, (row, col) in enumerate(mapping):
            # Place the simulation values (Ux, Uy, p)
            reshaped[t, row, col, :3] = sim_data[t, i, :3]  # Ux, Uy, p
            # For an existing cell, we set hole indicator to 0.
        # For positions that were not filled (hole positions), set channel 3 to 1.
        reshaped[t, :, :, 4] = mask
        reshaped[t, :, :, 5] = sdf_field

    return reshaped

def combine_and_reshape_trajectories(dataset_dir, cell_centers, grid_shape, timesteps, Re_values):
    """
    Combine all trajectory .npy files (each originally of shape (timesteps, num_cells, 3))
    and reshape them into a fixed grid of shape
         (num_trajectories, timesteps, n_rows, n_cols, final_channels)
    using the provided cell centers and grid shape.
    
    This function handles both regular and irregular geometries (with holes).
    """
    files = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".npy")])
    num_trajectories = len(files)
    combined_list = []
    
    for i, file in enumerate(files):
        print(f"Processing trajectory {i+1}/{num_trajectories}: {file}")
        sim_data = np.load(os.path.join(dataset_dir, file))
        Re_value = Re_values[i]
        reshaped_data = reshape_trajectory_data(sim_data, cell_centers, grid_shape, Re_value)
        combined_list.append(reshaped_data)
        
    combined = np.array(combined_list)
    return combined

# ================= Main Loop =================

def main():
    # We will assume that the mesh (and therefore cell centers) is fixed.
    # Generate cell centers using OpenFOAM's postProcess utility and parse them.
    generate_cell_centers(base_case_dir)
    centres_file = os.path.join(base_case_dir, "0", "C")
    cell_centers = parse_c_file(centres_file)
    print(f"Extracted {len(cell_centers)} cell centers from {centres_file}")

    # Main dataset generation loop
    for traj in range(num_trajectories):
        print(f"\nGenerating trajectory {traj+1}/{num_trajectories}...")
        U_value = U_values[traj]

        # Step 1: Update U file with current U_value
        update_U_file(base_case_dir, U_value)

        # Step 2: Run the simulation (standard blockMesh, icoFoam, etc.)
        run_simulation(base_case_dir)

        # Step 3: (Re)generate cell centers (if needed) and parse them.
        # In a fixed-domain, cell_centers should remain the same.
        generate_cell_centers(base_case_dir)
        cell_centers = parse_c_file(centres_file)
        print(f"Trajectory {traj+1}: {len(cell_centers)} cell centers extracted.")

        # Step 4: Extract simulation data (Ux, Uy, p) at the cell centers
        trajectory_data = extract_velocity_data(base_case_dir, timesteps, cell_centers)

        # Step 5: Save raw trajectory data
        traj_filename = os.path.join(output_dir, f"trajectory_{traj:04d}.npy")
        np.save(traj_filename, trajectory_data)
        print(f"Saved trajectory data to {traj_filename}")

    print("\nDataset generation complete. Now combining and reshaping trajectories...")

    # Combine and reshape all trajectories.
    # The final grid is defined by final_grid_shape.
    combined_data = combine_and_reshape_trajectories(dataset_dir, cell_centers, final_grid_shape, timesteps, Re_values)
    print(f"Combined dataset shape: {combined_data.shape}")

    # Save the combined dataset
    combined_output_file = os.path.join(final_output_dir, "Final_irr.npy")
    np.save(combined_output_file, combined_data)
    print(f"Saved combined dataset to {combined_output_file}")

    # Delete individual .npy files after combining
    for npy_file in glob.glob(os.path.join(dataset_dir, "*.npy")):
        os.remove(npy_file)
    print("Deleted individual trajectory .npy files.")

if __name__ == '__main__':
    main()
