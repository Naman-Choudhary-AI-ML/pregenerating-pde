import os
import shutil
import subprocess
import time
import numpy as np
import logging
from datetime import datetime, timedelta
import re
import random
from tqdm import tqdm
import gc
import json

# Configure logging
logging.basicConfig(
    filename="simulation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Function to log periodic success
last_log_time = datetime.now()
def log_periodic_success():
    global last_log_time
    if datetime.now() - last_log_time >= timedelta(minutes=30):
        logging.info("All operations successful so far.")
        last_log_time = datetime.now()

def copy_main_folder(main_folder, num_copies):
    """Copies the main folder content into num_copies of folders."""
    new_folders = []
    for i in range(1, num_copies + 1):
        new_folder = f"{main_folder}_copy_{i}"
        try:
            shutil.copytree(main_folder, new_folder)
            new_folders.append(new_folder)
            print(f"Created folder: {new_folder}")
            logging.info(f"Created folder: {new_folder}")
            time.sleep(1)
        except Exception as e:
            print(f"Error copying folder: {e}")
            logging.error(f"Error copying folder: {e}")
    return new_folders

def generate_U_file(folder):
    """Runs the U-file generation script (generate_setfields.py) inside the specified folder."""
    u_generator_script = os.path.join(folder, "generate_setfields.py")
    logging.info(f"Checking for generate_setfields.py in: {u_generator_script}")
    print(f"Checking for generate_setfields.py in: {u_generator_script}")
    if os.path.exists(u_generator_script):
        print(f"Running {u_generator_script} to generate U file in {folder}")
        logging.info(f"Running {u_generator_script} to generate U file in {folder}")
        try:
            process = subprocess.Popen(["python", "generate_setfields.py"], cwd=os.path.dirname(u_generator_script))
            process.communicate()
        except Exception as e:
            logging.error(f"Error running generate_setfields.py: {e}")
            print(f"Error running generate_setfields.py: {e}")
            return
        time.sleep(2)

        u_file = os.path.join(folder, "0", "U")
        if os.path.exists(u_file):
            print(f"U file generated successfully in folder: {folder}")
            logging.info(f"U file generated successfully in folder: {folder}")
        else:
            print(f"Error: U file was not created in {folder}. Check generate_setfields.py script for issues.")
            logging.error(f"Error: U file was not created in {folder}. Check generate_setfields.py script for issues.")
    else:
        print(f"Error: generate_setfields.py not found in {folder}. Expected path was: {u_generator_script}")
        logging.error(f"Error: generate_setfields.py not found in {folder}. Expected path was: {u_generator_script}")

def run_rhoPimpleFoam(folder):
    """Runs rhoPimpleFoam for the specified folder and returns True if it converged, False otherwise."""
    command = ["rhoPimpleFoam"]
    print(f"Running rhoPimpleFoam in folder: {folder}")
    try:
        process = subprocess.Popen(command, cwd=folder)
        process.communicate()
        # Check the return code
        if process.returncode == 0:
            print(f"rhoPimpleFoam simulation completed successfully in folder: {folder}")
            logging.info(f"rhoPimpleFoam simulation completed successfully in folder: {folder}")
            return True
        else:
            print(f"rhoPimpleFoam simulation failed or did not converge in folder: {folder}")
            logging.warning(f"rhoPimpleFoam simulation failed or did not converge in folder: {folder}")
            return False
    except FileNotFoundError:
        print("Error: rhoPimpleFoam command not found.")
        logging.error("Error: rhoPimpleFoam command not found.")
        return False
    except Exception as e:
        print(f"Error running rhoPimpleFoam: {e}")
        logging.error(f"Error running rhoPimpleFoam: {e}")
        return False

def run_setfields(folder):
    """Runs setFields in the specified folder."""
    command = ["setFields"]
    print(f"Running setFields in folder: {folder}")
    logging.info(f"Running setFields in folder: {folder}")
    try:
        process = subprocess.Popen(command, cwd=folder)
        process.communicate()
        if process.returncode == 0:
            print(f"setFields completed in folder: {folder}")
            logging.info(f"setFields completed in folder: {folder}")
        else:
            print(f"setFields failed in folder: {folder}")
            logging.warning(f"setFields failed in folder: {folder}")
    except FileNotFoundError:
        print("Error: setFields command not found.")
        logging.error("Error: setFields command not found.")
    except Exception as e:
        print(f"Error running setFields: {e}")
        logging.error(f"Error running setFields: {e}")

def generate_cell_centers(folder):
    """Generates cell center coordinates using OpenFOAM post-processing utilities."""
    command = ["postProcess", "-func", "writeCellCentres"]
    print(f"Running postProcess to generate cell centers in folder: {folder}")
    logging.info(f"Running postProcess to generate cell centers in folder: {folder}")
    
    try:
        # Run the postProcess command
        process = subprocess.Popen(command, cwd=folder)
        process.communicate()
        
        if process.returncode == 0:
            print(f"Cell center coordinates generated successfully in folder: {folder}")
            logging.info(f"Cell center coordinates generated successfully in folder: {folder}")
            
            # Path to the generated C file
            c_file_path = os.path.join(folder, "0", "C")
            
            # Check if the C file exists
            if os.path.isfile(c_file_path):
                return c_file_path  # Return the valid file path
            else:
                print(f"Error: C file not found at {c_file_path}")
                logging.error(f"Error: C file not found at {c_file_path}")
                return None  # Return None if the file doesn't exist
        else:
            print(f"Failed to generate cell center coordinates in folder: {folder}")
            logging.warning(f"Failed to generate cell center coordinates in folder: {folder}")
            return None  # Return None on failure
    except FileNotFoundError:
        print("Error: postProcess command not found.")
        logging.error("Error: postProcess command not found.")
        return None
    except Exception as e:
        print(f"Error running postProcess: {e}")
        logging.error(f"Error running postProcess: {e}")
        return None

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


def reshape_trajectory_data(sim_data, cell_centers, grid_shape):
    """
    Reshape simulation data (timesteps, num_cells, 4) to a fixed grid of shape
    (timesteps, n_rows, n_cols, 5). The five channels are:
       - Channel 0: ρ
       - Channel 1: Ux
       - Channel 2: Uy
       - Channel 3: P
       - Channel 4: hole indicator (0 if cell exists; 1 if hole)
    
    Each cell center (x,y) is mapped to a grid index based on the bounding box of cell_centers:
         col = round((x - x_min) / (x_max - x_min) * (n_cols - 1))
         row = round((y - y_min) / (y_max - y_min) * (n_rows - 1))
    
    Cells that are not filled by any simulation cell remain zero in the first four channels,
    and their hole indicator is left as 1.
    """
    n_rows, n_cols = grid_shape
    T = sim_data.shape[0]
    
    # Determine domain boundaries from the provided cell centers
    x_min, x_max = np.min(cell_centers[:, 0]), np.max(cell_centers[:, 0])
    y_min, y_max = np.min(cell_centers[:, 1]), np.max(cell_centers[:, 1])
    
    # Initialize grid with zeros; add one extra channel for the binary mask
    reshaped = np.zeros((T, n_rows, n_cols, 5))
    mask = np.ones((n_rows, n_cols))  # default: hole everywhere
    
    # Compute mapping for each cell center in this simulation.
    mapping = []
    for (x, y) in cell_centers:
        col = int(round((x - x_min) / (x_max - x_min) * (n_cols - 1)))
        row = int(round((y - y_min) / (y_max - y_min) * (n_rows - 1)))
        mapping.append((row, col))
        mask[row, col] = 0  # mark that a cell exists here

    # We now assume that sim_data.shape[1] is the number of simulation cells.
    # We'll fill the grid for each timestep using the mapping computed from the cell centers.
    # Note: If the number of simulation cells differs from the number of cell centers,
    #       you may decide to fill only up to the available count.
    n_cells_sim = sim_data.shape[1]
    n_cells_mapping = len(mapping)
    if n_cells_mapping != n_cells_sim:
        logging.warning(f"Number of cell centers ({n_cells_mapping}) does not match simulation cells ({n_cells_sim}). "
                        "Mapping will be computed using available cell centers.")
        n_cells = min(n_cells_mapping, n_cells_sim)
        mapping = mapping[:n_cells]
    else:
        n_cells = n_cells_sim

    for t in range(T):
        for i, (row, col) in enumerate(mapping):
            if i >= n_cells:
                break
            reshaped[t, row, col, 0:4] = sim_data[t, i, :]
            reshaped[t, row, col, 4] = 0  # cell exists
        reshaped[t, :, :, 4] = mask  # fill any remaining holes
    return reshaped

def combine_and_reshape_trajectories(dataset, cell_centers, grid_shape):
    """
    Combine all trajectory data (from a single results.npy file with shape
         (num_trajectories, timesteps, num_cells, 4))
    and reshape them into a fixed grid of shape
         (num_trajectories, timesteps, n_rows, n_cols, 5)
    using the provided cell centers and grid shape.
    """
    # num_trajectories = dataset.shape[0]
    combined_list = []
    
    for sim_data in tqdm(dataset, desc="Reshaping Trajectories"):
        # Extract the simulation data for trajectory i (all timesteps, all cells, all 4 channels)
        # sim_data = dataset[i, :, :, :]  # shape: (timesteps, 16320, 4)
        reshaped_data = reshape_trajectory_data(sim_data, cell_centers, grid_shape)  # shape: (timesteps, 128, 128, 5)
        combined_list.append(reshaped_data)
        gc.collect()  # free memory if needed
        
    combined = np.array(combined_list)
    return combined

def parse_internal_field(file_path, field_type="vector", default_value=None):
    """
    Parses the internal field data from an OpenFOAM field file.
    
    For a "vector" field, it returns the first two components (Ux, Uy).
    If the file specifies a uniform field (i.e. "uniform" appears without "nonuniform"),
    the function returns None so that the calling code can skip this timestep.
    
    Args:
        file_path (str): Path to the OpenFOAM field file.
        field_type (str): "vector" or "scalar".
        default_value (float, optional): Default value if no data is found (for nonuniform fields).
    
    Returns:
        np.ndarray or None:
          - For "vector": shape = (n_points, 2)
          - For "scalar": shape = (n_points,)
          - Returns None if a uniform field is encountered.
    
    Raises:
        FileNotFoundError: If file_path is not found.
        ValueError: If file format is unexpected.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    data_lines = []
    inside_data_block = False
    n_points = None  # To be set once a line with a digit is found.
    found_internalField = False  # Flag indicating we've seen "internalField"

    with open(file_path, "r") as f:
        for line in f:
            stripped = line.strip()

            # If we accidentally enter the next section, break out.
            if inside_data_block and stripped.startswith("boundaryField"):
                break

            # Look for the internalField definition.
            if "internalField" in stripped:
                found_internalField = True
                # First, check for nonuniform (note: "nonuniform" includes "uniform", so check it first)
                if "nonuniform" in stripped:
                    inside_data_block = True
                    continue
                # Otherwise, if a uniform field is detected, return None.
                elif "uniform" in stripped:
                    return None

            # If we've seen "internalField" but haven't started the data block yet,
            # check if this line indicates a nonuniform block.
            if found_internalField and not inside_data_block:
                if "nonuniform" in stripped:
                    inside_data_block = True
                    continue

            # Process lines inside the data block.
            if inside_data_block:
                # Check if this line indicates the number of points.
                s = stripped.rstrip(";")
                if n_points is None and s.isdigit():
                    n_points = int(s)
                    continue

                # Skip the opening parenthesis.
                if stripped == "(":
                    continue

                # Check for a closing parenthesis.
                if stripped in [")", ");"] or stripped.startswith(")"):
                    inside_data_block = False
                    if n_points is not None and len(data_lines) >= n_points:
                        break
                    continue

                # Append the data line.
                data_lines.append(stripped)
                if n_points is not None and len(data_lines) >= n_points:
                    break

    # If no data was collected and no default is provided, raise an error.
    if (n_points is None or len(data_lines) == 0) and default_value is None:
        raise ValueError(f"No internalField data found in {file_path}")

    # If a default is provided, use it.
    if (n_points is None or len(data_lines) == 0) and default_value is not None:
        if n_points is None:
            raise ValueError("Cannot determine number of points to use default_value.")
        if field_type == "vector":
            return np.array([[default_value, default_value]] * n_points, dtype=np.float32)
        elif field_type == "scalar":
            return np.array([default_value] * n_points, dtype=np.float32)

    # If n_points was not explicitly set, assume it equals the number of collected lines.
    if n_points is None:
        n_points = len(data_lines)

    if len(data_lines) != n_points:
        raise ValueError(
            f"Expected {n_points} data lines, but got {len(data_lines)} in {file_path}"
        )

    # Parse the collected data.
    if field_type == "vector":
        vector_data = []
        for dl in data_lines:
            dl_clean = dl.strip("()")
            comps = dl_clean.split()
            try:
                comps_float = [float(c) for c in comps[:2]]
            except ValueError:
                raise ValueError(f"Could not parse vector line: '{dl}'")
            vector_data.append(comps_float)
        array_data = np.array(vector_data, dtype=np.float32)
        if array_data.shape[0] != n_points:
            raise ValueError(
                f"Expected {n_points} data lines, got {array_data.shape[0]} in {file_path}"
            )
        return array_data

    elif field_type == "scalar":
        scalar_data = []
        for dl in data_lines:
            try:
                val = float(dl)
            except ValueError:
                raise ValueError(f"Could not parse scalar line: '{dl}'")
            scalar_data.append(val)
        array_data = np.array(scalar_data, dtype=np.float32)
        if array_data.shape[0] != n_points:
            raise ValueError(
                f"Expected {n_points} data lines, got {array_data.shape[0]} in {file_path}"
            )
        return array_data

    else:
        raise ValueError("field_type must be either 'vector' or 'scalar'.")


def parse_simulation(sim_folder):
    """
    Parses a single simulation folder containing time-step directories 
    (e.g. 0, 0.1, 0.2, ...).

    For each time step it expects to find:
      - T  (scalar)
      - U  (vector)
      - p  (scalar)

    If any of these files return None (e.g. due to a uniform internalField),
    that time step is skipped.

    Returns:
        valid_time_dirs (list of str): Sorted list of time-step strings with valid data.
        results_array (np.ndarray): Array of shape (num_time_steps, n_points, 4) for channels [rho, Ux, Uy, p].
                                   Returns None if no valid time steps are found.
    """
    # List all numeric entries in the simulation folder.
    all_entries = os.listdir(sim_folder)
    time_dirs = []
    for entry in all_entries:
        try:
            float(entry)  # Only consider numeric directory names.
            entry_path = os.path.join(sim_folder, entry)
            if os.path.isdir(entry_path):
                time_dirs.append(entry)
        except ValueError:
            continue

    # Sort time directories numerically.
    time_dirs.sort(key=lambda x: float(x))

    valid_time_dirs = []
    results_per_time = []

    for tdir in time_dirs:
        tdir_path = os.path.join(sim_folder, tdir)
        # Prepare file paths for each field.
        T_file = os.path.join(tdir_path, "T")
        U_file = os.path.join(tdir_path, "U")
        p_file = os.path.join(tdir_path, "p")

        # Parse each field.
        T_data = parse_internal_field(T_file, field_type="scalar")
        U_data = parse_internal_field(U_file, field_type="vector")
        p_data = parse_internal_field(p_file, field_type="scalar")

        # Skip this timestep if any field returns None.
        if T_data is None or U_data is None or p_data is None:
            logging.info(f"Skipping timestep {tdir} in {sim_folder} due to uniform field.")
            continue

        try:
            # Compute density: rho = p / (T * 287). (Adjust constant if needed.)
            rho_data = p_data / (T_data * 287)
        except ZeroDivisionError:
            logging.error(f"Division by zero at timestep {tdir} in {sim_folder}. Skipping timestep.")
            continue

        # Combine data for each point: [rho, Ux, Uy, p]
        combined_data = np.column_stack([rho_data, U_data[:, 0], U_data[:, 1], p_data])
        valid_time_dirs.append(tdir)
        results_per_time.append(combined_data)

    if len(valid_time_dirs) == 0 or len(results_per_time) == 0:
        logging.warning(f"No valid timesteps found in simulation folder {sim_folder}.")
        return None

    results_array = np.stack(results_per_time, axis=0)  # Shape: (num_time_steps, n_points, 4)
    return valid_time_dirs, results_array


def gather_all_simulations(sim_folders, grid_shape=(128, 128), c_file_name="0/C"):
    """
    Given a list of simulation folders, parse them all (via parse_simulation),
    then reshape each simulation's data onto a fixed grid of shape (timesteps, n_rows, n_cols, 5).
    
    Each simulation uses its own cell centers (parsed from its own "C" file) to compute the mapping.
    This avoids mismatches when simulations have different numbers of cells.
    
    Returns:
        common_time_dirs (list of str): The time step labels from the first successful simulation.
        combined_dataset (np.ndarray): Array of shape (num_sims, timesteps, n_rows, n_cols, 5).
    """
    all_reshaped = []  # List to store reshaped simulation arrays (each: (timesteps, 128, 128, 5))
    common_time_dirs = None

    for folder in sim_folders:
        # Parse simulation data (raw data: (timesteps, num_cells, 4))
        sim_result = parse_simulation(folder)
        if sim_result is None:
            logging.error(f"No valid timesteps found in simulation folder {folder}. Skipping folder.")
            continue
        time_dirs, sim_array = sim_result
        logging.info(f"Folder {folder} => sim_array shape: {sim_array.shape} (timesteps, num_cells, 4)")
        logging.info(f"Time directories: {time_dirs}")

        # For the first successful simulation, record the time directories as reference.
        if common_time_dirs is None:
            common_time_dirs = time_dirs
        else:
            if len(time_dirs) != len(common_time_dirs):
                logging.warning(f"Folder {folder} has a different number of timesteps. Stacking by index.")

        # Parse cell centers for THIS simulation individually.
        c_file_path = os.path.join(folder, c_file_name)
        if not os.path.isfile(c_file_path):
            logging.error(f"Cell centers file not found for folder {folder}. Skipping folder.")
            continue
        cell_centers = parse_c_file(c_file_path)
        logging.info(f"Folder {folder} => {len(cell_centers)} cell centers parsed.")

        # Reshape the simulation's raw data onto the fixed grid using its own cell centers.
        reshaped = reshape_trajectory_data(sim_array, cell_centers, grid_shape)
        logging.info(f"Folder {folder} reshaped to: {reshaped.shape}")
        all_reshaped.append(reshaped)

    if len(all_reshaped) == 0:
        logging.warning("No simulations were successfully processed.")
        return None, None

    # Stack the reshaped simulations along a new axis.
    combined_dataset = np.stack(all_reshaped, axis=0)  # (num_sims, timesteps, 128, 128, 5)
    return common_time_dirs, combined_dataset

def update_Umax(file_path, new_value):
    """
    Reads the file at file_path, replaces the Umax value with new_value, and writes it back.
    The line to be replaced is assumed to be of the form:
        const scalar Umax = 10.0;
    """
    # Pattern to match the Umax definition
    pattern = r"(const\s+scalar\s+Umax\s*=\s*)([\d\.Ee+-]+)(\s*;)"

    with open(file_path, "r") as f:
        content = f.read()

    # Replace the old value with the new one
    def replacement(match):
        return f"{match.group(1)}{new_value}{match.group(3)}"

    updated_content, count = re.subn(pattern, replacement, content)

    if count == 0:
        raise ValueError(f"Could not find the Umax definition in {file_path}")

    with open(file_path, "w") as f:
        f.write(updated_content)

    logging.info(f"Updated {file_path}: Umax set to {new_value}")

def update_Umax_in_simulation_folder(sim_folder, min_value=1.0, max_value=10.0):
    """
    Updates the Umax value in the U file (located in the '0' subfolder) of the given simulation folder.
    A random Umax value is chosen between min_value and max_value.
    """
    u_file_path = os.path.join(sim_folder, "0", "U")
    if not os.path.exists(u_file_path):
        raise FileNotFoundError(f"U file not found at {u_file_path}")
    
    # Choose a new random Umax value in the specified range
    new_value = random.uniform(min_value, max_value)
    update_Umax(u_file_path, new_value)

def delete_blockMeshDict(folder_path):
    """
    Deletes the existing blockMeshDict file in the given folder.
    
    Parameters:
    folder_path (str): Path to the folder where blockMeshDict exists.
    """
    blockMeshDict_path = os.path.join(folder_path, "system", "blockMeshDict")

    try:
        if os.path.exists(blockMeshDict_path):
            os.remove(blockMeshDict_path)
            print(f"Deleted existing blockMeshDict in {folder_path}")
        else:
            print(f"No blockMeshDict found in {folder_path}, skipping deletion.")
    except Exception as e:
        print(f"Error deleting blockMeshDict in {folder_path}: {e}")
scaled_hole1 = (0.625 * 32, 0.9375 * 32, 0.125 * 32, 0.125 * 32)
scaled_hole2 = (1.25 * 32, 0.9375 * 32, 0.125 * 32, 0.125 * 32)
new_holes = [scaled_hole1, scaled_hole2]
def generate_blockMeshDict(domain=(0, 64, 0, 64),
                           resolution=(128, 128),
                           # holes given as list of tuples: (hole_x, hole_y, hole_width, hole_height)
                           holes=new_holes,
                           z_thickness=0.1,
                           run_blockMesh=False,
                           out_file="blockMeshDict"):
    """
    Generate a blockMeshDict for a 2x2 domain with a given mesh resolution (128x128)
    and holes of fixed physical size (0.125 x 0.125). Holes are specified by their
    lower-left corner coordinates.
    """
    import math
    import os
    import subprocess

    xmin, xmax, ymin, ymax = domain
    nx, ny = resolution
    dx = (xmax - xmin) / nx   # cell size in x
    dy = (ymax - ymin) / ny   # cell size in y

    # Create sets of x and y boundaries: include domain boundaries...
    x_boundaries = {xmin, xmax}
    y_boundaries = {ymin, ymax}

    # ...and add the boundaries of each hole
    for (hx, hy, hw, hh) in holes:
        x_boundaries.add(hx)
        x_boundaries.add(hx + hw)
        y_boundaries.add(hy)
        y_boundaries.add(hy + hh)

    x_coords = sorted(x_boundaries)
    y_coords = sorted(y_boundaries)
    # For our example with two holes, we expect something like:
    #   x_coords = [0, 0.625, 0.75, 1.25, 1.375, 2]
    #   y_coords = [0, 0.9375, 1.0625, 2]

    Nx_points = len(x_coords)
    Ny_points = len(y_coords)

    # Generate vertices for two z-planes (z=0 and z=z_thickness)
    vertices = []
    for k in [0, z_thickness]:
        for j in range(Ny_points):
            for i in range(Nx_points):
                vertices.append((x_coords[i], y_coords[j], k))

    # Build candidate blocks (each cell in the x-y grid) and mark those that fall inside a hole.
    blocks = []
    block_indices = {}  # map grid cell (i,j) -> block id (for later lookup)
    missing = {}        # mark grid cells that lie within a hole
    for j in range(Ny_points - 1):
        for i in range(Nx_points - 1):
            # center of candidate block
            cx = 0.5 * (x_coords[i] + x_coords[i + 1])
            cy = 0.5 * (y_coords[j] + y_coords[j + 1])
            in_hole = False
            for (hx, hy, hw, hh) in holes:
                if hx <= cx <= hx + hw and hy <= cy <= hy + hh:
                    in_hole = True
                    break
            if not in_hole:
                # Compute number of cells in this block (should come out as an integer)
                ncx = int(round((x_coords[i + 1] - x_coords[i]) / dx))
                ncy = int(round((y_coords[j + 1] - y_coords[j]) / dy))
                blocks.append({'i': i, 'j': j, 'ncx': ncx, 'ncy': ncy})
                block_indices[(i, j)] = len(blocks) - 1
            else:
                missing[(i, j)] = True

    # Helper: compute vertex index from grid indices (i,j,k)
    def vertex_index(i, j, k):
        return i + j * Nx_points + k * (Nx_points * Ny_points)

    # Create block definitions (list of 8 vertex indices plus cell counts)
    block_defs = []
    for block in blocks:
        i = block['i']
        j = block['j']
        v0 = vertex_index(i, j, 0)
        v1 = vertex_index(i + 1, j, 0)
        v2 = vertex_index(i + 1, j + 1, 0)
        v3 = vertex_index(i, j + 1, 0)
        v4 = vertex_index(i, j, 1)
        v5 = vertex_index(i + 1, j, 1)
        v6 = vertex_index(i + 1, j + 1, 1)
        v7 = vertex_index(i, j + 1, 1)
        block_defs.append((v0, v1, v2, v3, v4, v5, v6, v7, block['ncx'], block['ncy']))

    # --- Boundary patches ---
    # Outer boundaries (those faces that lie on the domain edges)
    boundaries = {"left": [], "right": [], "bottom": [], "top": [], "frontAndBack": []}
    for block in blocks:
        i = block['i']
        j = block['j']
        # Left face:
        if i == 0:
            face = (vertex_index(i, j, 0),
                    vertex_index(i, j + 1, 0),
                    vertex_index(i, j + 1, 1),
                    vertex_index(i, j, 1))
            boundaries["left"].append(face)
        # Right face:
        if i == Nx_points - 2:
            face = (vertex_index(i + 1, j, 0),
                    vertex_index(i + 1, j + 1, 0),
                    vertex_index(i + 1, j + 1, 1),
                    vertex_index(i + 1, j, 1))
            boundaries["right"].append(face)
        # Bottom face:
        if j == 0:
            face = (vertex_index(i, j, 0),
                    vertex_index(i + 1, j, 0),
                    vertex_index(i + 1, j, 1),
                    vertex_index(i, j, 1))
            boundaries["bottom"].append(face)
        # Top face:
        if j == Ny_points - 2:
            face = (vertex_index(i, j + 1, 0),
                    vertex_index(i + 1, j + 1, 0),
                    vertex_index(i + 1, j + 1, 1),
                    vertex_index(i, j + 1, 1))
            boundaries["top"].append(face)
    # Front and back faces: every block gets these (for 2D, they are type empty)
    frontAndBack = []
    for block in block_defs:
        v0, v1, v2, v3, v4, v5, v6, v7, _, _ = block
        frontAndBack.append((v0, v1, v2, v3))  # front (z=0)
        frontAndBack.append((v4, v5, v6, v7))  # back  (z=z_thickness)
    boundaries["frontAndBack"] = frontAndBack

    # --- Internal (hole) boundaries ---
    # Instead of hard-coding two hole patches, we first group the missing cells (grid cells
    # inside any hole) into connected components using a flood-fill algorithm.
    missing_cells = set(missing.keys())
    visited = set()
    hole_groups = []
    for cell in missing_cells:
        if cell in visited:
            continue
        group = []
        stack = [cell]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            group.append(cur)
            i, j = cur
            # Check 4-connected neighbors (left, right, bottom, top)
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (i + di, j + dj)
                if neighbor in missing_cells and neighbor not in visited:
                    stack.append(neighbor)
        hole_groups.append(group)

    # For each hole group, determine the boundary faces where a missing cell touches a valid block.
    hole_boundaries = {}
    for idx, group in enumerate(hole_groups, start=1):
        faces = set()
        for (i, j) in group:
            # Left neighbor: if exists in block_indices then add left face.
            if (i - 1, j) in block_indices:
                face = (vertex_index(i, j, 0),
                        vertex_index(i, j + 1, 0),
                        vertex_index(i, j + 1, 1),
                        vertex_index(i, j, 1))
                faces.add(face)
            # Right neighbor:
            if (i + 1, j) in block_indices:
                face = (vertex_index(i + 1, j, 0),
                        vertex_index(i + 1, j + 1, 0),
                        vertex_index(i + 1, j + 1, 1),
                        vertex_index(i + 1, j, 1))
                faces.add(face)
            # Bottom neighbor:
            if (i, j - 1) in block_indices:
                face = (vertex_index(i, j, 0),
                        vertex_index(i + 1, j, 0),
                        vertex_index(i + 1, j, 1),
                        vertex_index(i, j, 1))
                faces.add(face)
            # Top neighbor:
            if (i, j + 1) in block_indices:
                face = (vertex_index(i, j + 1, 0),
                        vertex_index(i + 1, j + 1, 0),
                        vertex_index(i + 1, j + 1, 1),
                        vertex_index(i, j + 1, 1))
                faces.add(face)
        hole_boundaries[f"hole{idx}"] = list(faces)

    # Add the hole boundaries to the overall boundaries dictionary.
    boundaries.update(hole_boundaries)

    # --- Write out the blockMeshDict file ---
    out_lines = []
    out_lines.append("FoamFile")
    out_lines.append("{")
    out_lines.append("    version     2.0;")
    out_lines.append("    format      ascii;")
    out_lines.append("    class       dictionary;")
    out_lines.append("    object      blockMeshDict;")
    out_lines.append("}")
    out_lines.append("convertToMeters 1;\n")

    # Vertices
    out_lines.append("vertices")
    out_lines.append("(")
    for v in vertices:
        out_lines.append("    ({} {} {})".format(v[0], v[1], v[2]))
    out_lines.append(");\n")

    # Blocks
    out_lines.append("blocks")
    out_lines.append("(")
    for block in block_defs:
        v0, v1, v2, v3, v4, v5, v6, v7, ncx, ncy = block
        out_lines.append("    hex ({} {} {} {} {} {} {} {}) ({} {} 1) simpleGrading (1 1 1)".format(
            v0, v1, v2, v3, v4, v5, v6, v7, ncx, ncy))
    out_lines.append(");\n")

    # Boundaries
    out_lines.append("boundary")
    out_lines.append("(")
    for patch, faces in boundaries.items():
        # Determine type: for frontAndBack, use empty; for outer boundaries, you might use patch or wall;
        # for holes, we use wall.
        if patch == "frontAndBack":
            ptype = "empty"
        elif patch in ["bottom", "top"]:
            ptype = "wall"
        elif "hole" in patch:
            ptype = "wall"
        else:
            ptype = "patch"
        out_lines.append("    {}".format(patch))
        out_lines.append("    {")
        out_lines.append("        type    {};".format(ptype))
        out_lines.append("        faces")
        out_lines.append("        (")
        for face in faces:
            face_str = "(" + " ".join(str(v) for v in face) + ")"
            out_lines.append("            " + face_str)
        out_lines.append("        );")
        out_lines.append("    }")
    out_lines.append(");\n")

    out_lines.append("mergePatchPairs\n(")
    out_lines.append(");\n")

    with open(out_file, "w") as f:
        f.write("\n".join(out_lines))
    print("blockMeshDict written to", out_file)

    if run_blockMesh:
        # system_dir = .../design_point_X/system
        system_dir = os.path.dirname(os.path.abspath(out_file))
        # base_dir = .../design_point_X (one level above system)
        base_dir = os.path.dirname(system_dir)
        
        old_cwd = os.getcwd()
        try:
            # Move up one directory level
            os.chdir(base_dir)
            subprocess.run(["blockMesh"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"blockMesh failed: {e}")
        finally:
            os.chdir(old_cwd)
    return [patch for patch in boundaries.keys() if patch.startswith("hole")]
# Example usage:
import random

def randomize_holes(n, hole_size=(4.0, 4.0), domain=(0, 64, 0, 64), grid_step=0.50):
    """
    Generate n random holes within the given domain.
    
    Each hole is defined as a tuple (hole_x, hole_y, hole_width, hole_height),
    where (hole_x, hole_y) is the lower-left corner.
    
    The lower-left coordinates are multiples of grid_step to ensure consistency
    with the cell resolution (e.g., 128 cells over a 2×2 domain).
    
    Parameters:
        n (int): Number of holes to generate.
        hole_size (tuple): Fixed size for each hole (width, height).
        domain (tuple): (xmin, xmax, ymin, ymax) defining the overall domain.
        grid_step (float): The resolution increment (default 0.015625).
    
    Returns:
        list of tuples: Each tuple is (hole_x, hole_y, hole_width, hole_height).
    """
    xmin, xmax, ymin, ymax = domain
    hole_width, hole_height = hole_size

    # To keep the hole fully inside the domain, the lower-left corner can vary from:
    x_min_possible = xmin
    x_max_possible = xmax - hole_width
    y_min_possible = ymin
    y_max_possible = ymax - hole_height

    # Create lists of possible x and y coordinates (multiples of grid_step)
    num_x = int((x_max_possible - xmin) / grid_step) + 1
    num_y = int((y_max_possible - ymin) / grid_step) + 1
    x_positions = [round(xmin + i * grid_step, 12) for i in range(num_x)]
    y_positions = [round(ymin + j * grid_step, 12) for j in range(num_y)]
    
    holes = []
    for _ in range(n):
        hole_x = random.choice(x_positions)
        hole_y = random.choice(y_positions)
        holes.append((hole_x, hole_y, hole_width, hole_height))
    
    return holes

def update_field_file(field_file_path, hole_patch_names, default_patch_content):
    """
    Update a field file (e.g., 0/U) by inserting new patch definitions for each hole.
    default_patch_content is a list of strings (each line) for the patch's settings.
    """
    with open(field_file_path, 'r') as f:
        lines = f.readlines()

    # Find the boundaryField block
    start_index = None
    for i, line in enumerate(lines):
        if "boundaryField" in line:
            start_index = i
            break
    if start_index is None:
        print("boundaryField not found in", field_file_path)
        return

    # Find the opening brace of boundaryField
    for i in range(start_index, len(lines)):
        if "{" in lines[i]:
            start_brace_index = i
            break

    # Find the matching closing brace of the boundaryField block
    brace_count = 0
    end_index = None
    for i in range(start_brace_index, len(lines)):
        brace_count += lines[i].count("{")
        brace_count -= lines[i].count("}")
        if brace_count == 0:
            end_index = i
            break
    if end_index is None:
        print("Could not find the end of boundaryField in", field_file_path)
        return

    # Insert hole patch definitions before the closing brace.
    new_lines = lines[:end_index]
    for patch in hole_patch_names:
        new_lines.append("    " + patch + "\n")
        new_lines.append("    {\n")
        for line in default_patch_content:
            new_lines.append("        " + line + "\n")
        new_lines.append("    }\n")
    new_lines.extend(lines[end_index:])

    with open(field_file_path, 'w') as f:
        f.writelines(new_lines)
    print("Updated", field_file_path, "with patches", hole_patch_names)

def update_all_field_files(folder, hole_patch_names):
    """
    Update the field files in the 0/ folder to include hole patches.
    """
    field_dir = os.path.join(folder, "0")
    # Define default patch settings for each field.
    field_defaults = {
        "U": ["type fixedValue;", "value uniform (0 0 0);"],
        "p": ["type zeroGradient;"],
        "T": ["type zeroGradient;"],
        "rho": ["type zeroGradient;"]
    }
    for field, default_patch in field_defaults.items():
        field_file_path = os.path.join(field_dir, field)
        update_field_file(field_file_path, hole_patch_names, default_patch)

def main():
    total_trajectories = int(input("Enter the total number of trajectories to simulate: "))
    main_folder = "Design_Point_0"
    start_time = time.time()
    batch_size = 128  # Number of simulations per batch
    batch_id = 0
    processed_batches = []

    trajectories_done = 0  # Track how many have been processed

    # --- Process in Batches ---
    while trajectories_done < total_trajectories:
        current_batch_size = min(batch_size, total_trajectories - trajectories_done)

        # Step 1: Create a batch of folders
        batch_folders = copy_main_folder(main_folder, current_batch_size)
        sim_data = []
        converged_folders = []

        # Step 2: Run simulations for this batch
        for i, folder in enumerate(batch_folders):
            folder_start_time = time.time()

            # Generate randomized holes for this folder
            num_holes = random.randint(2, 15)
            logging.info(f"Processing folder {folder} with {num_holes} holes.")
            random_holes = randomize_holes(num_holes)

            # Generate blockMeshDict and update simulation fields
            delete_blockMeshDict(folder)
            hole_patch_names = generate_blockMeshDict(
                holes=random_holes,
                out_file=os.path.join(folder, "system", "blockMeshDict"),
                run_blockMesh=True
            )
            update_all_field_files(folder, hole_patch_names)
            update_Umax_in_simulation_folder(folder, min_value=1.125e-3, max_value=1.125e-1)

            # Generate and validate cell centers
            c_file = generate_cell_centers(folder)
            if not c_file:
                logging.error(f"Cell centers file not found for {folder}. Skipping.")
                continue

            # Run OpenFOAM simulation
            if not run_rhoPimpleFoam(folder):
                logging.warning(f"Solver failed for folder {folder}. Skipping.")
                continue

            elapsed_time = time.time() - folder_start_time
            logging.info(f"Simulation for {folder} completed in {elapsed_time:.2f} seconds")
            converged_folders.append(folder)

            # Store simulation metadata
            with open(c_file, "r") as f:
                c_contents = f.read()

            sim_data.append({"folder": folder, "c_contents": c_contents})

        # Step 3: Save simulation metadata
        if sim_data:
            sim_data_path = f"sim_data_batch_{batch_id}.json"
            with open(sim_data_path, "w") as jf:
                json.dump(sim_data, jf)
            logging.info(f"Sim data saved to {sim_data_path}")
        else:
            logging.warning("No valid simulations in this batch.")

        # Step 4: Run `gather_all_simulations` for this batch
        if converged_folders:
            try:
                final_time_dirs, final_data = gather_all_simulations(converged_folders)
                if final_data is not None:
                    batch_file = f"all_sims_data_batch_{batch_id}.npy"
                    np.save(batch_file, final_data)
                    logging.info(f"Batch {batch_id} dataset shape: {final_data.shape} saved as {batch_file}")

                    # Append time directories
                    with open("time_dirs.txt", "a") as f:
                        for t in final_time_dirs:
                            f.write(str(t) + "\n")

                    processed_batches.append(batch_file)
                    batch_id += 1
                else:
                    logging.error("No simulations successfully processed in this batch. Skipping.")
            except FileNotFoundError as e:
                logging.error(f"Error processing batch: {e}")

        # Step 5: Delete folders in the current batch to free space
        for folder in batch_folders:
            try:
                shutil.rmtree(folder)
                logging.info(f"Deleted folder {folder}")
            except Exception as e:
                logging.error(f"Error deleting folder {folder}: {e}")

        # Move to the next batch
        trajectories_done += current_batch_size

    # Final log: Summarize results
    total_time_elapsed = time.time() - start_time
    logging.info(f"Total execution time: {total_time_elapsed:.2f} seconds")
    print(f"\nTotal execution time: {total_time_elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
