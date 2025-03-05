import os
import shutil
import subprocess
import time
import numpy as np
import logging
from datetime import datetime, timedelta
import re
import json
import torch
import random
import gc
from tqdm import tqdm
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

def write_scalar_field(filename, fieldName, dataList, dimensions):
    """
    Write a scalar field in nonuniform List<scalar>.
    dataList must be 1D, length = #validCells, in the correct order.
    """
    with open(filename, 'w') as f:
        f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
        f.write("  =========                 |\n")
        f.write("  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox\n")
        f.write("   \\    /   O peration     | Website:  https://openfoam.org\n")
        f.write("    \\  /    A nd           | Version:  8\n")
        f.write("     \\/     M anipulation  |\n")
        f.write("\\*---------------------------------------------------------------------------*/\n")
        f.write("FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       volScalarField;\n")
        f.write(f"    object      {fieldName};\n}}\n\n")
        f.write(f"dimensions      {dimensions};\n")
        f.write("internalField   nonuniform List<scalar>\n")
        f.write(f"{len(dataList)}\n(\n")
        for val in dataList:
            f.write(f"    {val:.6f}\n")
        f.write(");\n\nboundaryField\n{\n")
        f.write("    left\n    {\n        type            cyclic;\n    }\n")
        f.write("    right\n    {\n        type            cyclic;\n    }\n")
        f.write("    bottom\n    {\n        type            cyclic;\n    }\n")
        f.write("    top\n    {\n        type            cyclic;\n    }\n")
        f.write("    hole\n    {\n        type            zeroGradient;\n    }\n")
        f.write("    frontAndBack\n    {\n        type            empty;\n    }\n}\n")

    logging.info(f"Wrote scalar field '{fieldName}' with {len(dataList)} entries to {filename}")

def write_vector_field(filename, fieldName, UxList, UyList, dimensions):
    """
    Write a vector field in nonuniform List<vector>.
    UxList, UyList must be 1D, same length, in the correct order.
    """
    with open(filename, 'w') as f:
        f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
        f.write("  =========                 |\n")
        f.write("  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox\n")
        f.write("   \\    /   O peration     | Website:  https://openfoam.org\n")
        f.write("    \\  /    A nd           | Version:  8\n")
        f.write("     \\/     M anipulation  |\n")
        f.write("\\*---------------------------------------------------------------------------*/\n")
        f.write("FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       volVectorField;\n")
        f.write(f"    object      {fieldName};\n}}\n\n")
        f.write(f"dimensions      {dimensions};\n")
        f.write("internalField   nonuniform List<vector>\n")
        f.write(f"{len(UxList)}\n(\n")
        for (ux, uy) in zip(UxList, UyList):
            f.write(f"    ({ux:.6f} {uy:.6f} 0.000000)\n")
        f.write(");\n\nboundaryField\n{\n")
        f.write("    left\n    {\n        type            cyclic;\n    }\n")
        f.write("    right\n    {\n        type            cyclic;\n    }\n")
        f.write("    bottom\n    {\n        type            cyclic;\n    }\n")
        f.write("    top\n    {\n        type            cyclic;\n    }\n")
        f.write("    hole\n    {\n        type            fixedValue;\n        value uniform (0 0 0);\n    }\n")
        f.write("    frontAndBack\n    {\n        type            empty;\n    }\n}\n")

    logging.info(f"Wrote vector field '{fieldName}' with {len(UxList)} entries to {filename}")

def generate_random_coefficients(p):
    """
    Generate random coefficients for perturbations.
    
    Args:
        p (int): Number of modes for perturbations.
        
    Returns:
        dict: A dictionary containing random coefficients for alpha0, beta0, alpha1, and beta1.
    """
    return {
        "alpha0": np.random.uniform(0, 1, p),
        "beta0": np.random.uniform(0, 1, p),
        "alpha1": np.random.uniform(0, 1, p),
        "beta1": np.random.uniform(0, 1, p),
    }

def get_block_slices(i_c, j_c, N=128, hole_size=None):
    """
    Given:
      - i_c, j_c: integer indices of the hole center in [0..N-1].
      - N: total resolution (128).
      - hole_size: number of cells along each side of the hole (8).

    Returns a list of (iRange, jRange) for the 8 blocks around the hole.
    The center block (the hole) is omitted.
    """

    # 1) Derive the hole boundaries
    half = hole_size // 2  # 8//2 = 4
    i_hole_min = i_c - half
    i_hole_max = i_hole_min + hole_size - 1  # e.g. i_hole_min+7
    j_hole_min = j_c - half
    j_hole_max = j_hole_min + hole_size - 1

    # 2) The number of cells in each sub-region (matching blockMesh):
    Nx_left   = i_hole_min            # from 0..(i_hole_min - 1)
    Nx_hole   = hole_size             # the 8 cells in the hole
    Nx_right  = N - (i_hole_max + 1)  # from (i_hole_max+1)..(N-1)
    
    Ny_bottom = j_hole_min
    Ny_hole   = hole_size
    Ny_top    = N - (j_hole_max + 1)

    # 3) Define the slices in i-direction
    i_slices = [
        range(0, Nx_left),                           # left
        range(Nx_left, Nx_left + Nx_hole),           # center
        range(Nx_left + Nx_hole, Nx_left + Nx_hole + Nx_right)  # right
    ]
    # 4) Define the slices in j-direction
    j_slices = [
        range(0, Ny_bottom), 
        range(Ny_bottom, Ny_bottom + Ny_hole), 
        range(Ny_bottom + Ny_hole, Ny_bottom + Ny_hole + Ny_top)
    ]

    # 5) Combine them in the same order as your 8 hex blocks:
    #        1) bottom-left,    (i_slices[0], j_slices[0])
    #        2) bottom-center,  (i_slices[1], j_slices[0])
    #        3) bottom-right,   (i_slices[2], j_slices[0])
    #        4) middle-left,    (i_slices[0], j_slices[1])
    #        5) middle-right,   (i_slices[2], j_slices[1])
    #        6) top-left,       (i_slices[0], j_slices[2])
    #        7) top-center,     (i_slices[1], j_slices[2])
    #        8) top-right,      (i_slices[2], j_slices[2])
    #
    #     The center block (i_slices[1], j_slices[1]) is omitted => the hole itself.
    #     So we skip that entirely.
    #
    # But your "middle-center" block is *not* in the final list.
    blocks_slices = [
        (i_slices[0], j_slices[0]),  # bottom-left
        (i_slices[1], j_slices[0]),  # bottom-center
        (i_slices[2], j_slices[0]),  # bottom-right
        (i_slices[0], j_slices[1]),  # middle-left
        (i_slices[2], j_slices[1]),  # middle-right
        (i_slices[0], j_slices[2]),  # top-left
        (i_slices[1], j_slices[2]),  # top-center
        (i_slices[2], j_slices[2])   # top-right
    ]

    return blocks_slices

def generate_U_file(folder, i_c, j_c, hole_size, coefficients):
    # -------------------------
    # User Parameters
    # -------------------------
    p = 4              # Number of modes for perturbations
    N = 128            # Mesh resolution (N x N)
    epsilon = 0.05     # Perturbation amplitude
    R = 287.0          # Specific gas constant for air (J/(kg·K))

    # Initial condition parameters
    rho_low = 1.0      # Density for regions y < 0.25 + sigma0 or y > 0.75 + sigma1
    rho_high = 2.0     # Density for regions 0.25 + sigma0 <= y <= 0.75 + sigma1
    vx_low = 0.5       # Velocity vx for rho_low regions
    vx_high = -0.5     # Velocity vx for rho_high regions
    vy = 0.0           # Velocity vy (constant)
    p_initial = 2.5    # Pressure (constant)

    Nx, Ny = 128, 128

    # -------------------------
    # Generate Random Coefficients for Perturbations
    # -------------------------
    # Generate Random Coefficients for Perturbations
    # coefficients = generate_random_coefficients(p)
    alpha0 = coefficients["alpha0"]
    beta0 = coefficients["beta0"]
    alpha1 = coefficients["alpha1"]
    beta1 = coefficients["beta1"]

    logging.info(f"Generated alpha0: {alpha0}")
    logging.info(f"Generated beta0: {beta0}")
    logging.info(f"Generated alpha1: {alpha1}")
    logging.info(f"Generated beta1: {beta1}")

    # -------------------------
    # Define sigma0(x) and sigma1(x)
    # -------------------------
    def sigma(x, alpha, beta, p):
        """
        sigma function for x array, combining p modes.
        x, alpha, beta are 1D arrays or scalars
        """
        val = np.zeros_like(x)
        j_indices = np.arange(1, p+1)
        for j in range(p):
            val += alpha[j] * np.cos(2 * np.pi * j_indices[j] * (x + beta[j]))
        return epsilon * val

    # -------------------------
    # Create Coordinate Arrays for Cell Centers
    # -------------------------
    dx = 1.0 / N
    dy = 1.0 / N

    x_coords = (np.arange(N) + 0.5) * dx  # Cell centers in x
    y_coords = (np.arange(N) + 0.5) * dy  # Cell centers in y

    # Make 2D meshgrid (N x N)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')  # shape: (N, N)

    # -------------------------
    # Compute Perturbations
    # -------------------------
    # We only need sigma0(x) and sigma1(x) for x in [0..N-1], so let's use X[:,0]
    # i.e. the x-values for the first column, repeated across y dimension
    sigma0_1D = sigma(X[:,0], alpha0, beta0, p)  # shape (N,)
    sigma1_1D = sigma(X[:,0], alpha1, beta1, p)  # shape (N,)

    # Broadcast sigma0 and sigma1 across columns
    sigma0_2D = np.tile(sigma0_1D[:, np.newaxis], (1, N))  # shape: (N, N)
    sigma1_2D = np.tile(sigma1_1D[:, np.newaxis], (1, N))  # shape: (N, N)

    # -------------------------
    # Assign Initial Conditions (full NxN)
    # -------------------------
    rho_field_full = np.where(
        (Y < (0.25 + sigma0_2D)) | (Y > (0.75 + sigma1_2D)),
        rho_low,
        rho_high
    )

    vx_field_full = np.where(
        (Y < (0.25 + sigma0_2D)) | (Y > (0.75 + sigma1_2D)),
        vx_low,
        vx_high
    )

    vy_field_full = np.full((N, N), vy)
    p_field_full  = np.full((N, N), p_initial)

    # Temperature using Ideal Gas
    with np.errstate(divide='ignore', invalid='ignore'):
        T_field_full = p_field_full / (rho_field_full * R)
        T_field_full = np.nan_to_num(T_field_full)

    # Log some stats
    logging.info(f"rho_field statistics: min={rho_field_full.min()}, max={rho_field_full.max()}, mean={rho_field_full.mean()}")
    logging.info(f"vx_field statistics: min={vx_field_full.min()}, max={vx_field_full.max()}, mean={vx_field_full.mean()}")
    logging.info(f"vy_field statistics: min={vy_field_full.min()}, max={vy_field_full.max()}, mean={vy_field_full.mean()}")
    logging.info(f"p_field statistics: min={p_field_full.min()}, max={p_field_full.max()}, mean={p_field_full.mean()}")
    logging.info(f"T_field statistics: min={T_field_full.min()}, max={T_field_full.max()}, mean={T_field_full.mean()}")

    # -------------------------
    # Now reorder by blocks to align with blockMesh ordering
    # -------------------------
    rho_blockOrder = []
    p_blockOrder   = []
    T_blockOrder   = []
    vx_blockOrder  = []
    vy_blockOrder  = []

    block_slices = get_block_slices(i_c=i_c, j_c=j_c, N=N, hole_size=hole_size)
    
    for (iRange, jRange) in block_slices:
        for j_ in jRange:
            for i_ in iRange:
                # if (i_hole_min <= i_ <= i_hole_max) and (j_hole_min <= j_ <= j_hole_max):
                #     # It's in the hole or out of domain, skip
                #     continue
                rho_blockOrder.append(rho_field_full[i_, j_])
                p_blockOrder.append(p_field_full[i_, j_])
                T_blockOrder.append(T_field_full[i_, j_])
                vx_blockOrder.append(vx_field_full[i_, j_])
                vy_blockOrder.append(vy_field_full[i_, j_])

    Ntot = len(rho_blockOrder)
    logging.info(f"Block-ordered enumeration: total valid cells = {Ntot}")

    # Finally write
    zero_dir = os.path.join(folder, "0")
    os.makedirs(zero_dir, exist_ok=True)

    write_scalar_field(os.path.join(zero_dir, "rho"), "rho", rho_blockOrder, "[1 -3 0 0 0 0 0]")
    write_scalar_field(os.path.join(zero_dir, "p"),   "p",   p_blockOrder,   "[1 -1 -2 0 0 0 0]")
    write_scalar_field(os.path.join(zero_dir, "T"),   "T",   T_blockOrder,   "[0 0 0 1 0 0 0]")
    write_vector_field(os.path.join(zero_dir, "U"),   "U",   vx_blockOrder, vy_blockOrder, "[0 1 -1 0 0 0 0]")

    logging.info("All fields have been written successfully in block order.")
    print("Initial condition fields (rho, U, p, T) have been generated in the '0' directory, skipping the hole and reordering by blocks.")

    # Optional: Check that U file was created
    u_file = os.path.join(zero_dir, "U")
    if os.path.exists(u_file):
        print(f"U file generated successfully in folder: {folder}")
        logging.info(f"U file generated successfully in folder: {folder}")
    else:
        print(f"Error: U file was not created in {folder}. Check generation logic.")
        logging.error(f"Error: U file was not created in {folder}. Check generation logic.")

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


def generate_blockMeshDict(i_c, j_c, hole_size, output_path="blockMeshDict", run_blockMesh=False):
    """
    Generate a blockMeshDict for a domain [0,1] x [0,1] (z from 0 to 0.1)
    with a square hole of size 8x8 cells, centered on the grid node (i_c, j_c).
    
    Requirements:
    -------------
    1) i_c, j_c must be integers with 4 <= i_c <= 124 and 4 <= j_c <= 124.
    2) The total resolution is 128 x 128 in x,y, so the cell size is 1/128.
    3) The hole is always 8 cells wide and 8 cells tall (8/128 = 0.0625).
    4) The "hole" is the middle block [x1, x2] x [y1, y2], which is omitted.

    The resulting dictionary has the same topology (same vertex indexing,
    same block -> vertex connectivity) as the reference layout with 8 blocks
    around the missing center block.
    """
    # -------------- 1) Basic Input Checks ---------------
    if hole_size <= 0 or hole_size > 128:
        raise ValueError(f"Invalid hole_size: {hole_size} (must be 1..128)")
    half = hole_size / 2.0

    # If we require i_c, j_c, hole_size to be integers and we want an
    # EXACT half, hole_size should be even. Alternatively, if you allow
    # an odd hole_size, you can interpret it carefully or just round half
    # up. For now, assume hole_size is even:
    if hole_size % 2 != 0:
        raise ValueError("hole_size must be an even number for perfect centering.")

    if not (half <= i_c <= 128 - half):
        raise ValueError(f"i_c={i_c} is not valid for hole_size={hole_size}.")
    if not (half <= j_c <= 128 - half):
        raise ValueError(f"j_c={j_c} is not valid for hole_size={hole_size}.")

    # Convert center indices to physical coordinates
    x1 = ((i_c - half)/128) * 64
    x2 = ((i_c + half)/128) * 64
    y1 = ((j_c - half)/128) * 64
    y2 = ((j_c + half)/128) * 64

    Nx_left  = int(i_c - half)      # from 0 to x1
    Nx_hole  = int(hole_size)       # from x1 to x2
    Nx_right = 128 - (int(i_c + half))

    Ny_bottom = int(j_c - half)
    Ny_hole   = int(hole_size)
    Ny_top    = 128 - (int(j_c + half))

    # z-plane thickness
    Lz = 0.1

    # ------------------------------------------------
    # 1. FoamFile header
    # ------------------------------------------------
    text = r"""FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

convertToMeters 1;
"""

    # ------------------------------------------------
    # 2. Define the 32 vertices
    #
    # We'll label them exactly as in the example:
    #
    #  z=0 plane (v0..v15):
    #   v0  = (0   , 0   , 0)
    #   v1  = (x1  , 0   , 0)
    #   v2  = (x2  , 0   , 0)
    #   v3  = (1   , 0   , 0)
    #   v4  = (0   , y1  , 0)
    #   v5  = (x1  , y1  , 0)
    #   v6  = (x2  , y1  , 0)
    #   v7  = (1   , y1  , 0)
    #   v8  = (0   , y2  , 0)
    #   v9  = (x1  , y2  , 0)
    #   v10 = (x2  , y2  , 0)
    #   v11 = (1   , y2  , 0)
    #   v12 = (0   , 1   , 0)
    #   v13 = (x1  , 1   , 0)
    #   v14 = (x2  , 1   , 0)
    #   v15 = (1   , 1   , 0)
    #
    #  z=0.1 plane (v16..v31) same (x,y), but z=0.1
    # ------------------------------------------------

    # Helper to make lines like: "    (0.46875   0   0)    // v1\n"
    def vline(idx, x, y, z):
        return f"    ({x:.6g} {y:.6g} {z:.6g}) // v{idx}\n"

    # Construct the list of (x,y,z) for v0..v15 (z=0)
    coords_z0 = [
        (0   , 0   , 0),
        (x1  , 0   , 0),
        (x2  , 0   , 0),
        (64   , 0   , 0),
        (0   , y1  , 0),
        (x1  , y1  , 0),
        (x2  , y1  , 0),
        (64   , y1  , 0),
        (0   , y2  , 0),
        (x1  , y2  , 0),
        (x2  , y2  , 0),
        (64   , y2  , 0),
        (0   , 64   , 0),
        (x1  , 64   , 0),
        (x2  , 64   , 0),
        (64   , 64   , 0),
    ]
    # For z=0.1 plane, just replicate x,y but z=0.1
    coords_z1 = [(x, y, Lz) for (x,y,_) in coords_z0]

    text += "\nvertices\n(\n"
    for i, (x,y,z) in enumerate(coords_z0):
        text += vline(i, x, y, z)
    for i, (x,y,z) in enumerate(coords_z1, start=16):
        text += vline(i, x, y, z)
    text += ");\n\n"

    # ------------------------------------------------
    # 3. BLOCKS
    #    Each block references the 8 vertices in the form:
    #      hex (v0 v1 v2 v3  v0+16 v1+16 v2+16 v3+16) (Nx Ny 1)
    #
    #    The hole (center block) is omitted entirely in the list
    #    so that region is unmeshed.
    # ------------------------------------------------
    text += "blocks\n(\n"

    # bottom-left
    text += f"    hex (0 1 5 4 16 17 21 20) ({Nx_left} {Ny_bottom} 1) simpleGrading (1 1 1)\n"
    # bottom-center
    text += f"    hex (1 2 6 5 17 18 22 21) ({Nx_hole} {Ny_bottom} 1) simpleGrading (1 1 1)\n"
    # bottom-right
    text += f"    hex (2 3 7 6 18 19 23 22) ({Nx_right} {Ny_bottom} 1) simpleGrading (1 1 1)\n"
    # middle-left
    text += f"    hex (4 5 9 8 20 21 25 24) ({Nx_left} {Ny_hole} 1) simpleGrading (1 1 1)\n"
    # middle-right
    text += f"    hex (6 7 11 10 22 23 27 26) ({Nx_right} {Ny_hole} 1) simpleGrading (1 1 1)\n"
    # top-left
    text += f"    hex (8 9 13 12 24 25 29 28) ({Nx_left} {Ny_top} 1) simpleGrading (1 1 1)\n"
    # top-center
    text += f"    hex (9 10 14 13 25 26 30 29) ({Nx_hole} {Ny_top} 1) simpleGrading (1 1 1)\n"
    # top-right
    text += f"    hex (10 11 15 14 26 27 31 30) ({Nx_right} {Ny_top} 1) simpleGrading (1 1 1)\n"

    text += ");\n\n"

    # ------------------------------------------------
    # 4. EDGES (not used in this simple case)
    # ------------------------------------------------
    text += "edges\n(\n);\n\n"

    # ------------------------------------------------
    # 5. BOUNDARY
    #    The faces are topologically the same as your example.
    #    We reuse your listing verbatim.
    # ------------------------------------------------
    text += r"""boundary
(
    // -------------------------------
    // (A) LEFT boundary (x=0)
    left
    {
        type    patch;  
        faces
        (
            (0 4 20 16)
            (4 8 24 20)
            (8 12 28 24)
        );
    }

    // -------------------------------
    // (B) RIGHT boundary (x=1)
    right
    {
        type    patch;  
        faces
        (
            (3 7 23 19)
            (7 11 27 23)
            (11 15 31 27)
        );
    }

    // -------------------------------
    // (C) BOTTOM boundary (y=0)
    bottom
    {
        type    wall;  
        faces
        (
            (0 1 17 16)
            (1 2 18 17)
            (2 3 19 18)
        );
    }

    // -------------------------------
    // (D) TOP boundary (y=1)
    top
    {
        type    wall;  
        faces
        (
            (12 13 29 28)
            (13 14 30 29)
            (14 15 31 30)
        );
    }

    // -------------------------------
    // (E) HOLE boundary (inner ring)
    hole
    {
        type    wall;  // no-slip wall
        faces
        (
            (5 6 22 21)
            (5 9 25 21)
            (6 10 26 22)
            (9 10 26 25)
        );
    }

    // -------------------------------
    // (F) FRONT-AND-BACK boundary
    frontAndBack
    {
        type    empty;
        faces
        (
            (0 1 5 4)
            (16 20 21 17)

            (1 2 6 5)
            (17 21 22 18)

            (2 3 7 6)
            (18 22 23 19)

            (4 5 9 8)
            (20 24 25 21)

            (6 7 11 10)
            (22 26 27 23)

            (8 9 13 12)
            (24 28 29 25)

            (9 10 14 13)
            (25 29 30 26)

            (10 11 15 14)
            (26 30 31 27)
        );
    }
);
"""

    # ------------------------------------------------
    # 6. MERGE PATCH PAIRS (if needed)
    # ------------------------------------------------
    text += "\nmergePatchPairs\n(\n);\n\n"

    # ------------------------------------------------
    # 7. Write out the file
    # ------------------------------------------------
    with open(output_path, "w") as f:
        f.write(text)

    print(f"[INFO] blockMeshDict written to {output_path}")
    print(f"       Hole center @ i_c={i_c}, j_c={j_c}  =>  (x1={x1:.4f}, x2={x2:.4f}), (y1={y1:.4f}, y2={y2:.4f})")
    print(f"       Nx_left={Nx_left}, Nx_hole={Nx_hole}, Nx_right={Nx_right}  --  Ny_bottom={Ny_bottom}, Ny_hole={Ny_hole}, Ny_top={Ny_top}")
     # Optionally run blockMesh from one folder above 'system'
    if run_blockMesh:
        # system_dir = .../design_point_X/system
        system_dir = os.path.dirname(os.path.abspath(output_path))
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

def random_hole_centers(
    num_centers, 
    N, 
    min_hole_size=16, 
    max_hole_size=32
):
    """
    Generate 'num_centers' hole definitions, all placed exactly at the
    *center* of the N x N grid (i_c = j_c = N//2). The hole size is chosen
    randomly from the even integers in [min_hole_size..max_hole_size].

    Parameters
    ----------
    num_centers : int
        Number of hole definitions to generate.
    N : int
        Grid resolution (N x N).
    min_hole_size : int
        Minimum hole size (must be even).
    max_hole_size : int
        Maximum hole size (must be even).

    Returns
    -------
    centers : list of tuples
        Each tuple is (i_c, j_c, hole_size). i_c, j_c are integers
        at the domain center, and hole_size is even.
    """
    # Basic checks
    if min_hole_size % 2 != 0 or max_hole_size % 2 != 0:
        raise ValueError("Both min_hole_size and max_hole_size must be even.")
    if min_hole_size < 2 or max_hole_size > N:
        raise ValueError("Hole sizes must be between 2 and N (inclusive).")
    if min_hole_size > max_hole_size:
        raise ValueError("min_hole_size must be <= max_hole_size.")

    centers = []
    # The center of the domain
    i_c = N // 2
    j_c = N // 2

    # For each "copy," pick a random even hole size
    possible_sizes = range(min_hole_size, max_hole_size + 1, 2)

    for _ in range(num_centers):
        hole_size = np.random.choice(possible_sizes)
        centers.append((i_c, j_c, hole_size))

    # For debug logging, you can show the final hole:
    # (Though each may differ in hole_size, i_c,j_c is the same)
    cx = i_c / N
    cy = j_c / N
    logging.info(f"All holes centered at (i_c, j_c) = ({i_c},{j_c}) => (cx, cy)=({cx:.3f},{cy:.3f})")
    logging.info(f"Sample hole size in [min={min_hole_size}, max={max_hole_size}] => even integers")
    logging.info(f"hole size in {hole_size}")

    return centers

def count_hole_cells(coords, x_min, x_max, y_min, y_max):
    """
    Counts the number of cells within the specified hole region.

    :param coords: NumPy array of cell center coordinates.
    :param x_min: Minimum x-coordinate of the hole.
    :param x_max: Maximum x-coordinate of the hole.
    :param y_min: Minimum y-coordinate of the hole.
    :param y_max: Maximum y-coordinate of the hole.
    :return: Number of cells in the hole region, total number of cells.
    """
    # Debug: Log coordinate bounds
    logging.info(f"Checking hole bounds: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")

    # Count cells within bounds
    in_hole = (coords[:, 0] >= x_min) & (coords[:, 0] <= x_max) & \
              (coords[:, 1] >= y_min) & (coords[:, 1] <= y_max)

    hole_cells = np.sum(in_hole)
    total_cells = coords.shape[0]

    logging.info(f"Found {hole_cells} cells in hole region out of {total_cells} total cells.")
    return hole_cells, total_cells

def parse_cell_centers(c_file_path, precision=6):
    """
    Parses the cell center coordinates from the OpenFOAM 'C' file.

    :param c_file_path: Path to the 'C' file generated by OpenFOAM.
    :param precision: Decimal precision to round x and y coordinates.
    :return: NumPy array of cell center coordinates.
    """
    with open(c_file_path, "r") as f:
        lines = f.readlines()

    num_points = None
    start_index = None

    # Find 'internalField' line, parse number of points
    for i, line in enumerate(lines):
        if line.strip().startswith("internalField"):
            num_points = int(lines[i + 1].strip())  # The number after 'nonuniform List<vector>'
            start_index = i + 3  # Coordinates start 2 lines after the count line
            break

    if num_points is None or start_index is None:
        raise ValueError("Could not locate 'internalField' or number of points in the C file.")

    # Collect coordinates
    coords = []
    for line in lines[start_index:start_index + num_points]:
        line = line.strip("()\n")  # Remove parentheses and newline
        x_str, y_str, z_str = line.split()
        x, y, z = map(float, (x_str, y_str, z_str))
        coords.append((x, y, z))

    coords = np.array(coords)

    # Round x, y coordinates
    coords[:, 0] = np.round(coords[:, 0], precision)
    coords[:, 1] = np.round(coords[:, 1], precision)

    return coords

def validate_missing_coordinates(coords, hole_size=0.0625):
    """
    Identifies missing x and y coordinates based on the hole region.

    :param coords: NumPy array of cell center coordinates.
    :param cx: Hole center x-coordinate.
    :param cy: Hole center y-coordinate.
    :param hole_size: Size of the hole (side length).
    :return: Unique missing x and y coordinates.
    """
    half_size = hole_size / 2
    # x_min, x_max = cx - half_size, cx + half_size
    # y_min, y_max = cy - half_size, cy + half_size

    # Identify unique x and y coordinates
    x_unique, x_counts = np.unique(coords[:, 0], return_counts=True)
    y_unique, y_counts = np.unique(coords[:, 1], return_counts=True)

    # Identify missing coordinates
    expected_count = len(y_unique)  # Expected count for a fully populated mesh
    missing_x = x_unique[x_counts < expected_count]
    missing_y = y_unique[y_counts < expected_count]

    return missing_x, missing_y

def load_missing_xy(input_text, precision=6):
    """
    Parse the *original* partial coordinate text, find missing_x, missing_y,
    return them as arrays. Does NOT write or append anything.
    """
    lines = input_text.splitlines()

    num_points = None
    start_index = None

    for i, line in enumerate(lines):
        if line.strip().startswith("internalField"):
            num_points = int(lines[i + 1].strip())
            start_index = i + 3
            break

    coords = []
    for line in lines[start_index:start_index + num_points]:
        line = line.strip("()\n")
        x_str, y_str, z_str = line.split()
        x, y, z = map(float, (x_str, y_str, z_str))
        coords.append((x, y, z))
    coords = np.array(coords)
    coords[:, 0] = np.round(coords[:, 0], precision)
    coords[:, 1] = np.round(coords[:, 1], precision)

    x_unique, x_counts = np.unique(coords[:, 0], return_counts=True)
    y_unique, y_counts = np.unique(coords[:, 1], return_counts=True)

    expected_count_x = len(y_unique)
    expected_count_y = len(x_unique)
    missing_x = x_unique[x_counts < expected_count_x]
    missing_y = y_unique[y_counts < expected_count_y]

    return missing_x, missing_y

def plot_mesh_centers_and_detect_size(input_text, output_path, precision=6):
    """
    1) Detect how many x,y are missing => Nx, Ny
    2) Append missing centers => write updated coordinate file => full 128*128
    3) Return Nx.

    We assume the hole is square => Nx == Ny
    """
    lines = input_text.splitlines()

    num_points = None
    start_index = None

    # 1) Find internalField
    for i, line in enumerate(lines):
        if line.strip().startswith("internalField"):
            num_points = int(lines[i + 1].strip())
            start_index = i + 3
            break

    coords = []
    for line in lines[start_index:start_index + num_points]:
        line = line.strip("()\n")
        x_str, y_str, z_str = line.split()
        x, y, z = map(float, (x_str, y_str, z_str))
        coords.append((x, y, z))
    coords = np.array(coords)

    coords[:, 0] = np.round(coords[:, 0], precision)
    coords[:, 1] = np.round(coords[:, 1], precision)

    x_unique, x_counts = np.unique(coords[:, 0], return_counts=True)
    y_unique, y_counts = np.unique(coords[:, 1], return_counts=True)

    expected_count_x = len(y_unique)
    expected_count_y = len(x_unique)

    missing_x = x_unique[x_counts < expected_count_x]
    missing_y = y_unique[y_counts < expected_count_y]

    Nx = len(missing_x)
    Ny = len(missing_y)
    if Nx != Ny:
        raise ValueError(f"Hole is not square! Nx={Nx}, Ny={Ny}")

    z_val = coords[0, 2]
    hole_centers = [(mx, my, z_val) for mx in missing_x for my in missing_y]
    all_coords = np.concatenate([coords, hole_centers], axis=0)

    # Write updated file => full 128*128
    with open(output_path, 'w') as f:
        f.write(f"{len(all_coords)}\n(\n")
        for (x, y, z) in all_coords:
            f.write(f"({x:.{precision}f} {y:.{precision}f} {z:.{precision}f})\n")
        f.write(")\n")

    return Nx

def load_coordinates(C_file_path):
    """
    Load x and y coordinates from the updated C file.

    Args:
        C_file_path (str): Path to the updated C file.

    Returns:
        tuple: (x_coords, y_coords) as 2D arrays reshaped to (128, 128).
    """
    with open(C_file_path, "r") as file:
        lines = file.readlines()

    # Extract the internalField coordinates
    start_index = 2
    num_points = int(lines[0].strip())

    coords = []
    for line in lines[start_index : start_index + num_points]:
        x, y, _ = map(float, line.strip("()\n").split())
        coords.append((x, y))

    coords = np.array(coords)
    x_coords = coords[:, 0].reshape(128, 128)
    y_coords = coords[:, 1].reshape(128, 128)

    return x_coords, y_coords

logger = logging.getLogger(__name__)
def process_single_sim_variable_hole(
    this_data,
    c_content,
    output_folder,
    sim_index,
    precision=6
):
    """
    Process one simulation's data with a variable-size hole.
    1) Find missing Xs, missing Ys => Nx, Ny
    2) Append Nx*Ny dummy cells => reshape to (128,128)
    3) Create a mask that zeros out exactly those Nx*Ny coordinates
       in the final 2D layout.

    Args:
        this_data: np.ndarray, shape (21, num_cells, channels) or (N,21,num_cells,channels)
        c_content: string contents of the partial coordinate file with missing hole
        output_folder: where to write the updated coordinate file
        sim_index: integer for naming
        precision: rounding for coordinate detection

    Returns:
        final_tensor: PyTorch tensor, shape (1,21,128,128,[channels+something])
        c_output: Path to the updated coordinate file
    """
    # 0) Ensure shape => (1,21,num_cells,channels)
    if len(this_data.shape) == 3:
        this_data = this_data[None, ...]  # => (1,21,num_cells,channels)
    logger.info(f"Input data shape: {this_data.shape}")

    # 1) Write a new C file with appended hole cell-centers
    c_output = os.path.join(output_folder, f"C_updated_{sim_index}.txt")
    Nx = plot_mesh_centers_and_detect_size(c_content, c_output, precision=precision)
    num_cells = this_data.shape[2]  
    print("Original #cells from simulation:", num_cells)
    logging.info(f"Original #cells from simulation {num_cells}")
    # Nx = the dimension of the hole in x *and* y (we assume it's square, Nx x Nx)
    hole_cell_count = Nx * Nx
    print("Detected Nx:", Nx, " => Nx*Nx =", hole_cell_count)
    logging.info(f"Detected Nx:{Nx} => Nx*Nx={hole_cell_count}")

    # 2) Add Nx*Nx "dummy" cells => shape => (1,21,num_cells + Nx*Nx,channels)
    hole_cell_count = Nx * Nx
    data_with_holes = np.concatenate([
        this_data,
        np.full(
            (this_data.shape[0], this_data.shape[1], hole_cell_count, this_data.shape[3]),
            -100,
            dtype=this_data.dtype
        )
    ], axis=2)
    logger.info(f"After adding {hole_cell_count} dummy cells: {data_with_holes.shape}")

    # 3) Reshape => (1,21,128,128,channels)
    data_reshaped = data_with_holes.reshape(
        data_with_holes.shape[0],
        data_with_holes.shape[1],
        128, 128,
        data_with_holes.shape[3]
    )
    logger.info(f"Reshaped to {data_reshaped.shape}")

    # 4) Load full coordinates from c_output => 128*128
    x_coords, y_coords = load_coordinates(c_output)  # => shape (128,128)

    # 5) Tile x,y => shape (1,21,128,128)
    xv = np.tile(x_coords[None, None, :, :], (data_reshaped.shape[0], data_reshaped.shape[1], 1, 1))
    yv = np.tile(y_coords[None, None, :, :], (data_reshaped.shape[0], data_reshaped.shape[1], 1, 1))

    # 6) Concatenate => shape (1,21,128,128, channels+2)
    data_with_coords = np.concatenate([data_reshaped, xv[..., None], yv[..., None]], axis=-1)

    # 7) Create a mask of ones => shape (1,21,128,128)
    mask = np.ones(
        (data_reshaped.shape[0], data_reshaped.shape[1], 128, 128),
        dtype=data_reshaped.dtype
    )

    # 8) Zero out the missing cells by coordinate matching
    #    We'll parse them again from the partial hole detection step
    missing_x, missing_y = load_missing_xy(c_content, precision=precision)
    # missing_x => the 1D array of Nx distinct x-values that were missing
    # missing_y => the 1D array of Nx distinct y-values that were missing

    # For each x in missing_x, y in missing_y => find (i,j) in [0..127]
    # We'll do a direct comparison with x_coords, y_coords. Because of rounding,
    # we must be sure to use a tolerance-based search or direct float compare if safe.

    for mx in missing_x:
        for my in missing_y:
            # Find all (i,j) with x_coords[i,j]==mx and y_coords[i,j]==my
            # We'll do a boolean mask approach
            i_j_mask = np.isclose(x_coords, mx, atol=1e-12) & np.isclose(y_coords, my, atol=1e-12)
            # i_j_mask shape => (128,128)
            mask[:, :, i_j_mask] = 0.0  # set zeros for the entire time dimension (21 frames)

    # 9) Concatenate mask => final shape (1,21,128,128, channels+3)
    data_final = np.concatenate([data_with_coords, mask[..., None]], axis=-1)
    logger.info(f"Final shape (with mask): {data_final.shape}")

    # 10) Convert to torch
    final_tensor = torch.from_numpy(data_final).float()

    return final_tensor, c_output

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

import os
import time
import json
import shutil
import logging
import numpy as np
from tqdm import tqdm

def main():
    total_trajectories = int(input("Enter the total number of trajectories to simulate: "))
    batch_size = 128  # Adjust based on memory availability
    main_folder = "Design_Point_0"
    start_time = time.time()
    batch_id = 0
    processed_batches = []

    # Generate hole positions once for all trajectories
    N = 128
    min_size = 16
    max_size = 64
    hole_centers = random_hole_centers(total_trajectories, N, min_size, max_size)

    if total_trajectories > len(hole_centers):
        raise ValueError("Not enough hole centers specified for the number of folders.")

    trajectories_done = 0  # Track how many have been processed

    # --- Process in Batches ---
    while trajectories_done < total_trajectories:
        current_batch_size = min(batch_size, total_trajectories - trajectories_done)

        # Step 1: Create a batch of folders
        batch_folders = copy_main_folder(main_folder, current_batch_size)
        batch_hole_centers = hole_centers[trajectories_done:trajectories_done + current_batch_size]
        sim_data = []
        converged_folders = []

        # Step 2: Run simulations for this batch
        for i, folder in enumerate(batch_folders):
            i_c, j_c, hole_size = batch_hole_centers[i]
            logging.info(f"Processing folder {folder} with hole center ({i_c:.4f}, {j_c:.4f})")

            # Generate mesh
            bmd_path = os.path.join(folder, "system", "blockMeshDict")
            try:
                generate_blockMeshDict(i_c, j_c, hole_size=hole_size, output_path=bmd_path, run_blockMesh=True)
            except subprocess.CalledProcessError as e:
                logging.error(f"BlockMesh failed for {folder}: {e}")
                continue  # Skip this folder

            update_Umax_in_simulation_folder(folder, min_value=1.125e-3, max_value=1.125e-1)

            if not run_rhoPimpleFoam(folder):
                logging.warning(f"Solver failed for folder {folder}. Skipping.")
                continue

            # Generate and validate cell centers
            c_file = generate_cell_centers(folder)
            if not c_file:
                logging.error(f"Cell centers file not found for {folder}. Skipping.")
                continue

            converged_folders.append(folder)

            with open(c_file, "r") as f:
                c_contents = f.read()

            sim_data.append({
                "folder": folder,
                "c_contents": c_contents
            })

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
            final_time_dirs, final_data = gather_all_simulations(converged_folders)
            if final_data is None:
                logging.error("No simulations successfully processed in this batch. Skipping.")
            else:
                # Save batch dataset
                batch_file = f"all_sims_data_batch_{batch_id}.npy"
                np.save(batch_file, final_data)
                logging.info(f"Batch {batch_id} dataset shape: {final_data.shape} saved as {batch_file}")

                # Append time directories
                with open("time_dirs.txt", "a") as f:
                    for t in final_time_dirs:
                        f.write(str(t) + "\n")

                processed_batches.append(batch_file)
                batch_id += 1

        # Step 5: Delete folders in the current batch
        for folder in batch_folders:
            try:
                shutil.rmtree(folder)
                logging.info(f"Deleted folder {folder}")
            except Exception as e:
                logging.error(f"Error deleting folder {folder}: {e}")

        # Move to the next batch
        trajectories_done += current_batch_size

    # Final log: Summarize results
    if processed_batches:
        logging.info(f"Successfully processed {len(processed_batches)} batches.")
        logging.info("All done!")
    else:
        logging.error("No successful simulations. Exiting.")

    total_time_elapsed = time.time() - start_time
    print(f"\nTotal execution time: {total_time_elapsed:.2f} seconds")
    logging.info(f"Total execution time: {total_time_elapsed:.2f} seconds")

if __name__ == "__main__":
    main()

