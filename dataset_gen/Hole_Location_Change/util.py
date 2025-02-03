import os
import shutil
import subprocess
import time
import numpy as np
import logging
from datetime import datetime, timedelta
import re
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

def get_block_slices(i_c, j_c, N=128, hole_size=8):
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

def run_icoFoam(folder):
    """
    Runs icoFoam for the specified folder and returns True if it converged, False otherwise.
    """
    command = ["icoFoam"]
    print(f"Running icoFoam in folder: {folder}")
    
    try:
        process = subprocess.Popen(command, cwd=folder)
        process.communicate()

        # Check the return code
        if process.returncode == 0:
            print(f"icoFoam simulation completed successfully in folder: {folder}")
            logging.info(f"icoFoam simulation completed successfully in folder: {folder}")
            return True
        else:
            print(f"icoFoam simulation failed or did not converge in folder: {folder}")
            logging.warning(f"icoFoam simulation failed or did not converge in folder: {folder}")
            return False

    except FileNotFoundError:
        print("Error: icoFoam command not found.")
        logging.error("Error: icoFoam command not found.")
        return False

    except Exception as e:
        print(f"Error running icoFoam: {e}")
        logging.error(f"Error running icoFoam: {e}")
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

import os
import re
import numpy as np

def parse_internal_field(file_path, field_type="vector"):
    """
    Parses the internal field data from an OpenFOAM field file.

    If field_type == "vector", it will capture the first two velocity
    components (Ux, Uy) if you only want 2D.

    Args:
        file_path (str): Path to the OpenFOAM field file.
        field_type (str): Either "vector" (e.g., U) or "scalar" (e.g., p or rho).

    Returns:
        np.ndarray:
            - If field_type == "vector": shape = (n_points, 2) with Ux and Uy.
            - If field_type == "scalar": shape = (n_points,).

    Raises:
        FileNotFoundError: If the file is not found.
        ValueError: If the file format is unexpected or parsing fails.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    data_lines = []
    inside_data_block = False
    n_points = None  # Use None initially to ensure it gets set correctly

    with open(file_path, "r") as f:
        for line in f:
            # Detect the internalField block
            if "internalField" in line and "nonuniform" in line:
                inside_data_block = True
                continue

            if inside_data_block:
                stripped_line = line.strip()

                # Look for the number of points immediately after "internalField"
                if n_points is None and stripped_line.isdigit():
                    n_points = int(stripped_line)
                    continue

                # Skip the opening parenthesis "("
                if stripped_line == "(":
                    continue

                # End of the data block
                if stripped_line in [")", ");"]:
                    inside_data_block = False
                    continue

                # If we reach here, it's a data line
                data_lines.append(stripped_line)

    # Ensure we found the internalField data
    if n_points is None or not data_lines:
        raise ValueError(f"No internalField data found in {file_path}")

    # Parse the data
    if field_type == "vector":
        vector_data = []
        for dl in data_lines:
            # e.g. (0.184065 0.370922 0)
            dl = dl.strip("()")  # Remove parentheses
            comps = dl.split()
            # Convert only the first two components (Ux, Uy)
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

def parse_simulation(sim_folder, pde_name="default"):
    """
    Parses a single simulation folder containing time-step directories.

    For NS_Gauss (incompressible flow), skips temperature (T) and density (rho).

    Returns:
        time_steps: Sorted list of time-step strings (e.g. ["0", "0.1", "0.2", ...])
        results_array: np.ndarray of shape (num_time_steps, num_points, 4) 
                       for channels [rho, Ux, Uy, p] or [Ux, Uy, p] for NS_Gauss.
    """
    # Collect all numeric time directories
    all_entries = os.listdir(sim_folder)
    time_dirs = [entry for entry in all_entries if entry.replace('.', '', 1).isdigit() and os.path.isdir(os.path.join(sim_folder, entry))]
    time_dirs.sort(key=lambda x: float(x))

    results_per_time = []
    for tdir in time_dirs:
        this_tdir_path = os.path.join(sim_folder, tdir)

        # File paths
        u_file = os.path.join(this_tdir_path, "U")
        

        # Parse velocity and pressure
        u_data = parse_internal_field(u_file, field_type="vector")   # shape=(n_points, 2)
        

        if pde_name != "ns_gauss":
            # Parse temperature and density for other PDEs
            T_file = os.path.join(this_tdir_path, "T")
            p_file = os.path.join(this_tdir_path, "p")
            T_data = parse_internal_field(T_file, field_type="scalar")
            p_data = parse_internal_field(p_file, field_type="scalar")   # shape=(n_points,)

            try:
                rho_data = p_data / (T_data * 287)
            except ZeroDivisionError:
                raise ValueError("Encountered division by zero")

            # Combine [rho, Ux, Uy, p]
            combined_data = np.column_stack([rho_data, u_data[:, 0], u_data[:, 1], p_data])
        else:
            # For NS_Gauss, skip T and rho
            combined_data = np.column_stack([u_data[:, 0], u_data[:, 1]])  # Shape: (n_points, 3)

        results_per_time.append(combined_data)

    results_array = np.stack(results_per_time, axis=0)
    return time_dirs, results_array

def reconstruct_row_by_row(data, hole_centers, grid_size=128, hole_size=8):
    num_trajectories, timesteps, _, channels = data.shape
    full_data = np.zeros((num_trajectories, timesteps, grid_size, grid_size, channels+1))

    for traj_idx in range(num_trajectories):
        i_c, j_c = hole_centers[traj_idx]  # Hole center for this trajectory
        block_slices = get_block_slices(i_c, j_c, N=grid_size, hole_size=hole_size)

        # Create hole mask (1 for hole, 0 for valid points)
        hole_mask = np.zeros((grid_size, grid_size))
        half = hole_size // 2
        i_hole_min = i_c - half
        i_hole_max = i_hole_min + hole_size - 1  # Corrected
        j_hole_min = j_c - half
        j_hole_max = j_hole_min + hole_size - 1  # Corrected

        # Apply mask
        hole_mask[i_hole_min:i_hole_max + 1, j_hole_min:j_hole_max + 1] = 1


        for t in range(timesteps):
            # Initialize grid with zeros (hole regions will remain zero)
            grid = np.zeros((grid_size, grid_size, channels))  

            flat_index = 0  # Index to track the flattened OpenFOAM data

            # Place values in the **row-by-row** grid (reversing block order)
            for (i_range, j_range) in block_slices:
                for j_ in j_range:       # Loop over rows first (row-major order)
                    for i_ in i_range:   # Then columns
                        grid[i_, j_, :] = data[traj_idx, t, flat_index, :]
                        flat_index += 1  # Move to the next data point

            combined = np.concatenate((grid, hole_mask[..., np.newaxis]), axis=-1)
            # Store the reconstructed grid
            full_data[traj_idx, t] = combined

    return full_data


def gather_all_simulations(sim_folders, hole_centers, pde_name="default"):
    all_sim_results = []
    final_time_dirs = None

    for i, folder in enumerate(sim_folders):
        logging.info(f"Parsing simulation folder {folder}")
        tdirs, results_array = parse_simulation(folder, pde_name=pde_name)  # Pass pde_name

        if i == 0:
            final_time_dirs = tdirs
        else:
            if tdirs != final_time_dirs:
                logging.warning(f"Simulation {folder} has different time steps than the first simulation.")

        full_grid_data = reconstruct_row_by_row(
            results_array[np.newaxis, ...],
            [hole_centers[i]]
        )[0]
        all_sim_results.append(full_grid_data)

    final_results = np.stack(all_sim_results, axis=0)
    return final_time_dirs, final_results


def generate_blockMeshDict(i_c, j_c, output_path="blockMeshDict", run_blockMesh=False):
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
    # Check input
    if not(4 <= i_c <= 124):
        raise ValueError("i_c must be in [4..124] to keep a 4-cell margin.")
    if not(4 <= j_c <= 124):
        raise ValueError("j_c must be in [4..124] to keep a 4-cell margin.")

    # Convert center indices to physical coordinates
    x1 = (i_c - 4)/128
    x2 = (i_c + 4)/128
    y1 = (j_c - 4)/128
    y2 = (j_c + 4)/128

    # Number of cells in x and y directions for each sub-block
    Nx_left   = i_c - 4      # from 0 to x1
    Nx_hole   = 8            # from x1 to x2
    Nx_right  = 128 - (i_c + 4)  # from x2 to 1

    Ny_bottom = j_c - 4
    Ny_hole   = 8
    Ny_top    = 128 - (j_c + 4)

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
        (1   , 0   , 0),
        (0   , y1  , 0),
        (x1  , y1  , 0),
        (x2  , y1  , 0),
        (1   , y1  , 0),
        (0   , y2  , 0),
        (x1  , y2  , 0),
        (x2  , y2  , 0),
        (1   , y2  , 0),
        (0   , 1   , 0),
        (x1  , 1   , 0),
        (x2  , 1   , 0),
        (1   , 1   , 0),
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
        type    cyclic;  // or "wall" or "cyclic"
        neighbourPatch  right;
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
        type    cyclic;  // or "wall" or "cyclic"
        neighbourPatch  left;
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
        type    cyclic;  // or "wall" or "cyclic"
        neighbourPatch  top;
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
        type    cyclic;  // or "wall" or "cyclic"
        neighbourPatch  bottom;
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

def random_hole_centers(num_centers, N):
    """
    Generate random hole centers (cx, cy) such that:
    - The hole fits within the domain [0, 1].
    - The centers are restricted to a feasible range [lower_limit, upper_limit].

    Parameters:
    - num_centers: int, number of random hole centers to generate.
    - N: int, grid resolution (N x N).
    - lower_limit: float, minimum value for cx, cy (default 0.2).
    - upper_limit: float, maximum value for cx, cy (default 0.8).

    Returns:
    - List of (cx, cy) tuples.
    """
    centers = []
    for _ in range(num_centers):
        # Pick an integer center i_c in [4, N - 4]
        i_c = np.random.randint(4, N - 3)  # range(4, N-3) => up to N-4 inclusive
        j_c = np.random.randint(4, N - 3)

        # Convert these integer centers to continuous coordinates
        cx = i_c / N
        cy = j_c / N

        # centers.append((cx, cy))
        centers.append((i_c, j_c))
    logging.info(f"The center of the hole is:{cx}")
    logging.info(f"The center of the hole is:{cy}")
    logging.info(f"The initial calculated x_min is:{cx - 0.0625}")
    logging.info(f"The initial calculated x_max is:{cx + 0.0625}")
    logging.info(f"The initial calculated y_min is:{cy - 0.0625}")
    logging.info(f"The initial calculated y_max is:{cy + 0.0625}")
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