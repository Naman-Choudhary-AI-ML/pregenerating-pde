import os
import shutil
import subprocess
import time
import numpy as np
import logging
from datetime import datetime, timedelta
import re
import json
import random
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

# def generate_random_coefficients(p):
#     """
#     Generate random coefficients for perturbations.
    
#     Args:
#         p (int): Number of modes for perturbations.
        
#     Returns:
#         dict: A dictionary containing random coefficients for alpha0, beta0, alpha1, and beta1.
#     """
#     return {
#         "alpha0": np.random.uniform(0, 1, p),
#         "beta0": np.random.uniform(0, 1, p),
#         "alpha1": np.random.uniform(0, 1, p),
#         "beta1": np.random.uniform(0, 1, p),
#     }

def get_block_slices(i_c, j_c, N=128, hole_size=16):
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

# def generate_U_file(folder, i_c, j_c, coefficients, hole_size=0.0625):
#     # -------------------------
#     # User Parameters
#     # -------------------------
#     p = 4              # Number of modes for perturbations
#     N = 128            # Mesh resolution (N x N)
#     epsilon = 0.05     # Perturbation amplitude
#     R = 287.0          # Specific gas constant for air (J/(kg·K))

#     # Initial condition parameters
#     rho_low = 1.0      # Density for regions y < 0.25 + sigma0 or y > 0.75 + sigma1
#     rho_high = 2.0     # Density for regions 0.25 + sigma0 <= y <= 0.75 + sigma1
#     vx_low = 0.5       # Velocity vx for rho_low regions
#     vx_high = -0.5     # Velocity vx for rho_high regions
#     vy = 0.0           # Velocity vy (constant)
#     p_initial = 2.5    # Pressure (constant)

#     Nx, Ny = 128, 128

#     # -------------------------
#     # Generate Random Coefficients for Perturbations
#     # -------------------------
#     # Generate Random Coefficients for Perturbations
#     # coefficients = generate_random_coefficients(p)
#     alpha0 = coefficients["alpha0"]
#     beta0 = coefficients["beta0"]
#     alpha1 = coefficients["alpha1"]
#     beta1 = coefficients["beta1"]

#     logging.info(f"Generated alpha0: {alpha0}")
#     logging.info(f"Generated beta0: {beta0}")
#     logging.info(f"Generated alpha1: {alpha1}")
#     logging.info(f"Generated beta1: {beta1}")

#     # -------------------------
#     # Define sigma0(x) and sigma1(x)
#     # -------------------------
#     def sigma(x, alpha, beta, p):
#         """
#         sigma function for x array, combining p modes.
#         x, alpha, beta are 1D arrays or scalars
#         """
#         val = np.zeros_like(x)
#         j_indices = np.arange(1, p+1)
#         for j in range(p):
#             val += alpha[j] * np.cos(2 * np.pi * j_indices[j] * (x + beta[j]))
#         return epsilon * val

#     # -------------------------
#     # Create Coordinate Arrays for Cell Centers
#     # -------------------------
#     dx = 1.0 / N
#     dy = 1.0 / N

#     x_coords = (np.arange(N) + 0.5) * dx  # Cell centers in x
#     y_coords = (np.arange(N) + 0.5) * dy  # Cell centers in y

#     # Make 2D meshgrid (N x N)
#     X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')  # shape: (N, N)

#     # -------------------------
#     # Compute Perturbations
#     # -------------------------
#     # We only need sigma0(x) and sigma1(x) for x in [0..N-1], so let's use X[:,0]
#     # i.e. the x-values for the first column, repeated across y dimension
#     sigma0_1D = sigma(X[:,0], alpha0, beta0, p)  # shape (N,)
#     sigma1_1D = sigma(X[:,0], alpha1, beta1, p)  # shape (N,)

#     # Broadcast sigma0 and sigma1 across columns
#     sigma0_2D = np.tile(sigma0_1D[:, np.newaxis], (1, N))  # shape: (N, N)
#     sigma1_2D = np.tile(sigma1_1D[:, np.newaxis], (1, N))  # shape: (N, N)

#     # -------------------------
#     # Assign Initial Conditions (full NxN)
#     # -------------------------
#     rho_field_full = np.where(
#         (Y < (0.25 + sigma0_2D)) | (Y > (0.75 + sigma1_2D)),
#         rho_low,
#         rho_high
#     )

#     vx_field_full = np.where(
#         (Y < (0.25 + sigma0_2D)) | (Y > (0.75 + sigma1_2D)),
#         vx_low,
#         vx_high
#     )

#     vy_field_full = np.full((N, N), vy)
#     p_field_full  = np.full((N, N), p_initial)

#     # Temperature using Ideal Gas
#     with np.errstate(divide='ignore', invalid='ignore'):
#         T_field_full = p_field_full / (rho_field_full * R)
#         T_field_full = np.nan_to_num(T_field_full)

#     # Log some stats
#     logging.info(f"rho_field statistics: min={rho_field_full.min()}, max={rho_field_full.max()}, mean={rho_field_full.mean()}")
#     logging.info(f"vx_field statistics: min={vx_field_full.min()}, max={vx_field_full.max()}, mean={vx_field_full.mean()}")
#     logging.info(f"vy_field statistics: min={vy_field_full.min()}, max={vy_field_full.max()}, mean={vy_field_full.mean()}")
#     logging.info(f"p_field statistics: min={p_field_full.min()}, max={p_field_full.max()}, mean={p_field_full.mean()}")
#     logging.info(f"T_field statistics: min={T_field_full.min()}, max={T_field_full.max()}, mean={T_field_full.mean()}")

#     # -------------------------
#     # Now reorder by blocks to align with blockMesh ordering
#     # -------------------------
#     rho_blockOrder = []
#     p_blockOrder   = []
#     T_blockOrder   = []
#     vx_blockOrder  = []
#     vy_blockOrder  = []

#     block_slices = get_block_slices(i_c=i_c, j_c=j_c, N=N, hole_size=8)
    
#     for (iRange, jRange) in block_slices:
#         for j_ in jRange:
#             for i_ in iRange:
#                 # if (i_hole_min <= i_ <= i_hole_max) and (j_hole_min <= j_ <= j_hole_max):
#                 #     # It's in the hole or out of domain, skip
#                 #     continue
#                 rho_blockOrder.append(rho_field_full[i_, j_])
#                 p_blockOrder.append(p_field_full[i_, j_])
#                 T_blockOrder.append(T_field_full[i_, j_])
#                 vx_blockOrder.append(vx_field_full[i_, j_])
#                 vy_blockOrder.append(vy_field_full[i_, j_])

#     Ntot = len(rho_blockOrder)
#     logging.info(f"Block-ordered enumeration: total valid cells = {Ntot}")

#     # Finally write
#     zero_dir = os.path.join(folder, "0")
#     os.makedirs(zero_dir, exist_ok=True)

#     write_scalar_field(os.path.join(zero_dir, "rho"), "rho", rho_blockOrder, "[1 -3 0 0 0 0 0]")
#     write_scalar_field(os.path.join(zero_dir, "p"),   "p",   p_blockOrder,   "[1 -1 -2 0 0 0 0]")
#     write_scalar_field(os.path.join(zero_dir, "T"),   "T",   T_blockOrder,   "[0 0 0 1 0 0 0]")
#     write_vector_field(os.path.join(zero_dir, "U"),   "U",   vx_blockOrder, vy_blockOrder, "[0 1 -1 0 0 0 0]")

#     logging.info("All fields have been written successfully in block order.")
#     print("Initial condition fields (rho, U, p, T) have been generated in the '0' directory, skipping the hole and reordering by blocks.")

#     # Optional: Check that U file was created
#     u_file = os.path.join(zero_dir, "U")
#     if os.path.exists(u_file):
#         print(f"U file generated successfully in folder: {folder}")
#         logging.info(f"U file generated successfully in folder: {folder}")
#     else:
#         print(f"Error: U file was not created in {folder}. Check generation logic.")
#         logging.error(f"Error: U file was not created in {folder}. Check generation logic.")

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

# def run_setfields(folder):
#     """Runs setFields in the specified folder."""
#     command = ["setFields"]
#     print(f"Running setFields in folder: {folder}")
#     logging.info(f"Running setFields in folder: {folder}")
#     try:
#         process = subprocess.Popen(command, cwd=folder)
#         process.communicate()
#         if process.returncode == 0:
#             print(f"setFields completed in folder: {folder}")
#             logging.info(f"setFields completed in folder: {folder}")
#         else:
#             print(f"setFields failed in folder: {folder}")
#             logging.warning(f"setFields failed in folder: {folder}")
#     except FileNotFoundError:
#         print("Error: setFields command not found.")
#         logging.error("Error: setFields command not found.")
#     except Exception as e:
#         print(f"Error running setFields: {e}")
#         logging.error(f"Error running setFields: {e}")

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

def parse_internal_field(file_path, field_type="vector", expected_n_points=16128, default_value=None):
    """
    Parses the internal field data from an OpenFOAM field file.
    
    If field_type == "vector", it will capture the first two velocity components (Ux, Uy).
    
    This version is robust when "internalField" and "nonuniform" appear on separate lines.
    
    Args:
        file_path (str): Path to the OpenFOAM field file.
        field_type (str): Either "vector" (e.g., U) or "scalar" (e.g., p, T).
        expected_n_points (int, optional): Number of cells expected in the field.
            Required if you want to substitute a default.
        default_value (float, optional): Value to use if no data is found.
    
    Returns:
        np.ndarray:
            - For "vector": shape = (n_points, 2) with Ux and Uy.
            - For "scalar": shape = (n_points,).
    
    Raises:
        FileNotFoundError: If the file is not found.
        ValueError: If the file format is unexpected and no default is provided.
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
                # First check for a nonuniform field.
                if "nonuniform" in stripped:
                    inside_data_block = True
                    continue
                # Now check for a uniform field.
                elif "uniform" in stripped:
                    value_start = stripped.find("(")
                    value_end = stripped.find(")")
                    if value_start != -1 and value_end != -1:
                        uniform_value = stripped[value_start+1:value_end].strip()
                        if expected_n_points is None:
                            raise ValueError("expected_n_points must be provided for a uniform field.")
                        if field_type == "vector":
                            comps = uniform_value.strip("()").split()
                            if len(comps) < 2:
                                raise ValueError("Uniform vector field does not have at least two components.")
                            try:
                                val1 = float(comps[0])
                                val2 = float(comps[1])
                            except ValueError:
                                raise ValueError("Cannot parse uniform vector field values.")
                            return np.array([[val1, val2]] * expected_n_points, dtype=np.float32)
                        elif field_type == "scalar":
                            try:
                                val = float(uniform_value)
                            except ValueError:
                                raise ValueError("Cannot parse uniform scalar field value.")
                            return np.array([val] * expected_n_points, dtype=np.float32)
                    continue
            
            # If we've seen "internalField" but haven't started the data block yet,
            # check if this line indicates a nonuniform block.
            if found_internalField and not inside_data_block:
                if "nonuniform" in stripped:
                    inside_data_block = True
                    continue
            
            # Once inside the data block, process the lines.
            if inside_data_block:
                # If we haven't set n_points yet, check if this line is the number of points.
                s = stripped.rstrip(";")  # Remove any trailing semicolon
                if n_points is None and s.isdigit():
                    n_points = int(s)
                    continue
                
                # Skip the opening parenthesis.
                if stripped == "(":
                    continue
                
                # Check for a closing parenthesis.
                if stripped in [")", ");"] or stripped.startswith(")"):
                    inside_data_block = False
                    # We can break early if we have already collected the expected number.
                    if n_points is not None and len(data_lines) >= n_points:
                        break
                    continue
                
                # Append data lines (which should be values like "(0.169834 -0.026175 0)")
                data_lines.append(stripped)
                
                # If we've collected the expected number of data lines, break out.
                if n_points is not None and len(data_lines) >= n_points:
                    break

    # If no data was collected and no default is provided, raise an error.
    if (n_points is None or len(data_lines) == 0) and default_value is None:
        raise ValueError(f"No internalField data found in {file_path}")
    
    # If a default is provided, use it.
    if (n_points is None or len(data_lines) == 0) and default_value is not None:
        if expected_n_points is None:
            raise ValueError("expected_n_points must be provided when using default_value.")
        if field_type == "vector":
            return np.array([[default_value, default_value]] * expected_n_points, dtype=np.float32)
        elif field_type == "scalar":
            return np.array([default_value] * expected_n_points, dtype=np.float32)
    
    # If n_points was not explicitly set, assume it equals the number of lines collected.
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
            # Remove enclosing parentheses.
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

def parse_simulation(sim_folder, expected_n_points=16128):
    """
    Parses a single simulation folder containing time-step directories 
    (e.g. "0", "0.1", "0.2", ...). **Time step "0" is skipped.**
    
    Returns:
        time_steps: Sorted list of time-step strings (excluding "0").
        results_array: np.ndarray of shape (num_time_steps, n_points, 4) 
                       with channels [rho, Ux, Uy, p].
    """
    # Collect numeric time directories (skip "constant", "system", etc.)
    all_entries = os.listdir(sim_folder)
    time_dirs = []
    for entry in all_entries:
        try:
            float(entry)  # Check if numeric
            if os.path.isdir(os.path.join(sim_folder, entry)) and entry != "0":
                time_dirs.append(entry)
        except ValueError:
            pass

    # Sort directories numerically
    time_dirs.sort(key=lambda x: float(x))
    
    results_per_time = []
    for tdir in time_dirs:
        tdir_path = os.path.join(sim_folder, tdir)
        
        # Construct file paths.
        T_file = os.path.join(tdir_path, "T")
        u_file = os.path.join(tdir_path, "U")
        p_file = os.path.join(tdir_path, "p")
        
        # Parse each field (no default values provided so errors are raised if data is missing).
        T_data = parse_internal_field(T_file, field_type="scalar", 
                                      expected_n_points=expected_n_points, default_value=None)
        u_data = parse_internal_field(u_file, field_type="vector", 
                                      expected_n_points=expected_n_points, default_value=None)
        p_data = parse_internal_field(p_file, field_type="scalar", 
                                      expected_n_points=expected_n_points, default_value=None)
        
        # Compute density (rho = p / (T * R) with R=287).
        rho_data = p_data / (T_data * 287)
        
        # Combine channels into one array per time step.
        # Each point: [rho, Ux, Uy, p]
        combined_data = np.column_stack([rho_data, u_data[:, 0], u_data[:, 1], p_data])
        results_per_time.append(combined_data)
    
    results_array = np.stack(results_per_time, axis=0)  # shape: (num_time_steps, n_points, 4)
    return time_dirs, results_array


def gather_all_simulations(sim_folders, expected_n_points=16128):
    """
    Given a list of simulation folders, parse them all and stack results into a single array.
    
    Returns:
        final_time_dirs (list of str): The time steps used (assuming all sims have the same times).
        final_results (np.ndarray): shape (num_sims, num_time_steps, n_points, 4).
    """
    all_sim_results = []
    final_time_dirs = None
    
    for i, folder in enumerate(sim_folders):
        logging.info(f"Parsing simulation folder {folder}")
        tdirs, results_array = parse_simulation(folder, expected_n_points)
        if i == 0:
            # Keep track of the "canonical" time steps
            final_time_dirs = tdirs
        else:
            # (Optional) check that tdirs == final_time_dirs if you assume they must match
            if tdirs != final_time_dirs:
                logging.warning(
                    f"Simulation {folder} has different time steps than the first simulation. "
                    f"This code assumes consistent time steps across simulations."
                )
        all_sim_results.append(results_array)
    
    # Stack them: from list of [ (num_time_steps, n_points, 4), ... ] 
    # into (num_sims, num_time_steps, n_points, 4)
    final_results = np.stack(all_sim_results, axis=0)
    # => shape (num_sims, num_time_steps, n_points, 4)
    
    return final_time_dirs, final_results

def generate_blockMeshDict(i_c, j_c, output_path="blockMeshDict", run_blockMesh=False):
    """
    Generate a blockMeshDict for a domain [0,1] x [0,1] (z from 0 to 0.1)
    with a square hole of size 16x16 cells, centered on the grid node (i_c, j_c).
    
    Requirements:
    -------------
    1) i_c, j_c must be integers with 8 <= i_c <= 120 to maintain an 8-cell margin.
    2) The total resolution is 128 x 128 in x,y, so the cell size is 1/128.
    3) The hole is always 16 cells wide and 16 cells tall (16/128 = 0.125).
    4) The "hole" is the middle block [x1, x2] x [y1, y2], which is omitted.

    The resulting dictionary has the same topology (same vertex indexing,
    same block -> vertex connectivity) as the reference layout with blocks
    around the missing center block.
    """
    # Check input: update margins from 4 cells to 8 cells.
    if not (12 <= i_c <= 120):
        raise ValueError("i_c must be in [8..120] to keep an 8-cell margin.")
    if not (12 <= j_c <= 120):
        raise ValueError("j_c must be in [8..120] to keep an 8-cell margin.")

    # Convert center indices to physical coordinates:
    # The hole now extends 8 cells to each side of the center.
    x1 = ((i_c - 8) / 128) * 64
    x2 = ((i_c + 8) / 128) * 64
    y1 = ((j_c - 8) / 128) * 64
    y2 = ((j_c + 8) / 128) * 64
    
    # Number of cells in x and y directions for each sub-block
    Nx_left   = i_c - 8          # from 0 to x1
    Nx_hole   = 16               # hole spans 16 cells (from x1 to x2)
    Nx_right  = 128 - (i_c + 8)   # from x2 to 64

    Ny_bottom = j_c - 8
    Ny_hole   = 16               # hole spans 16 cells in y-direction
    Ny_top    = 128 - (j_c + 8)

    # z-plane thickness remains unchanged
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
    # 2. Define the 64 vertices
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
        i_c = np.random.randint(12, N - 8)  # range(4, N-3) => up to N-4 inclusive
        j_c = np.random.randint(12, N - 8)

        # Convert these integer centers to continuous coordinates
        cx = (i_c / N) * 64
        cy = (j_c / N) * 64

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

def validate_missing_coordinates(coords, hole_size=0.250):
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

def main():
    total_trajectories = int(input("Enter the total number of trajectories to simulate: "))
    batch_size = 128  # Adjust based on memory availability
    main_folder = "Design_Point_0"
    start_time = time.time()
    batch_index = 1

    # Generate hole positions once for all trajectories
    N = 128
    hole_centers = random_hole_centers(total_trajectories, N)

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
            i_c, j_c = batch_hole_centers[i]
            logging.info(f"Processing folder {folder} with hole center ({i_c:.4f}, {j_c:.4f})")

            # Generate mesh
            bmd_path = os.path.join(folder, "system", "blockMeshDict")
            try:
                generate_blockMeshDict(i_c=i_c, j_c=j_c, output_path=bmd_path, run_blockMesh=True)
            except subprocess.CalledProcessError as e:
                logging.error(f"BlockMesh failed for {folder}: {e}")
                continue  # Skip this folder

            # Run solver
            update_Umax_in_simulation_folder(folder, min_value=1.125e-3, max_value=1.125e-1)
            if not run_rhoPimpleFoam(folder):
                logging.warning(f"Solver failed for folder {folder}. Skipping.")
                continue

            # Generate and validate cell centers
            c_file = generate_cell_centers(folder)
            if not c_file:
                logging.error(f"Cell centers file not found for {folder}. Skipping.")
                continue

            coords = parse_cell_centers(c_file)
            missing_x, missing_y = validate_missing_coordinates(coords, hole_size=0.125)
            if len(missing_x) != 16 or len(missing_y) != 16:
                logging.warning(f"Validation failed for {folder}. Skipping.")
                continue

            # Passed all checks, so we record this folder in converged_folders
            converged_folders.append(folder)

            logging.info(f"Validation passed for folder {folder}")

            # Store simulation metadata
            with open(c_file, "r") as f:
                c_contents = f.read()

            sim_data.append({"folder": folder, "c_contents": c_contents})

        # Step 3: Save simulation metadata
        if sim_data:
            sim_data_path = f"sim_data_batch_{batch_index}.json"
            with open(sim_data_path, "w") as jf:
                json.dump(sim_data, jf)
            logging.info(f"Sim data saved to {sim_data_path}")
        else:
            logging.warning("No valid simulations in this batch.")

        # Step 4: Run `gather_all_simulations` for this batch
        if converged_folders:
            try:
                time_dirs, batch_results = gather_all_simulations(converged_folders)
                if batch_results is not None:
                    batch_file = f"results_batch_{batch_index}.npy"
                    np.save(batch_file, batch_results)
                    logging.info(f"Batch {batch_index} dataset shape: {batch_results.shape} saved as {batch_file}")

                    # Append time directories
                    with open("time_dirs.txt", "a") as f:
                        for t in time_dirs:
                            f.write(str(t) + "\n")

                    batch_index += 1
                else:
                    logging.error("No simulations successfully processed in this batch. Skipping.")
            except FileNotFoundError as e:
                logging.error(f"Error processing batch: {e}")

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
    total_time_elapsed = time.time() - start_time
    logging.info(f"Total execution time: {total_time_elapsed:.2f} seconds")
    print(f"\nTotal execution time: {total_time_elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
