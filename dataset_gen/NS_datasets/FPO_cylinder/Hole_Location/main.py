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

def run_icoFoam(folder):
    """Runs icoFoam for the specified folder and returns True if it converged, False otherwise."""
    command = ["icoFoam"]
    print(f"Running icoFoam in folder: {folder}")
    try:
        process = subprocess.Popen(command, cwd=folder)
        process.communicate()
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

def parse_simulation(sim_folder, expected_n_points=16128, Umax_simulation=None, L=2, nu=1.53e-5):
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

    if Umax_simulation is None:
        raise ValueError("Umax_simulation must be provided to compute the Reynolds number.")
    Re_sim = Umax_simulation * L / nu
    
    results_per_time = []
    for tdir in time_dirs:
        tdir_path = os.path.join(sim_folder, tdir)
        
        # Construct file paths.
        u_file = os.path.join(tdir_path, "U")
        p_file = os.path.join(tdir_path, "p")
        
        # Parse each field (no default values provided so errors are raised if data is missing).
        u_data = parse_internal_field(u_file, field_type="vector", 
                                      expected_n_points=expected_n_points, default_value=None)
        p_data = parse_internal_field(p_file, field_type="scalar", 
                                      expected_n_points=expected_n_points, default_value=None)
        
        # Combine channels into one array per time step.
        # Each point: [rho, Ux, Uy, p]
        combined_data = np.column_stack([u_data[:, 0], u_data[:, 1], p_data])
        # Create a Reynolds number channel (constant across all cells in this time step).
        Re_channel = np.full((combined_data.shape[0], 1), Re_sim)
        
        # Append the Reynolds number channel.
        combined_data_with_Re = np.column_stack([combined_data, Re_channel])
        results_per_time.append(combined_data_with_Re)
    
    results_array = np.stack(results_per_time, axis=0)  # shape: (num_time_steps, n_points, 4)
    return time_dirs, results_array


def gather_all_simulations(sim_folders, expected_n_points=16128, L=2, nu=1.53e-5):
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
        # Read Umax for the simulation from the file.
        Umax_simulation = get_Umax_from_sim_folder(folder)
        logging.info(f"Umax for simulation {folder} is {Umax_simulation}")
        tdirs, results_array = parse_simulation(folder, expected_n_points, Umax_simulation, L, nu)
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
    # => shape (num_sims, num_time_steps, n_points, 5)
    
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
    x1 = ((i_c - 8) / 128) * 2
    x2 = ((i_c + 8) / 128) * 2
    y1 = ((j_c - 8) / 128) * 2
    y2 = ((j_c + 8) / 128) * 2
    
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
        (2   , 0   , 0),
        (0   , y1  , 0),
        (x1  , y1  , 0),
        (x2  , y1  , 0),
        (2   , y1  , 0),
        (0   , y2  , 0),
        (x1  , y2  , 0),
        (x2  , y2  , 0),
        (2   , y2  , 0),
        (0   , 2   , 0),
        (x1  , 2   , 0),
        (x2  , 2   , 0),
        (2   , 2   , 0),
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
        cx = (i_c / N) * 2
        cy = (j_c / N) * 2

        # centers.append((cx, cy))
        centers.append((i_c, j_c))
    logging.info(f"The center of the hole is:{cx}")
    logging.info(f"The center of the hole is:{cy}")
    logging.info(f"The initial calculated x_min is:{cx - 0.125}")
    logging.info(f"The initial calculated x_max is:{cx + 0.125}")
    logging.info(f"The initial calculated y_min is:{cy - 0.125}")
    logging.info(f"The initial calculated y_max is:{cy + 0.125}")
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

    coords = np.round(coords, decimals=2)

    # Identify unique x and y coordinates
    x_unique, x_counts = np.unique(coords[:, 0], return_counts=True)
    y_unique, y_counts = np.unique(coords[:, 1], return_counts=True)

    # Identify missing coordinates
    expected_count = len(y_unique)  # Expected count for a fully populated mesh
    missing_x = x_unique[x_counts < expected_count]
    missing_y = y_unique[y_counts < expected_count]

    return missing_x, missing_y

def compute_parabolic_inlet(Umax, H=2, num_points=128):
    """
    Compute the parabolic velocity profile for the inlet.
    
    Parameters:
    - Umax: Maximum velocity at the centerline.
    - H: Channel height (64m).
    - num_points: Number of discrete points in the y-direction.

    Returns:
    - List of velocity vectors as strings.
    """
    y_values = np.linspace(0, H, num_points)  # Discretize y positions
    velocities = []

    for y in y_values:
        u = 4.0 * Umax * y * (H - y) / (H * H)  # Parabolic profile
        velocities.append(f"({u} 0 0)")  # Format as OpenFOAM vector

    return velocities

def update_U_file(sim_folder, Umax):
    """
    Updates the U file to use a nonuniform fixedValue inlet condition.

    Parameters:
    - sim_folder: Path to the simulation folder.
    - Umax: Computed Umax based on the given Reynolds number.
    """
    u_file_path = os.path.join(sim_folder, "0", "U")

    if not os.path.exists(u_file_path):
        raise FileNotFoundError(f"U file not found in {u_file_path}")

    # Compute parabolic inlet velocities
    velocity_values = compute_parabolic_inlet(Umax)

    boundaryField = (
    f"    left\n"
    f"    {{\n"
    f"        type            fixedValue;\n"
    f"        value           nonuniform List<vector>\n"
    f"        {len(velocity_values)}\n"
    f"        (\n"
    + '\n'.join(velocity_values) + "\n"
    f"        );\n"
    f"    }}"
    )


    # Read the existing file and replace only the left boundary definition
    with open(u_file_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    inside_left_boundary = False

    for line in lines:
        if "left" in line:
            inside_left_boundary = True
            new_lines.append(boundaryField)  # Replace with new boundary definition
        elif inside_left_boundary and "}" in line:
            inside_left_boundary = False  # Stop skipping
        elif not inside_left_boundary:
            new_lines.append(line)

    # Write the updated file
    with open(u_file_path, "w") as f:
        f.writelines(new_lines)

    logging.info(f"Updated {u_file_path} with nonuniform parabolic inlet velocities.")


def generate_normal_re_values(num_samples, mean=5000, std_dev=2000, min_re=100, max_re=10000):
    """
    Generates a normally distributed set of Reynolds numbers.

    Parameters:
    - num_samples: Total number of trajectories (Reynolds numbers) to generate.
    - mean: Mean of the normal distribution.
    - std_dev: Standard deviation.
    - min_re, max_re: Limits to ensure Re values remain within physical bounds.

    Returns:
    - np.array of normally distributed Reynolds numbers.
    """
    re_values = np.random.normal(loc=mean, scale=std_dev, size=num_samples)
    re_values = np.clip(re_values, min_re, max_re)  # Ensure values are within range
    return re_values

def update_Umax_in_simulation_folder(sim_folder, re_value, L=2, nu=1.53e-5):
    """
    Updates the Umax value in the simulation folder using a precomputed Reynolds number.
    
    Parameters:
    - sim_folder: Path to the simulation folder.
    - re_value: Precomputed Reynolds number for this simulation.
    - L: Characteristic length.
    - nu: Kinematic viscosity.
    """
    u_file_path = os.path.join(sim_folder, "0", "U")
    if not os.path.exists(u_file_path):
        raise FileNotFoundError(f"U file not found at {u_file_path}")

    # Compute Umax from the provided Reynolds number
    Umax_simulation = (re_value * nu) / L

    # Update the U file with the new Umax value
    update_U_file(sim_folder, Umax_simulation)

    
    # Save Umax to a file
    Umax_txt_path = os.path.join(sim_folder, "Umax.txt")
    with open(Umax_txt_path, "w") as f:
        f.write(str(Umax_simulation))
    
    logging.info(f"Updated {u_file_path}: Umax set to {Umax_simulation} (Re: {re_value})")
    logging.info(f"Saved Umax value {Umax_simulation} in {Umax_txt_path}")

def get_Umax_from_sim_folder(sim_folder):
    """
    Reads the Umax value for the simulation from a file.
    This assumes that each simulation folder contains a file 'Umax.txt'
    with a single number representing the Umax for that simulation.
    """
    Umax_file = os.path.join(sim_folder, "Umax.txt")
    if not os.path.exists(Umax_file):
        raise FileNotFoundError(f"Umax.txt not found in {sim_folder}")
    with open(Umax_file, "r") as f:
        Umax_simulation = float(f.read().strip())
    return Umax_simulation

def main():
    save_dir = "/data/user_data/namancho/FPO_cylinder_hole_location"
    os.makedirs(save_dir, exist_ok=True)
    total_trajectories = int(input("Enter the total number of trajectories to simulate: "))
    batch_size = 128  # Adjust based on memory availability
    main_folder = "Design_Point_0"
    start_time = time.time()
    batch_index = 1

    # Generate all Re values upfront
    re_values = generate_normal_re_values(total_trajectories, mean=5000, std_dev=2000)

    trajectory_idx = 0

    # Generate hole positions once for all trajectories
    N = 128 #grid size
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
            update_Umax_in_simulation_folder(folder, re_values[trajectory_idx])
            trajectory_idx += 1
            if not run_icoFoam(folder):
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
            sim_data_path = os.path.join(save_dir, f"2sim_data_batch_{batch_index}.json")
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
                    batch_file = os.path.join(save_dir, f"2results_batch_{batch_index}.npy")
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
