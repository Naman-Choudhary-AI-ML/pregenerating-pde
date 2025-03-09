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

def generate_random_coefficients():
    """
    Generate random coefficients for perturbations.
    
    Args:
        p (int): Number of modes for perturbations.
        
    Returns:
        dict: A dictionary containing random coefficients for alpha0, beta0, alpha1, and beta1.
    """
    quadrants = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        quadrants[q] = {
            "rho": np.random.uniform(0.1, 1.0),
            "vx":  np.random.uniform(-1.0, 1.0),
            "vy":  np.random.uniform(-1.0, 1.0),
            "p":   np.random.uniform(0.1, 1.0),
        }
    return quadrants

def generate_U_file(folder, coefficients):

    # Sample for each quadrant
    quadrants = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        rho = coefficients[q]['rho']
        vx = coefficients[q]['vx']
        vy = coefficients[q]['vy']
        p = coefficients[q]['p']
        
        # Assume ideal gas with R=1 for simplicity (dimensionless)
        R = 287
        T = p/(rho*R)
        
        quadrants[q] = {
            'rho': rho,
            'vx': vx,
            'vy': vy,
            'p' : p,
            'T' : T
        }

    # Now create the setFieldsDict
    setFieldsDict_content = """FoamFile
    {
        version     2.0;
        format      ascii;
        class       dictionary;
        object      setFieldsDict;
    }
    defaultFieldValues
    (
        volVectorFieldValue U (0 0 0)
        volScalarFieldValue p 1
        volScalarFieldValue T 1
    );

    regions
    (
    """
    # Define each quadrant region and its field values
    # Quadrant 1: [0,0.5)x[0,0.5)
    setFieldsDict_content += f"""
        boxToCell
        {{
            type    boxToCell;
            box     (0 0 0) (0.5 0.5 0.5);
            fieldValues
            (
                volVectorFieldValue U ({quadrants['Q1']['vx']} {quadrants['Q1']['vy']} 0)
                volScalarFieldValue p {quadrants['Q1']['p']}
                volScalarFieldValue T {quadrants['Q1']['T']}
            );
        }}
    """

    # Quadrant 2: [0.5,1.0)x[0,0.5)
    setFieldsDict_content += f"""
        boxToCell
        {{
            type    boxToCell;
            box     (0.5 0 0) (1.0 0.5 0.5);
            fieldValues
            (
                volVectorFieldValue U ({quadrants['Q2']['vx']} {quadrants['Q2']['vy']} 0)
                volScalarFieldValue p {quadrants['Q2']['p']}
                volScalarFieldValue T {quadrants['Q2']['T']}
            );
        }}
    """

    # Quadrant 3: [0,0.5)x[0.5,1.0)
    setFieldsDict_content += f"""
        boxToCell
        {{
            type    boxToCell;
            box     (0 0.5 0) (0.5 1.0 0.5);
            fieldValues
            (
                volVectorFieldValue U ({quadrants['Q3']['vx']} {quadrants['Q3']['vy']} 0)
                volScalarFieldValue p {quadrants['Q3']['p']}
                volScalarFieldValue T {quadrants['Q3']['T']}
            );
        }}
    """

    # Quadrant 4: [0.5,1.0)x[0.5,1.0)
    setFieldsDict_content += f"""
        boxToCell
        {{
            type    boxToCell;
            box     (0.5 0.5 0) (1.0 1.0 0.5);
            fieldValues
            (
                volVectorFieldValue U ({quadrants['Q4']['vx']} {quadrants['Q4']['vy']} 0)
                volScalarFieldValue p {quadrants['Q4']['p']}
                volScalarFieldValue T {quadrants['Q4']['T']}
            );
        }}
    );

    """

    # Write to system/setFieldsDict in the specified design point folder
    system_path = os.path.join(folder, "system")
    os.makedirs(system_path, exist_ok=True)
    setFieldsDict_path = os.path.join(system_path, "setFieldsDict")
    with open(setFieldsDict_path, "w") as f:
        f.write(setFieldsDict_content)

    print("Random initial conditions generated and setFieldsDict created.")
    for q, vals in quadrants.items():
        print(f"{q}: rho={vals['rho']}, p={vals['p']}, U=({vals['vx']}, {vals['vy']}) -> T={vals['T']}")


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

def parse_simulation(sim_folder):
    """
    Parses a single simulation folder containing time-step directories 
    (e.g. 0, 0.1, 0.2, ...).
    
    Returns:
        time_steps: Sorted list of time-step strings (e.g. ["0", "0.1", "0.2", ...])
        results_array: np.ndarray of shape (num_time_steps, num_points, 4) 
                       for channels [rho, Ux, Uy, p].
    """
    # Collect all numeric time directories (excluding things like "constant", "system", etc.)
    all_entries = os.listdir(sim_folder)
    time_dirs = []
    for entry in all_entries:
        # Test if `entry` is something like "0", "0.1", "10", etc.
        try:
            float(entry)  # If this works, it is likely a time directory
            if os.path.isdir(os.path.join(sim_folder, entry)):
                time_dirs.append(entry)
        except ValueError:
            # Not a numeric directory
            pass
    
    # Sort them by numerical value
    time_dirs.sort(key=lambda x: float(x))

    # We expect to parse for each time step:
    #   - rho  (scalar)
    #   - U    (vector)
    #   - p    (scalar)
    # You also mentioned T; if you want T, add it in similarly.
    # But your final request was channels: [rho, Ux, Uy, P], so let's do that.

    results_per_time = []
    for tdir in time_dirs:
        this_tdir_path = os.path.join(sim_folder, tdir)
        
        # Prepare file paths
        T_file = os.path.join(this_tdir_path, "T")
        u_file   = os.path.join(this_tdir_path, "U")
        p_file   = os.path.join(this_tdir_path, "p")

        # Parse each field
        T_data = parse_internal_field(T_file, field_type="scalar")  # shape=(n_points,)
        u_data   = parse_internal_field(u_file,  field_type="vector")   # shape=(n_points,2 or 3)
        p_data   = parse_internal_field(p_file,  field_type="scalar")   # shape=(n_points,)

        try:
            rho_data = p_data / (T_data * 287)
        except ZeroDivisionError:
            raise ValueError("Encountered division by zero")

        # If you have 3D velocity (Ux, Uy, Uz), you’d need to slice out just Ux, Uy 
        # or keep all three. For your example, let’s assume 2D: shape=(n_points, 2).

        # Reorganize into [rho, Ux, Uy, p] per point
        # So shape will be: (n_points, 4)
        combined_data = np.column_stack([rho_data, 
                                         u_data[:, 0],  # Ux
                                         u_data[:, 1],  # Uy
                                         p_data])
        results_per_time.append(combined_data)

    # Now we have a list of length num_time_steps, 
    # each element shape (n_points, 4)
    results_array = np.stack(results_per_time, axis=0)  
    # => shape (num_time_steps, n_points, 4)

    return time_dirs, results_array


def gather_all_simulations(sim_folders):
    """
    Given a list of simulation folders, parse them all and stack results into a single array.
    
    Returns:
        final_time_dirs (list of str): The time steps used (assuming all sims have same times).
        final_results (np.ndarray): shape (num_sims, num_time_steps, n_points, 4).
    """
    all_sim_results = []
    final_time_dirs = None
    
    for i, folder in enumerate(sim_folders):
        logging.info(f"Parsing simulation folder {folder}")
        tdirs, results_array = parse_simulation(folder)
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

def compute_hole_coords(cx, cy, hole_size=0.0625, z_min=0.0, z_max=0.1):
    """
    Given a hole center (cx, cy), returns the dict of new vertex coords
    for the 'square hole' corners, preserving the hole size.
    
    hole_size: the side length of the hole
    z_min, z_max: the z-limits in your domain
    """
    half = hole_size / 2.0
    
    x_min = cx - half
    x_max = cx + half
    y_min = cy - half
    y_max = cy + half
    
    # Sanity check: x_max - x_min must be hole_size
    if not np.isclose(x_max - x_min, hole_size, atol=1e-12):
        raise ValueError(f"x-dimension mismatch: {(x_max - x_min)} != {hole_size}")
    if not np.isclose(y_max - y_min, hole_size, atol=1e-12):
        raise ValueError(f"y-dimension mismatch: {(y_max - y_min)} != {hole_size}")

    logging.info(f"Hole corners: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}. "
                 f"Expected hole size = {hole_size}")

    new_coords = {
        1: (x_min, 0, z_min),
        2: (x_max, 0, z_min),
        4: (0, y_min, z_min),
        5:  (x_min, y_min, z_min),
        6:  (x_max, y_min, z_min),
        7: (1.0, y_min, z_min),
        8: (0, y_max, z_min),
        9:  (x_min, y_max, z_min),
        10: (x_max, y_max, z_min),
        11: (1.0, y_max, z_min),
        13: (x_min, 1.0, z_min),
        14: (x_max, 1.0, z_min),
        17: (x_min, 0, z_max),
        18: (x_max, 0, z_max),
        20: (0, y_min, z_max),
        21: (x_min, y_min, z_max),
        22: (x_max, y_min, z_max),
        23: (1.0, y_min, z_max),
        24: (0.0, y_max, z_max),
        25: (x_min, y_max, z_max),
        26: (x_max, y_max, z_max),
        27: (1.0, y_max, z_max),
        29: (x_min, 1.0, z_max),
        30: (x_max, 1.0, z_max)
    }
    return new_coords

def parse_and_modify_blockMeshDict(
    blockMeshDict_path,
    output_path=None,
    new_vertex_coords=None,
    run_blockMesh=False,
):
    """
    Parses the given blockMeshDict, updates specific vertices, and (optionally) 
    runs blockMesh after modification.

    :param blockMeshDict_path: Path to the existing blockMeshDict file.
    :param output_path: Path to write the modified file. If None, it overwrites the original.
    :param new_vertex_coords: Dictionary mapping vertex indices to new coordinates, 
                              e.g. {5: (0.45, 0.45, 0.0), 6: (0.55, 0.45, 0.0), ...}
    :param run_blockMesh: If True, run 'blockMesh' from the folder one level above blockMeshDict.
    """
    if output_path is None:
        output_path = blockMeshDict_path  # overwrite the original

    # Read all lines
    with open(blockMeshDict_path, 'r') as f:
        lines = f.readlines()
    
    # We'll store the new lines here
    new_lines = []
    
    in_vertices_block = False
    vertex_pattern = re.compile(r'^\s*\(([^)]+)\)\s*//\s*v(\d+)\s*$')
    # Explanation of the regex:
    #  ^\s*\(([^)]+)\)\s*//\s*v(\d+)\s*$
    #  - ^\s*    => start of line, ignoring leading spaces
    #  - \(      => literal '('
    #  - ([^)]+) => capture all text until the next ')'
    #  - \)      => literal ')'
    #  - \s*//\s* v(\d+) => the comment which indicates vertex index, e.g. "// v5"
    #  - \s*$    => trailing spaces until end of line

    for line in lines:
        # Check if we are in the "vertices" section
        if line.strip().startswith("vertices"):
            in_vertices_block = True
            new_lines.append(line)
            continue
        
        # The vertices block ends when we see ");" on a line by itself (or with whitespace).
        if in_vertices_block and line.strip().startswith(");"):
            in_vertices_block = False
            new_lines.append(line)
            continue

        if in_vertices_block:
            # Attempt to parse a vertex line
            match = vertex_pattern.match(line.strip())
            if match:
                # We got a line like "(0.46875   0.46875   0) // v5"
                coords_str, v_index_str = match.groups()
                v_index = int(v_index_str)
                
                # Original coords as strings
                coords_list_str = coords_str.split()
                # Convert to floats
                x_orig, y_orig, z_orig = map(float, coords_list_str)
                
                # If user provided new coords for this vertex, update
                if new_vertex_coords and v_index in new_vertex_coords:
                    x_new, y_new, z_new = new_vertex_coords[v_index]
                else:
                    x_new, y_new, z_new = x_orig, y_orig, z_orig
                
                # Rebuild the line
                new_line = f"    ({x_new} {y_new} {z_new}) // v{v_index}\n"
                new_lines.append(new_line)
            else:
                # Keep the line as-is, in case there's formatting or comment lines
                new_lines.append(line)
        else:
            # Outside vertices block, just copy line
            new_lines.append(line)
    
    # Write out the modified file
    with open(output_path, 'w') as f:
        f.writelines(new_lines)
    
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
    
    # Write out the modified file
    with open(output_path, 'w') as f:
        f.writelines(new_lines)

    # Optional: re-check the file to confirm the new coords match what was intended
    if new_vertex_coords is not None:
        check_blockMesh_vertices(output_path, new_vertex_coords)

def check_blockMesh_vertices(blockMeshDict_path, expected_coords_map):
    """
    Re-reads the blockMeshDict and checks that for each vIndex in expected_coords_map,
    the coordinate stored in the file matches the expected (within some tolerance).
    """
    vertex_pattern = re.compile(r'^\s*\(([^)]+)\)\s*//\s*v(\d+)\s*$')
    
    with open(blockMeshDict_path, 'r') as f:
        lines = f.readlines()
    
    found_vertices = {}
    for line in lines:
        match = vertex_pattern.match(line.strip())
        if match:
            coords_str, v_index_str = match.groups()
            v_index = int(v_index_str)
            coords_list_str = coords_str.split()
            x, y, z = map(float, coords_list_str)
            found_vertices[v_index] = (x, y, z)
    
    # Now compare
    for v_idx, (x_exp, y_exp, z_exp) in expected_coords_map.items():
        x_act, y_act, z_act = found_vertices.get(v_idx, (None, None, None))
        if x_act is None:
            raise ValueError(f"Expected vertex {v_idx} not found in blockMeshDict!")
        
        # Use a small tolerance
        if (not np.isclose(x_act, x_exp, atol=1e-12) or
            not np.isclose(y_act, y_exp, atol=1e-12) or
            not np.isclose(z_act, z_exp, atol=1e-12)):
            raise ValueError(
                f"Mismatch for vertex {v_idx}. "
                f"Expected ({x_exp:.6f}, {y_exp:.6f}, {z_exp:.6f}), "
                f"Got      ({x_act:.6f}, {y_act:.6f}, {z_act:.6f})"
            )

    logging.info("All modified vertices in blockMeshDict match expected values.")

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
        i_c = np.random.randint(4, N - 3)
        j_c = np.random.randint(4, N - 3)

        # Convert to continuous coordinates
        cx = i_c / N
        cy = j_c / N
        
        # Sanity check (these might be debug logs or outright assertions):
        expected_cell_x = round(cx * N)
        if expected_cell_x != i_c:
            raise ValueError(f"Mismatch in cell-center integer conversion: got {expected_cell_x}, expected {i_c}.")

        centers.append((cx, cy))

        logging.info(f"The chosen integer center (i_c, j_c): ({i_c}, {j_c}) => Continuous (cx, cy): ({cx}, {cy})")

    return centers

def concatenate_c_files(c_files):
    """
    Concatenates all generated C files into a single file.
    Args:
        c_files (list): List of paths to the generated C files.
    Returns:
        str: Path to the concatenated C file.
    """
    output_file = "concatenated_C.txt"
    with open(output_file, "w") as outfile:
        for c_file in c_files:
            with open(c_file, "r") as infile:
                outfile.write(infile.read())
                outfile.write("\n")  # Add a newline between files for clarity
    return output_file

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

def validate_missing_coordinates(coords, cx, cy, hole_size=0.0625):
    """
    Identifies missing x and y coordinates based on the hole region.

    :param coords: NumPy array of cell center coordinates.
    :param cx: Hole center x-coordinate.
    :param cy: Hole center y-coordinate.
    :param hole_size: Size of the hole (side length).
    :return: Unique missing x and y coordinates.
    """
    half_size = hole_size / 2
    x_min, x_max = cx - half_size, cx + half_size
    y_min, y_max = cy - half_size, cy + half_size

    # Identify unique x and y coordinates
    x_unique, x_counts = np.unique(coords[:, 0], return_counts=True)
    y_unique, y_counts = np.unique(coords[:, 1], return_counts=True)

    # Identify missing coordinates
    expected_count = len(y_unique)  # Expected count for a fully populated mesh
    missing_x = x_unique[x_counts < expected_count]
    missing_y = y_unique[y_counts < expected_count]

    return missing_x, missing_y

def main():
    num_copies = int(input("Enter the number of folders to create: "))
    main_folder = "Design_Point_0"

    # Ask if parameters should be fixed across all hole locations
    fixed_params = input("Keep parameters fixed for all hole locations? (yes/no): ").strip().lower() in ["yes", "y"]

    # 1) Copy the main folder 'num_copies' times
    new_folders = copy_main_folder(main_folder, num_copies)

    # 2) Generate hole centers
    N = 128
    hole_centers = random_hole_centers(num_copies, N)
    if num_copies > len(hole_centers):
        raise ValueError("Not enough hole centers specified for the number of folders.")

    # If fixed_params = True, generate random coeffs only once
    random_coeffs = None
    if fixed_params:
        random_coeffs = generate_random_coefficients()

    # Lists to store final data
    converged_folders = []  # For folders that pass all checks
    sim_data = []           # For JSON serialization
    # We'll store final OpenFOAM results into a NumPy array later

    for i, folder in enumerate(new_folders, start=1):
        cx, cy = hole_centers[i - 1]
        logging.info(f"Processing folder {folder} with hole center ({cx:.4f}, {cy:.4f})")

        # Path to blockMeshDict
        bmd_path = os.path.join(folder, "system", "blockMeshDict")

        # Compute new hole coordinates
        hole_dict = compute_hole_coords(cx, cy, hole_size=0.0625)

        # 3A) Modify blockMeshDict and run blockMesh
        try:
            parse_and_modify_blockMeshDict(
                blockMeshDict_path=bmd_path,
                new_vertex_coords=hole_dict,
                run_blockMesh=True
            )
        except subprocess.CalledProcessError as e:
            logging.warning(f"BlockMesh failed for folder {folder}: {e}")
            continue  # Skip this folder entirely

        logging.info(f"Design point {i}: Moved hole center to ({cx}, {cy}). BlockMesh successful.")

        # 3B) Generate velocity field, setFields, etc.
        if not fixed_params:
            random_coeffs = generate_random_coefficients()
        generate_U_file(folder, random_coeffs)
        run_setfields(folder)

        # 3C) Run solver
        if not run_rhoPimpleFoam(folder):
            logging.warning(f"Solver failed for folder {folder}. Skipping.")
            continue

        # 3D) Generate and validate cell centers
        c_file = generate_cell_centers(folder)
        if not c_file:
            logging.error(f"Cell centers file not found for folder {folder}. Skipping.")
            continue

        coords = parse_cell_centers(c_file)
        missing_x, missing_y = validate_missing_coordinates(coords, cx, cy, hole_size=0.0625)
        if len(missing_x) != 8 or len(missing_y) != 8:
            logging.warning(
                f"Validation failed for folder {folder}: "
                f"missing_x={len(missing_x)}, missing_y={len(missing_y)}. Expected 8 each."
            )
            continue

        # Passed all checks, so we record this folder in converged_folders
        converged_folders.append(folder)

        logging.info(
            f"Validation passed for folder {folder}: "
            f"missing_x={len(missing_x)}, missing_y={len(missing_y)}."
        )

        # Optionally store the contents of C file (or other data) in sim_data
        with open(c_file, "r") as f:
            c_contents = f.read()

        sim_data.append({
            "folder": folder,
            "hole_center": (cx, cy),
            "c_contents": c_contents
        })

    # 4) Write out the simulation data (cell centers, hole info, etc.) to JSON
    if sim_data:
        sim_data_path = "sim_data_with_contents.json"
        with open(sim_data_path, "w") as jf:
            json.dump(sim_data, jf)
        logging.info(f"Sim data saved to {sim_data_path}")
    else:
        logging.warning("No valid simulations or cell center validations. No sim data generated.")

    # 5) Gather final results only from converged folders
    if converged_folders:
        try:
            time_dirs, final_results = gather_all_simulations(converged_folders)
        except FileNotFoundError as e:
            logging.error(f"Error processing converged folders: {e}")
            return

        # Save final results to .npy
        np.save("results.npy", final_results)
        logging.info(f"Saved final results with shape: {final_results.shape}")
        print(f"\nFinal results saved to results.npy with shape {final_results.shape}")
    else:
        logging.warning("No valid (converged) folders. Final results not generated.")
        print("\nNo valid simulations. No final results generated.")


if __name__ == "__main__":
    main()
