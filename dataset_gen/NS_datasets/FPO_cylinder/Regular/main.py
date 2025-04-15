import os
import shutil
import subprocess
import time
import numpy as np
import logging
from datetime import datetime, timedelta
import random
import re
import sys

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

def copy_main_folder(main_folder, num_copies, batch_name):
    """
    Copies the main folder content into num_copies of folders.
    Each new folder is namespaced by batch_name to avoid collisions.
    """
    new_folders = []
    for i in range(1, num_copies + 1):
        # Generate a unique folder name by prepending the batch name
        new_folder = f"{batch_name}_{main_folder}_copy_{i}"
        # If the folder exists from a previous run, delete it.
        if os.path.exists(new_folder):
            logging.info(f"Folder {new_folder} already exists. Removing it for a fresh copy.")
            shutil.rmtree(new_folder)
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

def update_seed_in_script(folder, new_seed):
    """Updates the seed in the generate_setfields.py script for a new trajectory."""
    script_path = os.path.join(folder, "generate_setfields.py")
    if os.path.exists(script_path):
        with open(script_path, 'r') as f:
            lines = f.readlines()

        updated_lines = []
        for line in lines:
            if "seed = " in line:
                updated_lines.append(f"seed = {new_seed}\n")
                logging.info(f"Updated seed to {new_seed} in {script_path}")
            else:
                updated_lines.append(line)

        with open(script_path, 'w') as f:
            f.writelines(updated_lines)
    else:
        print(f"generate_setfields.py not found in {folder}.")
        logging.error(f"generate_setfields.py not found in {folder}.")

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

def get_time_directories(folder):
    dirs = []
    for d in os.listdir(folder):
        if d.isdigit() or is_float(d):
            dirs.append(d)
    dirs = sorted(dirs, key=float)
    return dirs

def is_float(val):
    try:
        float(val)
        return True
    except ValueError:
        return False

def read_field_lines(field_path):
    with open(field_path, 'r') as f:
        lines = f.readlines()
    return lines

def parse_internal_field(lines, num_cells):
    """
    Parses the internalField data from an OpenFOAM file, handling both uniform
    and nonuniform cases.

    Parameters:
    - lines: List of strings representing lines of the file.
    - num_cells: Total number of cells (Nx * Ny) to expand uniform values.

    Returns:
    - data_lines: List of values (expanded for uniform fields).
    """
    data_lines = []
    inside_data_block = False
    N = None
    is_uniform = False

    for i, line in enumerate(lines):
        line = line.strip()

        # Detect uniform field
        if "internalField uniform" in line:
            print("Detected uniform field.")
            is_uniform = True
            # Extract the value inside parentheses
            value_start = line.find("(")
            value_end = line.find(")")
            uniform_value = line[value_start+1:value_end].strip()
            return [uniform_value] * num_cells  # Expand to match all cells

        # Detect nonuniform field
        if "internalField nonuniform" in line:
            print("Detected nonuniform field.")
            continue

        # If we haven't found a digit yet, try to find one
        if N is None and line.isdigit():
            N = int(line)  # Number of entries
            continue

        # Start of data block
        if line == "(":
            inside_data_block = True
            continue

        # End of data block
        if line == ")" and inside_data_block:
            inside_data_block = False
            break

        # Collect data lines if inside data block
        if inside_data_block:
            data_lines.append(line)

    # Handle missing entry count by fallback
    if N is None:
        N = len(data_lines)

    if len(data_lines) != N:
        raise ValueError(
            f"Expected {N} entries, but got {len(data_lines)}. "
            "Check if the file has unclosed parentheses or incomplete data."
        )

    return data_lines

def read_scalar_field(folder, time_dir, field_name, Nx=128, Ny=128):
    """
    Reads a scalar field (e.g., p, T) from an OpenFOAM file. If the field is missing
    or fails to parse, substitutes with default physical values.
    
    Parameters:
    - folder: Path to the simulation folder.
    - time_dir: Timestep directory (e.g., "0", "0.1").
    - field_name: Name of the field to read.
    - Nx, Ny: Dimensions of the grid.

    Returns:
    - A 2D NumPy array with the scalar field data.
    """
    num_cells = Nx * Ny
    field_path = os.path.join(folder, time_dir, field_name)

    # Define defaults
    default_values = {"p": 101325.0, "T": 300.0}  # Default pressure in Pa, temperature in K
    default_value = default_values.get(field_name, 0.0)  # Fallback to 0.0 if unknown field

    try:
        if not os.path.exists(field_path):
            print(f"Field '{field_name}' missing in timestep '{time_dir}', using default value {default_value}.")
            return np.full((Ny, Nx), default_value)  # Fill with default value if field is missing

        lines = read_field_lines(field_path)
        data_lines = parse_internal_field(lines, num_cells)
        data = np.array([float(val) for val in data_lines])
        return data.reshape((Ny, Nx))
    except Exception as e:
        print(f"Error reading scalar field '{field_name}' in timestep '{time_dir}': {e}")
        return np.full((Ny, Nx), default_value)  # Fill with default value if parsing fails

def read_vector_field(folder, time_dir, field_name, Nx=128, Ny=128):
    """
    Reads a vector field (e.g., U) from an OpenFOAM file. If the field is missing
    or fails to parse, substitutes with default zero velocity.

    Parameters:
    - folder: Path to the simulation folder.
    - time_dir: Timestep directory (e.g., "0", "0.1").
    - field_name: Name of the field to read.
    - Nx, Ny: Dimensions of the grid.

    Returns:
    - A 3D NumPy array with the vector field data (Ny, Nx, 2).
    """
    num_cells = Nx * Ny
    field_path = os.path.join(folder, time_dir, field_name)

    try:
        if not os.path.exists(field_path):
            print(f"Field '{field_name}' missing in timestep '{time_dir}', using default zero velocity.")
            return np.zeros((Ny, Nx, 2))  # Default zero velocity

        lines = read_field_lines(field_path)
        data_lines = parse_internal_field(lines, num_cells)
        U_data = []
        for val in data_lines:
            val = val.strip().strip("()")
            comps = val.split()
            Ux, Uy = float(comps[0]), float(comps[1])
            U_data.append([Ux, Uy])
        return np.array(U_data).reshape((Ny, Nx, 2))
    except Exception as e:
        print(f"Error reading vector field '{field_name}' in timestep '{time_dir}': {e}")
        return np.zeros((Ny, Nx, 2))  # Default zero velocity

def extract_results(folder, Nx=128, Ny=128, R=287):
    """
    Extracts results from simulation folders and processes fields.

    Parameters:
    - folder: Path to the simulation folder containing timestep directories.
    - Nx, Ny: Dimensions of the grid.
    - R: Gas constant for computing density.

    Returns:
    - A NumPy array with processed simulation data.
    """
    time_dirs = get_time_directories(folder)
    num_times = len(time_dirs)
    data_array = np.zeros((num_times, 3, Ny, Nx), dtype=np.float64)

    for i, tdir in enumerate(time_dirs):
        print(f"Processing timestep: {tdir}")

        U = read_vector_field(folder, tdir, "U", Nx, Ny)
        p = read_scalar_field(folder, tdir, "p", Nx, Ny)
        # T = read_scalar_field(folder, tdir, "T", Nx, Ny)

        # # Compute density with valid defaults
        # rho = p / (T * R)

        # # Store processed data
        # data_array[i, 0, :, :] = rho
        data_array[i, 0, :, :] = U[:, :, 0]  # Ux
        data_array[i, 1, :, :] = U[:, :, 1]  # Uy
        data_array[i, 2, :, :] = p

    return data_array

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

    # New boundaryField content
    # boundaryField = f"""
    # left
    # {{
    #     type            fixedValue;
    #     value           nonuniform List<vector>
    #     {len(velocity_values)}
    #     (
    #     {'\n'.join(velocity_values)}
    #     );
    # }}
    # """
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

def generate_cell_centers(folder):
    """Generates cell center coordinates using OpenFOAM post-processing utilities."""
    command = ["postProcess", "-func", "writeCellCentres"]
    print(f"Running postProcess to generate cell centers in folder: {folder}")
    logging.info(f"Running postProcess to generate cell centers in folder: {folder}")
    try:
        process = subprocess.Popen(command, cwd=folder)
        process.communicate()
        if process.returncode == 0:
            print(f"Cell center coordinates generated successfully in folder: {folder}")
            logging.info(f"Cell center coordinates generated successfully in folder: {folder}")
        else:
            print(f"Failed to generate cell center coordinates in folder: {folder}")
            logging.warning(f"Failed to generate cell center coordinates in folder: {folder}")
    except FileNotFoundError:
        print("Error: postProcess command not found.")
        logging.error("Error: postProcess command not found.")
    except Exception as e:
        print(f"Error running postProcess: {e}")
        logging.error(f"Error running postProcess: {e}")

def parse_internal_field(file_path, field_type="vector", expected_n_points=16384, default_value=None):
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
    import os
    import numpy as np

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

def parse_simulation(sim_folder, expected_n_points=16384, Umax_simulation=None, L=2, nu=1.53e-5):
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
        # T_file = os.path.join(tdir_path, "T")
        u_file = os.path.join(tdir_path, "U")
        p_file = os.path.join(tdir_path, "p")
        
        # Parse each field (no default values provided so errors are raised if data is missing).
        # T_data = parse_internal_field(T_file, field_type="scalar", 
        #                               expected_n_points=expected_n_points, default_value=None)
        u_data = parse_internal_field(u_file, field_type="vector", 
                                      expected_n_points=expected_n_points, default_value=None)
        p_data = parse_internal_field(p_file, field_type="scalar", 
                                      expected_n_points=expected_n_points, default_value=None)
        
        # # Compute density (rho = p / (T * R) with R=287).
        # rho_data = p_data / (T_data * 287)
        
        # Combine channels into one array per time step.
        # Each point: [rho, Ux, Uy, p]
        combined_data = np.column_stack([u_data[:, 0], u_data[:, 1], p_data])
        # Create a Reynolds number channel (constant across all cells in this time step).
        Re_channel = np.full((combined_data.shape[0], 1), Re_sim)
        
        # Append the Reynolds number channel.
        combined_data_with_Re = np.column_stack([combined_data, Re_channel])
        results_per_time.append(combined_data_with_Re)
        # results_per_time.append(combined_data)
    
    results_array = np.stack(results_per_time, axis=0)  # shape: (num_time_steps, n_points, 4)
    return time_dirs, results_array


def gather_all_simulations(sim_folders, expected_n_points=16384, L=2, nu=1.53e-5):
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

def main(batch_name: str, total_trajectories: int):
    save_dir = f"/data/user_data/namancho/FPO_cylinder_reg_new/{batch_name}"
    os.makedirs(save_dir, exist_ok=True)
    # total_trajectories = int(input("Enter the total number of trajectories to simulate: "))
    main_folder = "Design_Point_0"
    start_time = time.time()
    batch_size = 128  # Number of simulations per batch
    batch_index = 1

    # Generate all Re values upfront
    re_values = generate_normal_re_values(total_trajectories, mean=5000, std_dev=2000)

    trajectories_done = 0  # Track how many have been processed
    trajectory_idx = 0

    # --- Process in Batches ---
    while trajectories_done < total_trajectories:
        current_batch_size = min(batch_size, total_trajectories - trajectories_done)

        # Step 1: Create a batch of folders
        batch_folders = copy_main_folder(main_folder, current_batch_size, batch_name)
        converged_folders = []

        # Step 2: Run simulations for this batch
        for folder in batch_folders:
            folder_start_time = time.time()
            update_Umax_in_simulation_folder(folder, re_values[trajectory_idx])
            trajectory_idx += 1
            generate_cell_centers(folder)

            converged = run_icoFoam(folder)
            elapsed_time = time.time() - folder_start_time
            logging.info(f"Simulation for {folder} completed in {elapsed_time:.2f} seconds")

            if converged:
                converged_folders.append(folder)

        # Step 3: Save results for this batch
        if converged_folders:
            try:
                time_dirs, final_results = gather_all_simulations(converged_folders)
                if final_results is not None:
                    batch_file = os.path.join(save_dir, f"1results_batch_{batch_index}.npy")
                    np.save(batch_file, final_results)
                    logging.info(f"Batch {batch_index} dataset shape: {final_results.shape} saved as {batch_file}")

                    # Append time directories
                    with open("time_dirs.txt", "a") as f:
                        for t in time_dirs:
                            f.write(str(t) + "\n")

                    batch_index += 1
                else:
                    logging.error("No simulations successfully processed in this batch. Skipping.")
            except FileNotFoundError as e:
                logging.error(f"Error processing batch: {e}")

        # Step 4: Delete folders in the current batch to free space
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
    if len(sys.argv) < 3:
        print("Usage: python script.py <batch_name> <total_trajectories>")
        sys.exit(1)
    print(f"🧪 [TEST] Running with args: {sys.argv}")
    batch_name = sys.argv[1]
    total_trajectories = int(sys.argv[2])
    main(batch_name, total_trajectories)
