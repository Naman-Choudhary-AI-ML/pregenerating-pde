import os
import shutil
import subprocess
import time
import numpy as np
import logging
from datetime import datetime, timedelta
import re
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

import os
import numpy as np

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

def parse_simulation(sim_folder, expected_n_points=16128, Umax_simulation=None, L=64, nu=1.53e-5):
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
        # Create a Reynolds number channel (constant across all cells in this time step).
        Re_channel = np.full((combined_data.shape[0], 1), Re_sim)
        
        # Append the Reynolds number channel.
        combined_data_with_Re = np.column_stack([combined_data, Re_channel])
        results_per_time.append(combined_data_with_Re)
    
    results_array = np.stack(results_per_time, axis=0)  # shape: (num_time_steps, n_points, 4)
    return time_dirs, results_array


def gather_all_simulations(sim_folders, expected_n_points=16128, L=64, nu=1.53e-5):
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
    Also writes the new Umax value to a Umax.txt file in the simulation folder.
    """
    import os
    import random
    import logging

    u_file_path = os.path.join(sim_folder, "0", "U")
    if not os.path.exists(u_file_path):
        raise FileNotFoundError(f"U file not found at {u_file_path}")
    
    # Choose a new random Umax value in the specified range
    new_value = random.uniform(min_value, max_value)
    
    # Update the U file with the new Umax value
    update_Umax(u_file_path, new_value)
    
    # Write the Umax value to a separate file in the simulation folder for later reference.
    Umax_txt_path = os.path.join(sim_folder, "Umax.txt")
    with open(Umax_txt_path, "w") as f:
        f.write(str(new_value))
    
    logging.info(f"Updated {u_file_path}: Umax set to {new_value}")
    logging.info(f"Saved Umax value {new_value} in {Umax_txt_path}")

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
    num_copies = int(input("Enter the number of folders to create: "))
    main_folder = "Design_Point_0"
    start_time = time.time()
    # Step A: Copy folder
    new_folders = copy_main_folder(main_folder, num_copies)

    # Step B: Keep track of converged folders
    converged_folders = []

    # Step C: Run simulation steps, store only converged folders
    for folder in new_folders:
        # generate_U_file(folder)
        # run_setfields(folder)
        folder_start_time = time.time()
        update_Umax_in_simulation_folder(folder, min_value=2.39e-5, max_value=2.39e-3)
        generate_cell_centers(folder)
        converged = run_rhoPimpleFoam(folder)
        folder_end_time = time.time()
        elapsed_time = folder_end_time - folder_start_time
        logging.info(f"Simulation for {folder} completed in {elapsed_time:.2f} seconds")
        if converged:
            converged_folders.append(folder)

    # Step D: Parse fields only from converged folders → produce final npy
    if converged_folders:
        # gather_all_simulations() returns the combined results
        time_dirs, final_results = gather_all_simulations(converged_folders)
        np.save("results.npy", final_results)
        logging.info(f"Saved final results with shape: {final_results.shape}")
        print(f"\nFinal results saved to results.npy with shape {final_results.shape}")
    else:
        logging.warning("No simulations converged, skipping final file creation")
        print("\nNo simulations converged. No final results generated.")
    end_time = time.time()
    total_time_elapsed = end_time - start_time
    print(f"\nTotal execution time: {total_time_elapsed:.2f} seconds")
    logging.info(f"Total execution time: {total_time_elapsed:.2f} seconds")
if __name__ == "__main__":
    main()
