import os
import shutil
import subprocess
import time
import numpy as np
import logging
from datetime import datetime, timedelta
import re

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

def main():
    num_copies = int(input("Enter the number of folders to create: "))
    main_folder = "Design_Point_0"

    generate_setfields_path = os.path.join(main_folder, "generate_setfields.py")
    logging.info(
        f"Checking if generate_setfields.py exists in the original path: {generate_setfields_path}"
    )
    if not os.path.exists(generate_setfields_path):
        logging.error("Error: generate_setfields.py not found.")
        return

    # Step A: Copy folder
    new_folders = copy_main_folder(main_folder, num_copies)

    # Step B: Keep track of converged folders
    converged_folders = []

    # Step C: Run simulation steps, store only converged folders
    for folder in new_folders:
        generate_U_file(folder)
        run_setfields(folder)
        generate_cell_centers(folder)
        converged = run_rhoPimpleFoam(folder)
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

if __name__ == "__main__":
    main()
