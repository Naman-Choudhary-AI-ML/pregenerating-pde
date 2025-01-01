import os
import shutil
import subprocess
import time
import numpy as np
import logging
from datetime import datetime, timedelta

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

def parse_internal_field(lines):
    start_index = None
    end_index = None
    N = None
    
    for i, line in enumerate(lines):
        if 'internalField' in line and 'nonuniform' in line:
            pass
        if line.strip().isdigit():
            N = int(line.strip())
            start_index = i + 2
            end_index = start_index + N
            break
    
    if start_index is None or end_index is None or N is None:
        raise ValueError("Failed to parse internalField data.")
    
    data_lines = [l.strip() for l in lines[start_index:end_index]]
    return data_lines

def read_scalar_field(folder, time_dir, field_name):
    field_path = os.path.join(folder, time_dir, field_name)
    lines = read_field_lines(field_path)
    data_lines = parse_internal_field(lines)
    
    Nx = 128  
    Ny = 128  
    if len(data_lines) != Nx * Ny:
        raise ValueError(f"Data size {len(data_lines)} does not match Nx*Ny.")
    
    data = np.array([float(val) for val in data_lines])
    data = data.reshape((Ny, Nx))
    return data

def read_vector_field(folder, time_dir, field_name):
    field_path = os.path.join(folder, time_dir, field_name)
    lines = read_field_lines(field_path)
    data_lines = parse_internal_field(lines)
    
    Nx = 128
    Ny = 128
    if len(data_lines) != Nx * Ny:
        raise ValueError(f"Data size {len(data_lines)} does not match Nx*Ny.")
    
    U_data = []
    for val in data_lines:
        val = val.strip().strip("()")
        comps = val.split()
        Ux, Uy = float(comps[0]), float(comps[1])
        U_data.append([Ux, Uy])
    
    U_data = np.array(U_data).reshape((Ny, Nx, 2))
    return U_data

def extract_results(folder):
    time_dirs = get_time_directories(folder)
    Nx = 128
    Ny = 128
    num_times = len(time_dirs)
    data_array = np.zeros((num_times, 4, Ny, Nx), dtype=np.float64)
    R = 287

    for i, tdir in enumerate(time_dirs):
        U = read_vector_field(folder, tdir, "U")
        p = read_scalar_field(folder, tdir, "p")
        T = read_scalar_field(folder, tdir, "T")
        rho = p / (T * R)
        data_array[i, 0, :, :] = rho
        data_array[i, 1, :, :] = U[:, :, 0]
        data_array[i, 2, :, :] = U[:, :, 1]
        data_array[i, 3, :, :] = p

    return data_array

def main():
    num_copies = int(input("Enter the number of folders to create: "))
    main_folder = "Design_Point_0"

    generate_setfields_path = os.path.join(main_folder, "generate_setfields.py")
    logging.info(f"Checking if generate_setfields.py exists in the original path: {generate_setfields_path}")
    if not os.path.exists(generate_setfields_path):
        logging.error("Error: generate_setfields.py not found.")
        return

    new_folders = copy_main_folder(main_folder, num_copies)
    all_results = []
    for idx, folder in enumerate(new_folders):
        
        generate_U_file(folder)
        # run_setfields(folder)

        u_file = os.path.join(folder, "0", "U")
        if os.path.exists(u_file):
            # Run and check convergence
            converged = run_rhoPimpleFoam(folder)
            if converged:
                # Extract results
                result_data = extract_results(folder)
                all_results.append(result_data)
            else:
                logging.warning(f"Skipping adding results from {folder} since simulation did not converge.")
        else:
            logging.warning(f"Skipping simulation in {folder} as U file was not found.")

        log_periodic_success()  # Log success every 30 minutes

    if all_results:
        all_results = np.stack(all_results, axis=0)
        np.save("results.npy", all_results)
        logging.info(f"Results saved to results.npy with shape {all_results.shape}")
    else:
        logging.info("No converged simulations found. No results saved.")

if __name__ == "__main__":
    main()