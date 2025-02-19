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
from util import copy_main_folder, random_hole_centers, generate_blockMeshDict, run_rhoPimpleFoam, generate_cell_centers, gather_all_simulations
from pde_getter import get_pde_variant

def main():
    # PDE variant name from user:
    pde_name = input("Enter which PDE variant to use (pde1/pde2): ").strip().lower()

    # Instantiate PDE object
    pde_obj, base_folder = get_pde_variant(pde_name)
    num_copies = int(input("Enter the number of folders to create: "))
    main_folder = os.path.join(base_folder, "Design_Point_0")
    if not os.path.exists(main_folder):
        raise FileNotFoundError(f"Main folder {main_folder} does not exist!")

    # Ask the user if the parameters should be fixed
    fixed_params = input("Keep parameters fixed for all hole locations? (yes/no): ").strip().lower() in ["yes", "y"]

    # 2) Copy the main folder 'num_copies' times
    new_folders = copy_main_folder(main_folder, num_copies)

    N = 128
    min_size = 8
    max_size = 30
    hole_centers = random_hole_centers(num_copies, N, min_size, max_size)
    # hole_centers = ((0.4,0.6))

    # Sanity check: If your user wants to create e.g. 2 new folders,
    # make sure you have at least 2 hole centers
    if num_copies > len(hole_centers):
        raise ValueError("Not enough hole centers specified for the number of folders.")
    
    # 3) Generate random coefficients *once* for all holes
    #    Choose 'p' based on your problem setup
    random_coeffs = None
    if fixed_params:
        p = 4  # or the dimension of coefficients as required
        random_coeffs = pde_obj.generate_random_coefficients(p=p)

    # Lists to store final data
    converged_folders = []  # For folders that pass all checks
    sim_data = []           # For JSON serialization
    final_tensors = [] 
    # We'll store final OpenFOAM results into a NumPy array later

    # 3) Run simulation steps, store only converged folders
    #    Enumerate new_folders so we can pick the correct hole center
    for i, folder in enumerate(new_folders, start=1):
        # Retrieve the hole center for this design point
        i_c, j_c, hole_size = hole_centers[i - 1]
        # cx, cy = hole_centers
        logging.info(f"Processing folder {folder} with hole center ({i_c:.4f}, {j_c:.4f})")
        # Convert continuous coords [0..1] => integer indices [0..127]
        bmd_path = os.path.join(folder, "system", "blockMeshDict")

        try:
            generate_blockMeshDict(i_c, j_c, hole_size=hole_size, output_path=bmd_path, run_blockMesh=True)
        except subprocess.CalledProcessError as e:
            print(f"BlockMesh failed for design_point_{i}: {e}")
            # Decide whether to skip this design point or abort completely.
            # Here, we continue to the next folder.
            continue

        if not fixed_params:
            p = 4  # Dimension for coefficients (or parameterize this as needed)
            random_coeffs = pde_obj.generate_random_coefficients(p=p)
        pde_obj.generate_U_file(folder, i_c, j_c, hole_size=hole_size, coefficients=random_coeffs)         # This replaces the old generate_setfields.py call
        # run_setfields(folder)         # If you do use setFields or additional steps
        # 3C) Run solver
        if not run_rhoPimpleFoam(folder):
            logging.warning(f"Solver failed for folder {folder}. Skipping.")
            continue
        
        # 3D) Generate and validate cell centers
        c_file = generate_cell_centers(folder)
        if not c_file:
            logging.error(f"Cell centers file not found for folder {folder}. Skipping.")
            continue
        converged_folders.append(folder)
        # Optionally store the contents of C file (or other data) in sim_data
        with open(c_file, "r") as f:
            c_contents = f.read()

        sim_data.append({
            "folder": folder,
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
        # If all steps succeeded, append the folder to converged_folders

    # 5) Once all successful sims are collected, gather results
    if not converged_folders:
        logging.error("No simulations were successful. Exiting.")
        return

    final_time_dirs, final_data = gather_all_simulations(converged_folders)

    if final_data is None:
        logging.error("No simulations successfully processed. Exiting.")
        return

    # 6) Save the final dataset
    logging.info(f"Final dataset shape: {final_data.shape}")
    np.save("all_sims_data.npy", final_data)

    # Save the time directories
    with open("time_dirs.txt", "w") as f:
        for t in final_time_dirs:
            f.write(str(t) + "\n")

    logging.info("All done.")

if __name__ == "__main__":
    main()