import logging
import subprocess
import os
import json
import numpy as np
from pde_getter import get_pde_variant
from util import copy_main_folder, random_hole_centers, generate_blockMeshDict, run_rhoPimpleFoam, gather_all_simulations, generate_cell_centers, parse_cell_centers, validate_missing_coordinates 
from util import run_icoFoam
def main():
    # PDE variant name from user:
    pde_name = input("Enter which PDE variant to use (ce_xy/ns_xy): ").strip().lower()

    # Instantiate PDE object
    pde_obj, base_folder = get_pde_variant(pde_name)
    num_copies = int(input("Enter the number of folders to create: "))
    # Define the base `Design_Point_0` path for the selected PDE
    main_folder = os.path.join(base_folder, "Design_Point_0")
    if not os.path.exists(main_folder):
        raise FileNotFoundError(f"Main folder {main_folder} does not exist!")

    # Ask the user if the parameters should be fixed
    fixed_params = input("Keep parameters fixed for all hole locations? (yes/no): ").strip().lower() in ["yes", "y"]

    # 2) Copy the main folder 'num_copies' times
    new_folders = copy_main_folder(main_folder, num_copies)

    N = 128
    hole_centers = random_hole_centers(num_copies, N)
    # hole_centers = ((0.4,0.6))
    # Store hole centers to save later
    hole_center_list = []

    # Sanity check: If your user wants to create e.g. 2 new folders,
    # make sure you have at least 2 hole centers
    if num_copies > len(hole_centers):
        raise ValueError("Not enough hole centers specified for the number of folders.")
    
    # 3) Generate random coefficients *once* for all holes
    #    Choose 'p' based on your problem setup
    random_coeffs = None
    if fixed_params:
        p = 4  # or the dimension of coefficients as required
        random_coeffs = pde_obj.generate_random_coefficients(p)

    # Lists to store final data
    converged_folders = []  # For folders that pass all checks
    sim_data = []           # For JSON serialization
    # We'll store final OpenFOAM results into a NumPy array later

    # 3) Run simulation steps, store only converged folders
    #    Enumerate new_folders so we can pick the correct hole center
    for i, folder in enumerate(new_folders, start=1):
        # Retrieve the hole center for this design point
        i_c, j_c = hole_centers[i - 1]
        hole_center_list.append((i_c, j_c)) #save hole centre
        # cx, cy = hole_centers
        logging.info(f"Processing folder {folder} with hole center ({i_c:.4f}, {j_c:.4f})")
        # Convert continuous coords [0..1] => integer indices [0..127]
        bmd_path = os.path.join(folder, "system", "blockMeshDict")

        try:
            generate_blockMeshDict(i_c=i_c, j_c=j_c, output_path=bmd_path, run_blockMesh=True)
        except subprocess.CalledProcessError as e:
            print(f"BlockMesh failed for design_point_{i}: {e}")
            # Decide whether to skip this design point or abort completely.
            # Here, we continue to the next folder.
            continue

        if not fixed_params:
            p = 4  # Dimension for coefficients (or parameterize this as needed)
            random_coeffs = pde_obj.generate_random_coefficients(p=p)
        pde_obj.generate_U_file(folder, i_c, j_c, random_coeffs, hole_size=0.0625)         # This replaces the old generate_setfields.py call
        # run_setfields(folder)         # If you do use setFields or additional steps
        # 3C) Run solver
        if pde_name == "ns_gauss":
            if not run_icoFoam(folder):
                logging.warning(f"icoFoam solver failed for folder {folder}. Skipping.")
                continue
        else:
            if not run_rhoPimpleFoam(folder):
                logging.warning(f"rhoPimpleFoam solver failed for folder {folder}. Skipping.")
                continue
        
        # 3D) Generate and validate cell centers
        c_file = generate_cell_centers(folder)
        if not c_file:
            logging.error(f"Cell centers file not found for folder {folder}. Skipping.")
            continue

        coords = parse_cell_centers(c_file)
        missing_x, missing_y = validate_missing_coordinates(coords, hole_size=0.0625)
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
            "c_contents": c_contents
        })

    # 4) Write out the simulation data (cell centers, hole info, etc.) to JSON
    if sim_data:
        sim_data_path = os.path.join(base_folder, "sim_data_with_contents.json")
        with open(sim_data_path, "w") as jf:
            json.dump(sim_data, jf)
        logging.info(f"Sim data saved to {sim_data_path}")
    else:
        logging.warning("No valid simulations or cell center validations. No sim data generated.")

    # Save hole centers to JSON
    hole_center_path = os.path.join(base_folder, "hole_centers.json")
    with open(hole_center_path, "w") as f:
        json.dump(hole_center_list, f)

    # 5) Gather final results only from converged folders
    if converged_folders:
        try:
            # time_dirs, final_results = gather_all_simulations(converged_folders)
            time_dirs, final_results = gather_all_simulations(converged_folders, hole_center_list, pde_name=pde_name)

        except FileNotFoundError as e:
            logging.error(f"Error processing converged folders: {e}")
            return

        # Save final results to .npy
        results_path = os.path.join(base_folder, "results2.npy")
        np.save(results_path, final_results)
        logging.info(f"Saved final results with shape: {final_results.shape}")
        print(f"\nFinal results saved to results.npy with shape {final_results.shape}")
    else:
        logging.warning("No valid (converged) folders. Final results not generated.")
        print("\nNo valid simulations. No final results generated.")

if __name__ == "__main__":
    main()