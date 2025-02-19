import numpy as np
import logging
import os
from util import get_block_slices, write_scalar_field, write_vector_field

from pde_base import PDEBase

class CE_KH(PDEBase):
    def generate_random_coefficients(self, p=None):
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
    def generate_U_file(self, folder, i_c, j_c, coefficients, hole_size):
        # -------------------------
        # User Parameters
        # -------------------------
        p = 4              # Number of modes for perturbations
        N = 128            # Mesh resolution (N x N)
        epsilon = 0.05     # Perturbation amplitude
        R = 287.0          # Specific gas constant for air (J/(kg·K))

        # Initial condition parameters
        rho_low = 1.0      # Density for regions y < 0.25 + sigma0 or y > 0.75 + sigma1
        rho_high = 2.0     # Density for regions 0.25 + sigma0 <= y <= 0.75 + sigma1
        vx_low = 0.5       # Velocity vx for rho_low regions
        vx_high = -0.5     # Velocity vx for rho_high regions
        vy = 0.0           # Velocity vy (constant)
        p_initial = 2.5    # Pressure (constant)

        Nx, Ny = 128, 128

        # -------------------------
        # Generate Random Coefficients for Perturbations
        # -------------------------
        # Generate Random Coefficients for Perturbations
        # coefficients = generate_random_coefficients(p)
        alpha0 = coefficients["alpha0"]
        beta0 = coefficients["beta0"]
        alpha1 = coefficients["alpha1"]
        beta1 = coefficients["beta1"]

        logging.info(f"Generated alpha0: {alpha0}")
        logging.info(f"Generated beta0: {beta0}")
        logging.info(f"Generated alpha1: {alpha1}")
        logging.info(f"Generated beta1: {beta1}")

        # -------------------------
        # Define sigma0(x) and sigma1(x)
        # -------------------------
        def sigma(x, alpha, beta, p):
            """
            sigma function for x array, combining p modes.
            x, alpha, beta are 1D arrays or scalars
            """
            val = np.zeros_like(x)
            j_indices = np.arange(1, p+1)
            for j in range(p):
                val += alpha[j] * np.cos(2 * np.pi * j_indices[j] * (x + beta[j]))
            return epsilon * val

        # -------------------------
        # Create Coordinate Arrays for Cell Centers
        # -------------------------
        dx = 1.0 / N
        dy = 1.0 / N

        x_coords = (np.arange(N) + 0.5) * dx  # Cell centers in x
        y_coords = (np.arange(N) + 0.5) * dy  # Cell centers in y

        # Make 2D meshgrid (N x N)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')  # shape: (N, N)

        # -------------------------
        # Compute Perturbations
        # -------------------------
        # We only need sigma0(x) and sigma1(x) for x in [0..N-1], so let's use X[:,0]
        # i.e. the x-values for the first column, repeated across y dimension
        sigma0_1D = sigma(X[:,0], alpha0, beta0, p)  # shape (N,)
        sigma1_1D = sigma(X[:,0], alpha1, beta1, p)  # shape (N,)

        # Broadcast sigma0 and sigma1 across columns
        sigma0_2D = np.tile(sigma0_1D[:, np.newaxis], (1, N))  # shape: (N, N)
        sigma1_2D = np.tile(sigma1_1D[:, np.newaxis], (1, N))  # shape: (N, N)

        # -------------------------
        # Assign Initial Conditions (full NxN)
        # -------------------------
        rho_field_full = np.where(
            (Y < (0.25 + sigma0_2D)) | (Y > (0.75 + sigma1_2D)),
            rho_low,
            rho_high
        )

        vx_field_full = np.where(
            (Y < (0.25 + sigma0_2D)) | (Y > (0.75 + sigma1_2D)),
            vx_low,
            vx_high
        )

        vy_field_full = np.full((N, N), vy)
        p_field_full  = np.full((N, N), p_initial)

        # Temperature using Ideal Gas
        with np.errstate(divide='ignore', invalid='ignore'):
            T_field_full = p_field_full / (rho_field_full * R)
            T_field_full = np.nan_to_num(T_field_full)

        # Log some stats
        logging.info(f"rho_field statistics: min={rho_field_full.min()}, max={rho_field_full.max()}, mean={rho_field_full.mean()}")
        logging.info(f"vx_field statistics: min={vx_field_full.min()}, max={vx_field_full.max()}, mean={vx_field_full.mean()}")
        logging.info(f"vy_field statistics: min={vy_field_full.min()}, max={vy_field_full.max()}, mean={vy_field_full.mean()}")
        logging.info(f"p_field statistics: min={p_field_full.min()}, max={p_field_full.max()}, mean={p_field_full.mean()}")
        logging.info(f"T_field statistics: min={T_field_full.min()}, max={T_field_full.max()}, mean={T_field_full.mean()}")

        # -------------------------
        # Now reorder by blocks to align with blockMesh ordering
        # -------------------------
        rho_blockOrder = []
        p_blockOrder   = []
        T_blockOrder   = []
        vx_blockOrder  = []
        vy_blockOrder  = []

        block_slices = get_block_slices(i_c=i_c, j_c=j_c, N=N, hole_size=hole_size)
        
        for (iRange, jRange) in block_slices:
            for j_ in jRange:
                for i_ in iRange:
                    # if (i_hole_min <= i_ <= i_hole_max) and (j_hole_min <= j_ <= j_hole_max):
                    #     # It's in the hole or out of domain, skip
                    #     continue
                    rho_blockOrder.append(rho_field_full[i_, j_])
                    p_blockOrder.append(p_field_full[i_, j_])
                    T_blockOrder.append(T_field_full[i_, j_])
                    vx_blockOrder.append(vx_field_full[i_, j_])
                    vy_blockOrder.append(vy_field_full[i_, j_])

        Ntot = len(rho_blockOrder)
        logging.info(f"Block-ordered enumeration: total valid cells = {Ntot}")

        # Finally write
        zero_dir = os.path.join(folder, "0")
        os.makedirs(zero_dir, exist_ok=True)

        write_scalar_field(os.path.join(zero_dir, "rho"), "rho", rho_blockOrder, "[1 -3 0 0 0 0 0]")
        write_scalar_field(os.path.join(zero_dir, "p"),   "p",   p_blockOrder,   "[1 -1 -2 0 0 0 0]")
        write_scalar_field(os.path.join(zero_dir, "T"),   "T",   T_blockOrder,   "[0 0 0 1 0 0 0]")
        write_vector_field(os.path.join(zero_dir, "U"),   "U",   vx_blockOrder, vy_blockOrder, "[0 1 -1 0 0 0 0]")

        logging.info("All fields have been written successfully in block order.")
        print("Initial condition fields (rho, U, p, T) have been generated in the '0' directory, skipping the hole and reordering by blocks.")

        # Optional: Check that U file was created
        u_file = os.path.join(zero_dir, "U")
        if os.path.exists(u_file):
            print(f"U file generated successfully in folder: {folder}")
            logging.info(f"U file generated successfully in folder: {folder}")
        else:
            print(f"Error: U file was not created in {folder}. Check generation logic.")
            logging.error(f"Error: U file was not created in {folder}. Check generation logic.")