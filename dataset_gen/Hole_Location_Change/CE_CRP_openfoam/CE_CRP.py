from pde_base import PDEBase
import logging
import numpy as np
import os
from util import get_block_slices, write_scalar_field, write_vector_field

class CE_CRP(PDEBase):
    def generate_random_coefficients(self, p = None):
        """
        Generate random coefficients for perturbations.
        
        Args:
            p (int): Number of modes for perturbations.
            
        Returns:
            dict: A dictionary containing random coefficients for alpha0, beta0, alpha1, and beta1.
        """

        return {
            "alpha_x": np.random.uniform(-0.1, 0.1, (p, p)),
            "alpha_y": np.random.uniform(-0.1, 0.1, (p, p)),
            "beta_x": np.random.uniform(0, 1, (p, p)),
            "beta_y": np.random.uniform(0, 1, (p, p)),
            "rho_sub": np.random.uniform(0.1, 1.0, (p, p)),
            "p_sub": np.random.uniform(0.1, 1.0, (p, p)),
            "Ux_sub": np.random.uniform(-1.0, 1.0, (p, p)),
            "Uy_sub": np.random.uniform(-1.0, 1.0, (p, p))
        }

    def generate_U_file(self, folder, i_c, j_c, coefficients, hole_size=0.0625):
        # -------------------------
        # User Parameters
        # -------------------------
        p = 4           # number of partitions per dimension
        N = 128         # mesh resolution (128x128)
        R = 287         # specific gas constant
        # Hole definition
        # i_hole_min, i_hole_max = 60, 67
        # j_hole_min, j_hole_max = 60, 67

        # -------------------------
        # Generate Random Coefficients
        # -------------------------
        alpha_x = coefficients["alpha_x"]
        alpha_y = coefficients["alpha_y"]
        beta_x = coefficients["beta_x"]
        beta_y = coefficients["beta_y"]

        logging.info(f"Generated alpha_x: {alpha_x}")
        logging.info(f"Generated alpha_y: {alpha_y}")
        logging.info(f"Generated beta_x: {beta_x}")
        logging.info(f"Generated beta_y: {beta_y}")

        # -------------------------
        # Pre-generate Initial Conditions per Subdomain
        # -------------------------
        rho_sub = coefficients["rho_sub"]
        p_sub   = coefficients["p_sub"]
        Ux_sub  = coefficients["Ux_sub"]
        Uy_sub  = coefficients["Uy_sub"]

        logging.info(f"Generated random values for rho: {rho_sub}")
        logging.info(f"Generated random values for p: {p_sub}")
        logging.info(f"Generated random values for Ux: {Ux_sub}")
        logging.info(f"Generated random values for Uy: {Uy_sub}")

        # -------------------------
        # Define sigma_x and sigma_y
        # -------------------------
        def sigma_x_func(x, y):
            val = 0.0
            for i_ in range(p):
                for j_ in range(p):
                    val += alpha_x[i_, j_] * np.sin(2 * np.pi * (i_ + 1) * x 
                                                    + (j_ + 1) * y 
                                                    + beta_x[i_, j_])
            return val

        def sigma_y_func(x, y):
            val = 0.0
            for i_ in range(p):
                for j_ in range(p):
                    val += alpha_y[i_, j_] * np.sin(2 * np.pi * (i_ + 1) * x 
                                                    + (j_ + 1) * y 
                                                    + beta_y[i_, j_])
            return val

        # -------------------------
        # Structured Subdomain Initialization
        # We'll skip the hole in the center (8x8) of the 128x128 grid.
        # -------------------------
        dx = 1.0 / N
        dy = 1.0 / N
        # -----------------------------
        # STEP C: Reorder data by blocks
        # -----------------------------
        rho_list_blockOrder = []
        p_list_blockOrder   = []
        Ux_list_blockOrder  = []
        Uy_list_blockOrder  = []
        T_list_blockOrder   = []

        block_slices = get_block_slices(i_c=i_c, j_c=j_c, N=N, hole_size=8)

        # We must follow the same pattern blockMesh uses:
        for (iRange, jRange) in block_slices:
            for j_ in jRange:
                for i_ in iRange:
                    # Skip hole
                    # if (i_hole_min <= i_ <= i_hole_max) and (j_hole_min <= j_ <= j_hole_max):
                    #     continue

                    # Cell-center
                    x_c = (i_ + 0.5) * dx
                    y_c = (j_ + 0.5) * dy

                    # Perturbation logic
                    sx = sigma_x_func(x_c, y_c)
                    sy = sigma_y_func(x_c, y_c)
                    x_tilde = (x_c + sx) % 1.0
                    y_tilde = (y_c + sy) % 1.0

                    i_sub = min(max(int(np.floor((p+1) * x_tilde)), 0), p-1)
                    j_sub = min(max(int(np.floor((p+1) * y_tilde)), 0), p-1)

                    # Pull PDE solution from sub-fields
                    rho_val = rho_sub[i_sub, j_sub]
                    p_val   = p_sub[i_sub, j_sub]
                    Ux_val  = Ux_sub[i_sub, j_sub]
                    Uy_val  = Uy_sub[i_sub, j_sub]
                    
                    # Ideal gas law => T = p/(rho*R), guard for small rho
                    if rho_val > 1e-14:
                        T_val = p_val / (rho_val * R)
                    else:
                        T_val = 0.0

                    # Append to final lists
                    rho_list_blockOrder.append(rho_val)
                    p_list_blockOrder.append(p_val)
                    Ux_list_blockOrder.append(Ux_val)
                    Uy_list_blockOrder.append(Uy_val)
                    T_list_blockOrder.append(T_val)

        # Now *these* lists (rho_list_blockOrder, etc.) match the
        # exact block-by-block indexing that OpenFOAM uses internally.
        Ntot = len(rho_list_blockOrder)
        print("Block-ordered list length =", Ntot)

        # -----------------------------
        # STEP D: Write them out
        # -----------------------------


        zero_dir = os.path.join(folder, "0")
        os.makedirs(zero_dir, exist_ok=True)
        # Finally use the blockOrdered data
        write_scalar_field(os.path.join(zero_dir, "rho"), "rho", rho_list_blockOrder, "[1 -3 0 0 0 0 0]")
        write_scalar_field(os.path.join(zero_dir, "p"),   "p",   p_list_blockOrder,   "[1 -1 -2 0 0 0 0]")
        write_scalar_field(os.path.join(zero_dir, "T"),   "T",   T_list_blockOrder,   "[0 0 0 1 0 0 0]")
        write_vector_field(os.path.join(zero_dir, "U"),   "U",   Ux_list_blockOrder,  Uy_list_blockOrder, "[0 1 -1 0 0 0 0]")

        print("Done writing block-ordered fields.")