import numpy as np
import logging
import os
from scipy.fftpack import dst, idst
from util import get_block_slices, write_vector_field
from pde_base import PDEBase

class NS_Gauss(PDEBase):
    def generate_random_coefficients(self, p=None):
        """
        No random coefficients are needed since num_gaussians and max_velocity are fixed.
        Returning an empty dictionary to comply with PDEBase structure.
        """
        return {}

    def generate_gaussian_velocity(self, X, Y, num_gaussians=100, max_velocity=1.5):
        """
        Generate Gaussian-based velocity field for NS-Gauss PDE.

        Args:
            X, Y (2D arrays): Meshgrid coordinates for cell centers.
            num_gaussians (int): Fixed number of Gaussian components (100).
            max_velocity (float): Fixed maximum allowable velocity magnitude (1.5).

        Returns:
            vx_field, vy_field: 2D velocity fields for x and y components.
        """
        nx, ny = X.shape
        omega_grid = np.zeros((nx, ny))  # Vorticity grid

        # Adding Gaussian vortices
        for _ in range(num_gaussians):
            alpha = np.random.uniform(-1, 1)
            sigma = np.random.uniform(0.01, 0.1)
            x_i, y_i = np.random.uniform(0, 1), np.random.uniform(0, 1)

            omega_grid += alpha / sigma * np.exp(-((X - x_i) ** 2 + (Y - y_i) ** 2) / (2 * sigma ** 2))

        # Solving for stream function using DST
        omega_dst = dst(dst(omega_grid, type=1, axis=0), type=1, axis=1)
        kx = np.fft.fftfreq(nx, d=1 / nx) ** 2
        ky = np.fft.fftfreq(ny, d=1 / ny) ** 2
        denominator = kx[:, None] + ky[None, :]
        denominator[0, 0] = 1  # Prevent division by zero

        psi_dst = omega_dst / denominator
        psi_grid = idst(idst(psi_dst, type=1, axis=0), type=1, axis=1)

        # Velocity fields (u = ∂ψ/∂y, v = -∂ψ/∂x)
        vx_field = np.gradient(psi_grid, axis=1)
        vy_field = -np.gradient(psi_grid, axis=0)

        # Normalization
        velocity_magnitude = np.sqrt(vx_field ** 2 + vy_field ** 2)
        max_magnitude = velocity_magnitude.max()

        if max_magnitude > max_velocity:
            scale_factor = max_velocity / max_magnitude
            vx_field *= scale_factor
            vy_field *= scale_factor

        return vx_field, vy_field

    def generate_U_file(self, folder, i_c, j_c, coefficients, hole_size=0.05):
        """
        Generate and write the U file based on NS-Gauss initial conditions.

        Args:
            folder (str): Directory to save U file.
            i_c, j_c (int): Center indices for hole location.
            coefficients (dict): Not used here since initialization is fixed.
            hole_size (float): Size of the hole in the domain.
        """
        logging.info(f"[NS_Gauss] Generating U file in folder: {folder}")

        # -------------------------
        # Grid and Mesh Setup
        # -------------------------
        N = 128  # Mesh resolution (N x N)
        dx, dy = 1.0 / N, 1.0 / N

        x_coords = (np.arange(N) + 0.5) * dx  # Cell centers in x
        y_coords = (np.arange(N) + 0.5) * dy  # Cell centers in y

        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')  # Shape: (N, N)

        # -------------------------
        # Generate Gaussian Velocity Field
        # -------------------------
        vx_field, vy_field = self.generate_gaussian_velocity(X, Y)

        # -------------------------
        # Block Reordering for OpenFOAM
        # -------------------------
        vx_blockOrder = []
        vy_blockOrder = []

        block_slices = get_block_slices(i_c=i_c, j_c=j_c, N=N, hole_size=int(hole_size * N))

        for (i_range, j_range) in block_slices:
            for j_ in j_range:
                for i_ in i_range:
                    vx_blockOrder.append(vx_field[i_, j_])
                    vy_blockOrder.append(vy_field[i_, j_])

        Ntot = len(vx_blockOrder)
        logging.info(f"[NS_Gauss] Total valid cells after block ordering: {Ntot}")

        # -------------------------
        # Write the U File
        # -------------------------
        zero_dir = os.path.join(folder, "0")
        os.makedirs(zero_dir, exist_ok=True)

        write_vector_field(
            os.path.join(zero_dir, "U"),
            "U",
            vx_blockOrder,
            vy_blockOrder,
            dimensions="[0 1 -1 0 0 0 0]"
        )

        logging.info(f"[NS_Gauss] U file generated successfully in {folder}")

        # Validation Check
        u_file = os.path.join(zero_dir, "U")
        if os.path.exists(u_file):
            print(f"U file generated successfully at: {u_file}")
        else:
            print(f"Error: U file not created in {folder}")
