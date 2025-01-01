import numpy as np
import logging

# Configure logging
logging.basicConfig(
    filename="generate_ce_kh_setfields.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

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

# -------------------------
# Generate Random Coefficients for Perturbations
# -------------------------

# α and β for sigma0 and sigma1
alpha0 = np.random.uniform(0, 1, p)
beta0 = np.random.uniform(0, 1, p)
alpha1 = np.random.uniform(0, 1, p)
beta1 = np.random.uniform(0, 1, p)

logging.info(f"Generated alpha0: {alpha0}")
logging.info(f"Generated beta0: {beta0}")
logging.info(f"Generated alpha1: {alpha1}")
logging.info(f"Generated beta1: {beta1}")

# -------------------------
# Define sigma0(x) and sigma1(x)
# -------------------------
def sigma(x, alpha, beta, p):
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

# Create 2D meshgrid
X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')  # Shape: (N, N)

# -------------------------
# Compute Perturbations for All x
# -------------------------
sigma0 = sigma(X[:,0], alpha0, beta0, p)  # Shape: (N,)
sigma1 = sigma(X[:,0], alpha1, beta1, p)  # Shape: (N,)

# Broadcast sigma0 and sigma1 to match the 2D grid
sigma0_2D = np.tile(sigma0[:, np.newaxis], (1, N))  # Shape: (N, N)
sigma1_2D = np.tile(sigma1[:, np.newaxis], (1, N))  # Shape: (N, N)

# -------------------------
# Assign Initial Conditions
# -------------------------
rho_field = np.where(
    (Y < (0.25 + sigma0_2D)) | (Y > (0.75 + sigma1_2D)),
    rho_low,
    rho_high
)

vx_field = np.where(
    (Y < (0.25 + sigma0_2D)) | (Y > (0.75 + sigma1_2D)),
    vx_low,
    vx_high
)

vy_field = np.full((N, N), vy)

p_field = np.full((N, N), p_initial)

# Compute Temperature using Ideal Gas Law: T = p / (rho * R)
with np.errstate(divide='ignore', invalid='ignore'):
    T_field = p_field / (rho_field * R)
    T_field = np.nan_to_num(T_field)  # Replace NaNs and Infs if any

# -------------------------
# Logging the Initial Conditions Statistics
# -------------------------
logging.info(f"rho_field statistics: min={rho_field.min()}, max={rho_field.max()}, mean={rho_field.mean()}")
logging.info(f"vx_field statistics: min={vx_field.min()}, max={vx_field.max()}, mean={vx_field.mean()}")
logging.info(f"vy_field statistics: min={vy_field.min()}, max={vy_field.max()}, mean={vy_field.mean()}")
logging.info(f"p_field statistics: min={p_field.min()}, max={p_field.max()}, mean={p_field.mean()}")
logging.info(f"T_field statistics: min={T_field.min()}, max={T_field.max()}, mean={T_field.mean()}")

# -------------------------
# Function to Write Scalar Fields
# -------------------------
def write_scalar_field(filename, fieldName, fieldData, dimensions):
    """
    Write a scalar field to an OpenFOAM file.
    """
    Ntot = fieldData.size
    with open(filename, 'w') as f:
        f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
        f.write("  =========                 |\n")
        f.write("  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox\n")
        f.write("   \\    /   O peration     | Website:  https://openfoam.org\n")
        f.write("    \\  /    A nd           | Version:  8\n")
        f.write("     \\/     M anipulation  |\n")
        f.write("\\*---------------------------------------------------------------------------*/\n")
        f.write("FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       volScalarField;\n")
        f.write(f"    object      {fieldName};\n}}\n\n")
        f.write(f"dimensions      {dimensions};\n")
        f.write("internalField   nonuniform List<scalar>\n")
        f.write(f"{Ntot}\n(\n")
        # Write data in row-major order (y first)
        for j in range(N):
            for i in range(N):
                f.write(f"    {fieldData[i, j]:.6f}\n")
        f.write(");\n\nboundaryField\n{\n")
        f.write("    left\n    {\n        type            cyclic;\n    }\n")
        f.write("    right\n    {\n        type            cyclic;\n    }\n")
        f.write("    bottom\n    {\n        type            cyclic;\n    }\n")
        f.write("    top\n    {\n        type            cyclic;\n    }\n")
        f.write("    frontAndBack\n    {\n        type            empty;\n    }\n}\n")
    logging.info(f"Wrote scalar field to {filename}")

# -------------------------
# Function to Write Vector Fields
# -------------------------
def write_vector_field(filename, fieldName, Ux, Uy, dimensions):
    """
    Write a vector field to an OpenFOAM file.
    """
    Ntot = Ux.size
    with open(filename, 'w') as f:
        f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
        f.write("  =========                 |\n")
        f.write("  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox\n")
        f.write("   \\    /   O peration     | Website:  https://openfoam.org\n")
        f.write("    \\  /    A nd           | Version:  8\n")
        f.write("     \\/     M anipulation  |\n")
        f.write("\\*---------------------------------------------------------------------------*/\n")
        f.write("FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       volVectorField;\n")
        f.write(f"    object      {fieldName};\n}}\n\n")
        f.write(f"dimensions      {dimensions};\n")
        f.write("internalField   nonuniform List<vector>\n")
        f.write(f"{Ntot}\n(\n")
        # Write data in row-major order (y first)
        for j in range(N):
            for i in range(N):
                f.write(f"    ({Ux[i, j]:.6f} {Uy[i, j]:.6f} 0.0)\n")
        f.write(");\n\nboundaryField\n{\n")
        f.write("    left\n    {\n        type            cyclic;\n    }\n")
        f.write("    right\n    {\n        type            cyclic;\n    }\n")
        f.write("    bottom\n    {\n        type            cyclic;\n    }\n")
        f.write("    top\n    {\n        type            cyclic;\n    }\n")
        f.write("    frontAndBack\n    {\n        type            empty;\n    }\n}\n")
    logging.info(f"Wrote vector field to {filename}")

# -------------------------
# Write Fields to OpenFOAM Files
# -------------------------
write_scalar_field("0/rho", "rho", rho_field, "[1 -3 0 0 0 0 0]")
write_scalar_field("0/p", "p", p_field, "[1 -1 -2 0 0 0 0]")
write_scalar_field("0/T", "T", T_field, "[0 0 0 1 0 0 0]")
write_vector_field("0/U", "U", vx_field, vy_field, "[0 1 -1 0 0 0 0]")

logging.info("All fields have been written successfully.")

# -------------------------
# Summary Message
# -------------------------
print("Initial condition files (rho, U, p, T) have been generated successfully in the '0' directory.")
print("Please ensure that the '0' directory exists and that the mesh has been created using blockMesh before running this script.")
