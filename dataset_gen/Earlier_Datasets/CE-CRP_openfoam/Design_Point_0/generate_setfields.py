import numpy as np
import logging

# Configure logging
logging.basicConfig(
    filename="generate_setfields.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------
# User Parameters
# -------------------------
p = 4           # number of partitions per dimension
N = 128         # mesh resolution (128x128)
R = 287         # specific gas constant

def random_sample(low, high, size=None):
    return np.random.uniform(low, high, size)

# -------------------------
# Generate Random Coefficients
# -------------------------
alpha_x = random_sample(-0.1, 0.1, (p, p))
alpha_y = random_sample(-0.1, 0.1, (p, p))
beta_x = random_sample(0, 1, (p, p))
beta_y = random_sample(0, 1, (p, p))

logging.info(f"Generated alpha_x: {alpha_x}")
logging.info(f"Generated alpha_y: {alpha_y}")
logging.info(f"Generated beta_x: {beta_x}")
logging.info(f"Generated beta_y: {beta_y}")

# -------------------------
# Pre-generate Initial Conditions per Subdomain
# -------------------------
rho_sub = random_sample(0.1, 1.0, (p, p))
p_sub = random_sample(0.1, 1.0, (p, p))
Ux_sub = random_sample(-1.0, 1.0, (p, p))
Uy_sub = random_sample(-1.0, 1.0, (p, p))

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
            val += alpha_x[i_, j_] * np.sin(2 * np.pi * (i_ + 1) * x + (j_ + 1) * y + beta_x[i_, j_])
    return val

def sigma_y_func(x, y):
    val = 0.0
    for i_ in range(p):
        for j_ in range(p):
            val += alpha_y[i_, j_] * np.sin(2 * np.pi * (i_ + 1) * x + (j_ + 1) * y + beta_y[i_, j_])
    return val

# -------------------------
# Pre-generate Initial Conditions per Subdomain
# -------------------------
# Each subdomain D_{i,j} has one set of ICs applied uniformly to all its cells
rho_sub = np.random.uniform(0.0, 1.0, (p, p))
p_sub   = np.random.uniform(0.0, 1.0, (p, p))
Ux_sub  = np.random.uniform(-1.0, 1.0, (p, p))
Uy_sub  = np.random.uniform(-1.0, 1.0, (p, p))

# -------------------------
# Structured Subdomain Initialization
# -------------------------
dx = 1.0 / N
dy = 1.0 / N

# Initialize fields
rho_field = np.zeros((N, N))
p_field   = np.zeros((N, N))
U_field_x = np.zeros((N, N))
U_field_y = np.zeros((N, N))
T_field   = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        x_c = (i + 0.5) * dx
        y_c = (j + 0.5) * dy

        # Compute perturbations
        sx = sigma_x_func(x_c, y_c)
        sy = sigma_y_func(x_c, y_c)

        # Compute perturbed coordinates (mod 1 for periodicity)
        x_tilde = (x_c + sx) % 1.0
        y_tilde = (y_c + sy) % 1.0

        # Determine subdomain indices
        i_sub = int(np.floor((p+1)*x_tilde))
        j_sub = int(np.floor((p+1)*y_tilde))
        # Clamp indices just in case (should not be needed but for safety)
        i_sub = min(max(i_sub, 0), p-1)
        j_sub = min(max(j_sub, 0), p-1)

        # Retrieve pre-generated subdomain initial conditions
        rho_val = rho_sub[i_sub, j_sub]
        p_val   = p_sub[i_sub, j_sub]
        Ux_val  = Ux_sub[i_sub, j_sub]
        Uy_val  = Uy_sub[i_sub, j_sub]
        T_val   = p_val / (rho_val * R) if rho_val > 0 else 0.0  # Avoid division by zero

        # Assign values
        rho_field[i, j] = rho_val
        p_field[i, j]   = p_val
        U_field_x[i, j] = Ux_val
        U_field_y[i, j] = Uy_val
        T_field[i, j]   = T_val

# -------------------------
# Write Fields to OpenFOAM Files
# -------------------------
def write_scalar_field(filename, fieldName, fieldData, dimensions):
    Ntot = fieldData.size
    with open(filename, 'w') as f:
        f.write("FoamFile\n{\n    version 2.0;\n    format ascii;\n    class volScalarField;\n")
        f.write(f"    object {fieldName};\n}}\n\n")
        f.write(f"dimensions      {dimensions};\n")
        f.write("internalField   nonuniform List<scalar>\n")
        f.write(f"{Ntot}\n(\n")
        for val in fieldData.flatten():
            f.write(f"{val}\n")
        f.write(")\n;\n\nboundaryField\n{\n")
        f.write("    left\n    {\n        type            cyclic;\n    }\n")
        f.write("    right\n    {\n        type            cyclic;\n    }\n")
        f.write("    bottom\n    {\n        type            cyclic;\n    }\n")
        f.write("    top\n    {\n        type            cyclic;\n    }\n")
        f.write("    frontAndBack\n    {\n        type            empty;\n    }\n}\n")

def write_vector_field(filename, fieldName, Ux, Uy, dimensions):
    Ntot = Ux.size
    with open(filename, 'w') as f:
        f.write("FoamFile\n{\n    version 2.0;\n    format ascii;\n    class volVectorField;\n")
        f.write(f"    object {fieldName};\n}}\n\n")
        f.write(f"dimensions      {dimensions};\n")
        f.write("internalField   nonuniform List<vector>\n")
        f.write(f"{Ntot}\n(\n")
        for (ux, uy) in zip(Ux.flatten(), Uy.flatten()):
            f.write(f"({ux} {uy} 0)\n")
        f.write(")\n;\n\nboundaryField\n{\n")
        f.write("    left\n    {\n        type            cyclic;\n    }\n")
        f.write("    right\n    {\n        type            cyclic;\n    }\n")
        f.write("    bottom\n    {\n        type            cyclic;\n    }\n")
        f.write("    top\n    {\n        type            cyclic;\n    }\n")
        f.write("    frontAndBack\n    {\n        type            empty;\n    }\n}\n")

write_scalar_field("0/rho", "rho", rho_field, "[1 -3 0 0 0 0 0]")
write_scalar_field("0/p", "p", p_field, "[1 -1 -2 0 0 0 0]")
write_scalar_field("0/T", "T", T_field, "[0 0 0 1 0 0 0]")
write_vector_field("0/U", "U", U_field_x, U_field_y, "[0 1 -1 0 0 0 0]")
