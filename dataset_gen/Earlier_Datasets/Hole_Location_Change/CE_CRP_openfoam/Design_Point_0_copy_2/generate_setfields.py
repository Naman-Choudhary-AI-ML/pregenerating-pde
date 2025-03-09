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
# Hole definition
i_hole_min, i_hole_max = 60, 67
j_hole_min, j_hole_max = 60, 67

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
p_sub   = random_sample(0.1, 1.0, (p, p))
Ux_sub  = random_sample(-1.0, 1.0, (p, p))
Uy_sub  = random_sample(-1.0, 1.0, (p, p))

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
# (Re)Generate Subdomain-level random fields
# -------------------------
# You can keep these or reuse the ones above. Just for consistency:
rho_sub = np.random.uniform(0.0, 1.0, (p, p))
p_sub   = np.random.uniform(0.0, 1.0, (p, p))
Ux_sub  = np.random.uniform(-1.0, 1.0, (p, p))
Uy_sub  = np.random.uniform(-1.0, 1.0, (p, p))

# -------------------------
# Structured Subdomain Initialization
# We'll skip the hole in the center (8x8) of the 128x128 grid.
# -------------------------
dx = 1.0 / N
dy = 1.0 / N

# The block order from your blockMesh:
blocks_ij = [
    (range(0, 60),   range(0, 60)),    # bottom-left
    (range(60, 68),  range(0, 60)),    # bottom-center
    (range(68, 128), range(0, 60)),    # bottom-right
    (range(0, 60),   range(60, 68)),   # middle-left
    (range(68, 128), range(60, 68)),   # middle-right
    (range(0, 60),   range(68, 128)),  # top-left
    (range(60, 68),  range(68, 128)),  # top-center
    (range(68, 128), range(68, 128)),  # top-right
]

# -----------------------------
# STEP A & B: Row-major enumeration & store data
# -----------------------------
rho_list_rowMajor = []
p_list_rowMajor   = []
Ux_list_rowMajor  = []
Uy_list_rowMajor  = []
T_list_rowMajor   = []

# We also build a dictionary to map (i,j) -> rowMajorIndex
cellIndexMap = {}

rowMajorIndex = 0
for j in range(N):                 # outer loop over y
    for i in range(N):             # inner loop over x
        # Skip hole
        if (i_hole_min <= i <= i_hole_max) and (j_hole_min <= j <= j_hole_max):
            continue

        x_c = (i + 0.5)*dx
        y_c = (j + 0.5)*dy

        # Perturb
        sx = sigma_x_func(x_c, y_c)
        sy = sigma_y_func(x_c, y_c)
        x_tilde = (x_c + sx) % 1.0
        y_tilde = (y_c + sy) % 1.0

        i_sub = min(max(int(np.floor((p+1)*x_tilde)), 0), p-1)
        j_sub = min(max(int(np.floor((p+1)*y_tilde)), 0), p-1)

        rho_val = rho_sub[i_sub, j_sub]
        p_val   = p_sub[i_sub, j_sub]
        Ux_val  = Ux_sub[i_sub, j_sub]
        Uy_val  = Uy_sub[i_sub, j_sub]
        T_val   = p_val/(rho_val*R) if rho_val > 1e-14 else 0.0

        # Append to rowMajor lists
        rho_list_rowMajor.append(rho_val)
        p_list_rowMajor.append(p_val)
        Ux_list_rowMajor.append(Ux_val)
        Uy_list_rowMajor.append(Uy_val)
        T_list_rowMajor.append(T_val)

        # Record the rowMajor index
        cellIndexMap[(i,j)] = rowMajorIndex

        rowMajorIndex += 1

print("Row-major enumeration done. Total valid cells =", rowMajorIndex)

# -----------------------------
# STEP C: Reorder data by blocks
# -----------------------------
rho_list_blockOrder = []
p_list_blockOrder   = []
Ux_list_blockOrder  = []
Uy_list_blockOrder  = []
T_list_blockOrder   = []

# We must follow the same pattern blockMesh uses:
for (iRange, jRange) in blocks_ij:
    for j_ in jRange:
        for i_ in iRange:
            # (i_, j_) is definitely not in the hole in your blockMesh
            # But let's be safe, or skip if there's any mismatch:
            if (i_, j_) not in cellIndexMap:
                # That cell might be in the hole, skip
                continue

            rowIdx = cellIndexMap[(i_, j_)]
            rho_list_blockOrder.append(rho_list_rowMajor[rowIdx])
            p_list_blockOrder.append(p_list_rowMajor[rowIdx])
            Ux_list_blockOrder.append(Ux_list_rowMajor[rowIdx])
            Uy_list_blockOrder.append(Uy_list_rowMajor[rowIdx])
            T_list_blockOrder.append(T_list_rowMajor[rowIdx])

# Now *these* lists (rho_list_blockOrder, etc.) match the
# exact block-by-block indexing that OpenFOAM uses internally.
Ntot = len(rho_list_blockOrder)
print("Block-ordered list length =", Ntot)

# -----------------------------
# STEP D: Write them out
# -----------------------------
def write_scalar_field(filename, fieldName, fieldData, dimensions):
    with open(filename, 'w') as f:
        f.write(
            "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class volScalarField;\n"
            f"    object {fieldName};\n}}\n\n"
        )
        f.write(f"dimensions      {dimensions};\n")
        f.write("internalField   nonuniform List<scalar>\n")
        f.write(f"{len(fieldData)}\n(\n")
        for val in fieldData:
            f.write(f"{val}\n")
        f.write(")\n;\n\nboundaryField\n{\n")
        f.write("    left\n    {\n        type            cyclic;\n    }\n")
        f.write("    right\n    {\n        type            cyclic;\n    }\n")
        f.write("    bottom\n    {\n        type            cyclic;\n    }\n")
        f.write("    top\n    {\n        type            cyclic;\n    }\n")
        f.write("    hole\n    {\n        type            zeroGradient;\n    }\n")
        f.write("    frontAndBack\n    {\n        type            empty;\n    }\n}\n")

def write_vector_field(filename, fieldName, Ux, Uy, dimensions):
    with open(filename, 'w') as f:
        f.write(
            "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class volVectorField;\n"
            f"    object {fieldName};\n}}\n\n"
        )
        f.write(f"dimensions      {dimensions};\n")
        f.write("internalField   nonuniform List<vector>\n")
        f.write(f"{len(Ux)}\n(\n")
        for (ux, uy) in zip(Ux, Uy):
            f.write(f"({ux} {uy} 0)\n")
        f.write(")\n;\n\nboundaryField\n{\n")
        f.write("    left\n    {\n        type            cyclic;\n    }\n")
        f.write("    right\n    {\n        type            cyclic;\n    }\n")
        f.write("    bottom\n    {\n        type            cyclic;\n    }\n")
        f.write("    top\n    {\n        type            cyclic;\n    }\n")
        f.write("    hole\n    {\n        type            fixedValue;\n        value uniform (0 0 0);\n    }\n")
        f.write("    frontAndBack\n    {\n        type            empty;\n    }\n}\n")

# Finally use the blockOrdered data
write_scalar_field("0/rho", "rho", rho_list_blockOrder, "[1 -3 0 0 0 0 0]")
write_scalar_field("0/p",   "p",   p_list_blockOrder,   "[1 -1 -2 0 0 0 0]")
write_scalar_field("0/T",   "T",   T_list_blockOrder,   "[0 0 0 1 0 0 0]")
write_vector_field("0/U",   "U",   Ux_list_blockOrder,  Uy_list_blockOrder, "[0 1 -1 0 0 0 0]")

print("Done writing block-ordered fields.")

