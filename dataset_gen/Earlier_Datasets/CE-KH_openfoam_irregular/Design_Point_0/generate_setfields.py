import numpy as np
import logging
import os

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

# Define the hole indices (for i and j)
i_hole_min, i_hole_max = 60, 67
j_hole_min, j_hole_max = 60, 67

# Define the blocks around the hole
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

# -------------------------
# Generate Random Coefficients for Perturbations
# -------------------------
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
# Build row-major arrays skipping the hole
# and create a (i,j) -> rowIndex map
# -------------------------
rowMajorIndex = 0
cellIndexMap = {}

rho_rowMajor = []
p_rowMajor   = []
T_rowMajor   = []
vx_rowMajor  = []
vy_rowMajor  = []

for j in range(N):       # outer loop over y
    for i in range(N):   # inner loop over x
        # skip the hole
        if (i_hole_min <= i <= i_hole_max) and (j_hole_min <= j <= j_hole_max):
            continue
        
        rho_val = rho_field_full[i,j]
        p_val   = p_field_full[i,j]
        T_val   = T_field_full[i,j]
        vx_val  = vx_field_full[i,j]
        vy_val  = vy_field_full[i,j]
        
        rho_rowMajor.append(rho_val)
        p_rowMajor.append(p_val)
        T_rowMajor.append(T_val)
        vx_rowMajor.append(vx_val)
        vy_rowMajor.append(vy_val)
        
        cellIndexMap[(i,j)] = rowMajorIndex
        rowMajorIndex += 1

logging.info(f"Row-major enumeration: total valid cells = {rowMajorIndex}")

# -------------------------
# Now reorder by blocks to align with blockMesh ordering
# -------------------------
rho_blockOrder = []
p_blockOrder   = []
T_blockOrder   = []
vx_blockOrder  = []
vy_blockOrder  = []

for (iRange, jRange) in blocks_ij:
    for j_ in jRange:
        for i_ in iRange:
            if (i_, j_) not in cellIndexMap:
                # It's in the hole or out of domain, skip
                continue
            rmIdx = cellIndexMap[(i_, j_)]
            rho_blockOrder.append(rho_rowMajor[rmIdx])
            p_blockOrder.append(p_rowMajor[rmIdx])
            T_blockOrder.append(T_rowMajor[rmIdx])
            vx_blockOrder.append(vx_rowMajor[rmIdx])
            vy_blockOrder.append(vy_rowMajor[rmIdx])

Ntot = len(rho_blockOrder)
logging.info(f"Block-ordered enumeration: total valid cells = {Ntot}")

# -------------------------
# Write final fields in block order
# -------------------------
def write_scalar_field(filename, fieldName, dataList, dimensions):
    """
    Write a scalar field in nonuniform List<scalar>.
    dataList must be 1D, length = #validCells, in the correct order.
    """
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
        f.write(f"{len(dataList)}\n(\n")
        for val in dataList:
            f.write(f"    {val:.6f}\n")
        f.write(");\n\nboundaryField\n{\n")
        f.write("    left\n    {\n        type            cyclic;\n    }\n")
        f.write("    right\n    {\n        type            cyclic;\n    }\n")
        f.write("    bottom\n    {\n        type            cyclic;\n    }\n")
        f.write("    top\n    {\n        type            cyclic;\n    }\n")
        f.write("    hole\n    {\n        type            zeroGradient;\n    }\n")
        f.write("    frontAndBack\n    {\n        type            empty;\n    }\n}\n")

    logging.info(f"Wrote scalar field '{fieldName}' with {len(dataList)} entries to {filename}")

def write_vector_field(filename, fieldName, UxList, UyList, dimensions):
    """
    Write a vector field in nonuniform List<vector>.
    UxList, UyList must be 1D, same length, in the correct order.
    """
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
        f.write(f"{len(UxList)}\n(\n")
        for (ux, uy) in zip(UxList, UyList):
            f.write(f"    ({ux:.6f} {uy:.6f} 0.000000)\n")
        f.write(");\n\nboundaryField\n{\n")
        f.write("    left\n    {\n        type            cyclic;\n    }\n")
        f.write("    right\n    {\n        type            cyclic;\n    }\n")
        f.write("    bottom\n    {\n        type            cyclic;\n    }\n")
        f.write("    top\n    {\n        type            cyclic;\n    }\n")
        f.write("    hole\n    {\n        type            fixedValue;\n        value uniform (0 0 0);\n    }\n")
        f.write("    frontAndBack\n    {\n        type            empty;\n    }\n}\n")

    logging.info(f"Wrote vector field '{fieldName}' with {len(UxList)} entries to {filename}")

# Finally write
if not os.path.exists("0"):
    os.makedirs("0")

write_scalar_field("0/rho", "rho", rho_blockOrder, "[1 -3 0 0 0 0 0]")
write_scalar_field("0/p",   "p",   p_blockOrder,   "[1 -1 -2 0 0 0 0]")
write_scalar_field("0/T",   "T",   T_blockOrder,   "[0 0 0 1 0 0 0]")
write_vector_field("0/U",   "U",   vx_blockOrder,  vy_blockOrder, "[0 1 -1 0 0 0 0]")

logging.info("All fields have been written successfully in block order.")
print("Initial condition fields (rho, U, p, T) have been generated successfully in the '0' directory, skipping the hole and reordering by blocks.")
