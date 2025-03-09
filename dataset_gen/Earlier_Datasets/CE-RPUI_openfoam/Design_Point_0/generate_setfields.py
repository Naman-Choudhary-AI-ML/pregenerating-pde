import numpy as np
import os
import logging

# -------------------------------------------------------
# User parameters
# -------------------------------------------------------
p = 1  # number of subdomains in one direction (p+1 subdomains actually)
N = 128  # number of cells in x and y direction for the sample
R = 287  # gas constant for p = rho*R*T

# Domain ranges (assuming T^2 = [0,1]x[0,1])
xMin, xMax = 0.0, 1.0
yMin, yMax = 0.0, 1.0

# Define cell centers
x = np.linspace(xMin + 0.5*(xMax - xMin)/N, xMax - 0.5*(xMax - xMin)/N, N)
y = np.linspace(yMin + 0.5*(yMax - yMin)/N, yMax - 0.5*(yMax - yMin)/N, N)
X, Y = np.meshgrid(x, y, indexing='ij')

# -------------------------------------------------------
# Generate random coefficients for sigma_x and sigma_y
# -------------------------------------------------------
def fractional_part(val):
    # fractional part that respects the definition in the text
    # For x >= 0: fractional_part(x) = x - floor(x)
    # For x < 0: fractional_part(x) = x - floor(-|x|)
    # The given definition seems a bit unusual, but we can implement a consistent version:
    return val % 1.0

alpha_x = np.random.uniform(-0.01, 0.01, size=(p,p))
beta_x = np.random.uniform(0, 1, size=(p,p))
alpha_y = np.random.uniform(-0.01, 0.01, size=(p,p))
beta_y = np.random.uniform(0, 1, size=(p,p))

def sigma_x_fun(X, Y, p):
    # sigma_x(x,y) = sum_{i,j=1}^p alpha_{x,i,j} sin(2π(i+2p²)x + (j+2p²)y + β_{x,i,j})
    val = np.zeros_like(X)
    for i in range(p):
        for j in range(p):
            val += alpha_x[i,j]*np.sin(
    2*np.pi*((i+1) + 2*p**2)*X 
    + ((j+1) + 2*p**2)*Y 
    + beta_x[i,j]
)
    return val

def sigma_y_fun(X, Y, p):
    # sigma_y(x,y) = sum_{i,j=1}^p alpha_{y,i,j} sin(2π(i+2p²)x + (j+2p²)y + β_{y,i,j})
    val = np.zeros_like(X)
    for i in range(p):
        for j in range(p):
            val += alpha_x[i,j]*np.sin(
    2*np.pi*((i+1) + 2*p**2)*X 
    + ((j+1) + 2*p**2)*Y 
    + beta_x[i,j]
)
    return val

SIGMA_X = sigma_x_fun(X, Y, p)
SIGMA_Y = sigma_y_fun(X, Y, p)

# -------------------------------------------------------
# Define subdomains D_{i,j}
# Each subdomain interval in x: [x_min, x_max) = [i/(p+1), (i+1)/(p+1))
# and similarly for y
# -------------------------------------------------------
# We need to find, for each cell, which (i,j) subdomain it belongs to after perturbation.
# The condition is:
#   x_min <= {x + sigma_x(x,y) + 1} < x_max
#   y_min <= {y + sigma_y(x,y) + 1} < y_max

# Compute fractional parts
X_eff = fractional_part(X + SIGMA_X + 1.0)
Y_eff = fractional_part(Y + SIGMA_Y + 1.0)

# The domain is subdivided into (p+1)x(p+1) subdomains
num_sub = p+1
subdomain_indices = np.zeros((N,N,2), dtype=int)

# Compute the subdomain based on fractional coordinates
# For each cell, find which i,j satisfies:
# i/(p+1) <= X_eff < (i+1)/(p+1)
# j/(p+1) <= Y_eff < (j+1)/(p+1)

interval_edges = np.linspace(0,1,num_sub+1)
def find_interval(val, edges):
    # find the interval index i such that edges[i] <= val < edges[i+1]
    return np.searchsorted(edges, val, side='right') - 1

for ix in range(N):
    for iy in range(N):
        i_sub = find_interval(X_eff[ix,iy], interval_edges)
        j_sub = find_interval(Y_eff[ix,iy], interval_edges)
        subdomain_indices[ix,iy,0] = i_sub
        subdomain_indices[ix,iy,1] = j_sub

# -------------------------------------------------------
# Assign random initial conditions per subdomain
# We have (p+1)^2 subdomains, each with random (rho, vx, vy, p)
# rho ~ U[1,3], vx ~ U[-10,10], vy ~ U[-10,10], p ~ U[5,7]
# We'll generate these once and store them
rho_sub = np.random.uniform(1.0, 3.0, size=(num_sub,num_sub))
vx_sub = np.random.uniform(-10.0, 10.0, size=(num_sub,num_sub))
vy_sub = np.random.uniform(-10.0, 10.0, size=(num_sub,num_sub))
p_sub = np.random.uniform(5.0, 7.0, size=(num_sub,num_sub))

# Fill field arrays
rho_field = np.zeros((N,N))
vx_field = np.zeros((N,N))
vy_field = np.zeros((N,N))
p_field = np.zeros((N,N))
T_field = np.zeros((N,N))

for ix in range(N):
    for iy in range(N):
        i_sub = subdomain_indices[ix,iy,0]
        j_sub = subdomain_indices[ix,iy,1]
        rho_field[ix,iy] = rho_sub[i_sub,j_sub]
        vx_field[ix,iy]  = vx_sub[i_sub,j_sub]
        vy_field[ix,iy]  = vy_sub[i_sub,j_sub]
        p_field[ix,iy]   = p_sub[i_sub,j_sub]

# Compute T from p = rho * R * T
# T = p / (rho * R)
T_field = p_field / (rho_field * R)

# -------------------------------------------------------
# Write out the fields in a format OpenFOAM can read.
# Typically, OpenFOAM expects fields in a specific folder structure:
#   case/0/U
#   case/0/p
#   case/0/T
#
# We'll write ASCII format. You need to ensure that the dimensions match your mesh.
#
# Assume that you have already created an OpenFOAM case with a mesh of N x N cells
# in the x-y plane and at z=0 (2D). For a 2D case, OpenFOAM typically extrudes one layer in z.
# Make sure your blockMesh and decomposeParDict are consistent.
#
# We'll write a simple ASCII list of values. For a structured mesh, you might have to 
# adapt this to your mesh ordering or use a tool like `setFields` or `foamDictionary`.

output_dir = "0"  # Directly overwrite files in the '0' directory
if not os.path.exists(output_dir):
    logging.error("Output Directory not found")
    raise FileNotFoundError(f"Directory '{output_dir}' does not exist. Ensure you have created it.")

# -------------------------
# Function to Write Fields
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

logging.info("All fields have been written successfully.")
print("Initial condition files (rho, U, p, T) have been generated successfully in the '0' directory.")
