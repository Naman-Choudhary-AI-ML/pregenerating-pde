import os
import numpy as np
import scipy.fftpack as fft

# ----------------------------
# User-defined parameters
# ----------------------------
Nx = 128
Ny = 128
Lx = 1.0
Ly = 1.0

p = 100  # Number of Gaussian bumps

# Physical constants (example)
R = 287
rho = 1.0
P = 2.5
T = P/(R*rho)  # Perfect gas: P = rho R T

# 8 blocks, skipping i=60..67, j=60..67 (for example)
blocks_ij = [
    (range(0, 60),   range(0, 60)),    
    (range(60, 68),  range(0, 60)),    
    (range(68, 128), range(0, 60)),    
    (range(0, 60),   range(60, 68)),   
    (range(68, 128), range(60, 68)),   
    (range(0, 60),   range(68, 128)),  
    (range(60, 68),  range(68, 128)),  
    (range(68, 128), range(68, 128)),
]

# ----------------------------
# Generate coordinates
# ----------------------------
x = np.linspace(0, Lx, Nx, endpoint=False)
y = np.linspace(0, Ly, Ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')

# ----------------------------
# Sample initial vorticity
# ----------------------------
# \omega_0(x,y) = sum_{i=1}^p (alpha_i/sigma_i) exp(-((x - x_i)^2+(y - y_i)^2)/(2 sigma_i^2))

alpha = np.random.uniform(-1, 1, p)
sigma = np.random.uniform(0.01, 0.1, p)
x_c = np.random.uniform(0, 1, p)
y_c = np.random.uniform(0, 1, p)

omega = np.zeros((Nx, Ny))
for i in range(p):
    dx = X - x_c[i]
    dy = Y - y_c[i]
    r2 = dx**2 + dy**2
    omega += (alpha[i]/sigma[i]) * np.exp(-r2/(2*sigma[i]**2))

# ----------------------------
# Compute velocity from vorticity
# ----------------------------
# We use the streamfunction approach:
# In 2D, omega = ∂v/∂x - ∂u/∂y = ∇²ψ,
# where u = ∂ψ/∂y and v = -∂ψ/∂x.
#
# With periodic BCs, we can invert ∇²ψ = ω in Fourier space:
# ψ_k = -ω_k / (kx² + ky²), except k=0 mode handled separately.
#
# We'll use FFT (kx, ky from FFT frequencies).
kx = fft.fftfreq(Nx, d=Lx/Nx)*2*np.pi
ky = fft.fftfreq(Ny, d=Ly/Ny)*2*np.pi
kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')

omega_hat = fft.fft2(omega)
# Laplacian in Fourier space: k^2 = kx^2 + ky^2
k2 = kx_grid**2 + ky_grid**2
# Avoid division by zero at k=0 (set psi_hat=0 for that mode)
psi_hat = np.zeros_like(omega_hat, dtype=complex)
mask = (k2 != 0)
psi_hat[mask] = -omega_hat[mask]/k2[mask]

psi = np.real(fft.ifft2(psi_hat))

# u = dψ/dy, v = -dψ/dx
# Compute derivatives in Fourier space: 
# d/dx -> i*kx, d/dy -> i*ky
psi_hat = fft.fft2(psi)
u_hat = (1j*ky_grid)*psi_hat
v_hat = -(1j*kx_grid)*psi_hat

u = np.real(fft.ifft2(u_hat))
v = np.real(fft.ifft2(v_hat))

rowMajorIndex = 0
cellIndexMap = {}  # (i,j)->rowIndex

u_list_rowMajor = []
v_list_rowMajor = []
p_list_rowMajor = []
T_list_rowMajor = []

for j in range(Ny):
    for i in range(Nx):
        # skip the hole region
        if (60 <= i <= 67) and (60 <= j <= 67):
            continue
        
        # store in rowMajor
        u_list_rowMajor.append(u[i,j])
        v_list_rowMajor.append(v[i,j])
        p_list_rowMajor.append(p)
        T_list_rowMajor.append(T)
        
        cellIndexMap[(i,j)] = rowMajorIndex
        rowMajorIndex += 1

print("Row-major total valid cells =", rowMajorIndex)

u_list_blockOrder = []
v_list_blockOrder = []
p_list_blockOrder = []
T_list_blockOrder = []

for (iRange, jRange) in blocks_ij:
    for j_ in jRange:
        for i_ in iRange:
            # skip hole if needed
            if (i_, j_) not in cellIndexMap:
                continue
            # retrieve row-major index
            rmIdx = cellIndexMap[(i_, j_)]
            u_list_blockOrder.append(u_list_rowMajor[rmIdx])
            v_list_blockOrder.append(v_list_rowMajor[rmIdx])
            p_list_blockOrder.append(p_list_rowMajor[rmIdx])
            T_list_blockOrder.append(T_list_rowMajor[rmIdx])

Ntot = len(u_list_blockOrder)
print("Block-ordered total valid cells =", Ntot)


# ----------------------------
# Write OpenFOAM field files
# ----------------------------
# Field file template
def write_scalar_field(filename, fieldName, dimensions, dataList):
    with open(filename, 'w') as f:
        f.write("FoamFile\n{\n    version 2.0;\n    format ascii;\n")
        f.write(f"    class volScalarField;\n    object {fieldName};\n}}\n\n")
        f.write(f"dimensions      {dimensions};\n\n")
        f.write("internalField   nonuniform List<scalar>\n")
        f.write(f"{len(dataList)}\n(\n")
        for val in dataList:
            f.write(f"{val}\n")
        f.write(")\n;\n\nboundaryField\n{\n")
        f.write("    left\n    {\n        type cyclic;\n    }\n")
        f.write("    right\n    {\n        type cyclic;\n    }\n")
        f.write("    bottom\n    {\n        type cyclic;\n    }\n")
        f.write("    top\n    {\n        type cyclic;\n    }\n")
        f.write("    hole\n    {\n        type zeroGradient;\n    }\n")
        f.write("    frontAndBack\n    {\n        type empty;\n    }\n}\n")

def write_vector_field(filename, fieldName, dimensions, UxList, UyList):
    with open(filename, 'w') as f:
        f.write("FoamFile\n{\n    version 2.0;\n    format ascii;\n")
        f.write(f"    class volVectorField;\n    object {fieldName};\n}}\n\n")
        f.write(f"dimensions      {dimensions};\n\n")
        f.write("internalField   nonuniform List<vector>\n")
        f.write(f"{len(UxList)}\n(\n")
        for (ux, uy) in zip(UxList, UyList):
            f.write(f"({ux} {uy} 0)\n")  # 2D => z=0
        f.write(")\n;\n\nboundaryField\n{\n")
        f.write("    left\n    {\n        type cyclic;\n    }\n")
        f.write("    right\n    {\n        type cyclic;\n    }\n")
        f.write("    bottom\n    {\n        type cyclic;\n    }\n")
        f.write("    top\n    {\n        type cyclic;\n    }\n")
        f.write("    hole\n    {\n        type            fixedValue;\n")
        f.write("        value           uniform (0 0 0);\n    }\n")
        f.write("    frontAndBack\n    {\n        type empty;\n    }\n}\n")

write_vector_field("0/U", "U", "[0 1 -1 0 0 0 0]",
                   u_list_blockOrder, v_list_blockOrder)
write_scalar_field("0/p", "p", "[1 -1 -2 0 0 0 0]",
                   p_list_blockOrder)
write_scalar_field("0/T", "T", "[0 0 0 1 0 0 0]",
                   T_list_blockOrder)

