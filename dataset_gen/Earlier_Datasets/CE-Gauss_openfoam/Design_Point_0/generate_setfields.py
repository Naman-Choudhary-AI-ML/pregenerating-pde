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

# ----------------------------
# Create output directory if needed
# ----------------------------
if not os.path.exists('0'):
    os.makedirs('0')

# ----------------------------
# Write OpenFOAM field files
# ----------------------------
# Field file template
def write_field(filename, fieldName, dimensions, fieldData, fieldType='scalar', boundaryType='cyclic'):
    """
    Generalized function to write OpenFOAM scalar or vector fields.
    """
    Nx, Ny = fieldData.shape[:2]
    Ntot = Nx * Ny  # Total number of cells
    
    with open(filename, 'w') as f:
        # Header
        f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
        f.write("  =========                 |\n")
        f.write("  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox\n")
        f.write("   \\    /   O peration     | Website:  https://openfoam.org\n")
        f.write("    \\  /    A nd           | Version:  8\n")
        f.write("     \\/     M anipulation  |\n")
        f.write("\\*---------------------------------------------------------------------------*/\n")
        f.write("FoamFile\n{\n")
        f.write("    version     2.0;\n")
        f.write("    format      ascii;\n")
        f.write(f"    class       vol{fieldType.capitalize()}Field;\n")
        f.write(f"    location    \"0\";\n")
        f.write(f"    object      {fieldName};\n")
        f.write("}\n\n")
        
        # Dimensions
        f.write(f"dimensions      {dimensions};\n\n")
        
        # Internal Field
        f.write("internalField   nonuniform List<")
        f.write("scalar" if fieldType == 'scalar' else "vector")
        f.write(">\n")
        f.write(f"{Ntot}\n(\n")
        
        if fieldType == 'scalar':
            # Write scalar values in row-major order
            data_flat = fieldData.flatten(order='C')
            for val in data_flat:
                f.write(f"    {val:.6f}\n")
        else:
            # Write vector values
            data_flat = fieldData.reshape(-1, fieldData.shape[-1])
            for row in data_flat:
                f.write(f"    ({row[0]:.6f} {row[1]:.6f} 0.000000)\n")  # Append zero for 2D
        f.write(");\n\n")
        
        # Boundary Field
        f.write("boundaryField\n")
        f.write("{\n")
        for patchName in ["left", "right", "bottom", "top", "frontAndBack"]:
            f.write(f"    {patchName}\n")
            f.write("    {\n")
            if patchName == "frontAndBack":
                f.write("        type            empty;\n")  # Correct for 2D cases
            else:
                f.write(f"        type            {boundaryType};\n")
            f.write("    }\n")
        f.write("}\n")

    print(f"{fieldType.capitalize()} field '{fieldName}' written to {filename}")

# Write U (vector field)
# U is (u,v) in 2D. OpenFOAM typically uses 3D vectors, so append a zero.
U_field = np.stack((u, v), axis=-1)  # shape (Nx, Ny, 2)
write_field('0/U', 'U', '[0 1 -1 0 0 0 0]', U_field, fieldType='vector')

# Write P (scalar field)
P_field = np.full((Nx, Ny), P)
write_field('0/P', 'P', '[0 2 -2 0 0 0 0]', P_field, fieldType='scalar')

# Write T (scalar field)
T_field = np.full((Nx, Ny), T)
write_field('0/T', 'T', '[0 0 0 1 0 0 0]', T_field, fieldType='scalar')

print("Initial fields written to 0/U, 0/P, and 0/T.")
