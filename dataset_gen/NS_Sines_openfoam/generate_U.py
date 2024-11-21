import numpy as np

# Mesh size
nx, ny = 128, 128
p = 10

# Random coefficients
alpha = np.random.uniform(-1, 1, (p, p))
beta = np.random.uniform(0, 2 * np.pi, (p, p))
gamma = np.random.uniform(0, 2 * np.pi, (p, p))

# Grid
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)

# Velocity field
u = np.zeros_like(X)
v = np.zeros_like(Y)
for i in range(p):
    for j in range(p):
        u += alpha[i, j] * np.sin(2 * np.pi * i * X + beta[i, j]) * np.sin(2 * np.pi * j * Y + gamma[i, j])
        v += alpha[i, j] * np.cos(2 * np.pi * i * X + beta[i, j]) * np.cos(2 * np.pi * j * Y + gamma[i, j])

# Write to U file
with open("0/U", "w") as f:
    f.write("FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       volVectorField;\n    object      U;\n}\n")
    f.write("dimensions      [0 1 -1 0 0 0 0];\n")
    f.write("internalField   nonuniform List<vector>\n")
    f.write(f"{nx * ny}\n(\n")
    for i in range(nx):
        for j in range(ny):
            f.write(f"({u[i, j]} {v[i, j]} 0.0)\n")
    f.write(");\n")
    f.write("boundaryField\n{\n    left\n    {\n        type cyclic;\n    }\n    right\n    {\n        type cyclic;\n    }\n    bottom\n    {\n        type cyclic;\n    }\n    top\n    {\n        type cyclic;\n    }\n    frontAndBack\n    {\n        type empty;\n    }\n}\n")
