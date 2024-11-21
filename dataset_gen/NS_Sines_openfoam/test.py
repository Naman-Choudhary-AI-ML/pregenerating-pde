import numpy as np
import matplotlib.pyplot as plt

# Load a single trajectory file
trajectory_data = np.load("./NS_Sines_dataset/trajectory_0000.npy")  # Replace with your file
timesteps, num_points, channels = trajectory_data.shape

# Grid dimensions
nx, ny = 128, 128  # Replace with your grid dimensions

# Reshape trajectory data
velocity_data = trajectory_data.reshape((timesteps, nx, ny, channels))

# Visualize velocity field at a specific timestep
timestep_to_plot = 0  # Change this to visualize other timesteps
U = velocity_data[timestep_to_plot, :, :, 0]  # Horizontal velocity (Ux)
V = velocity_data[timestep_to_plot, :, :, 1]  # Vertical velocity (Uy)

# Plot horizontal velocity (Ux)
plt.figure(figsize=(5, 4))
plt.title(f"Horizontal Velocity (Ux) at Timestep {timestep_to_plot}")
plt.imshow(U, origin="lower", extent=[0, 1, 0, 1], cmap="gist_ncar")
plt.colorbar(label="Ux")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
plt.savefig('output.png')

# Plot vertical velocity (Uy)
plt.figure(figsize=(10, 8))
plt.title(f"Vertical Velocity (Uy) at Timestep {timestep_to_plot}")
plt.imshow(V, origin="lower", extent=[0, 1, 0, 1], cmap="viridis")
plt.colorbar(label="Uy")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
plt.savefig()
