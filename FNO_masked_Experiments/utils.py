import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FixedLocator

def plot_mesh_centers(input_path, output_path, precision=6):
    """
    Process the C file to find the 64 missing cell centers in the hole and append them to the end.

    Args:
        input_path (str): Path to the input C file.
        output_path (str): Path to save the new C file with hole centers.
        precision (int): Number of decimal places to round the coordinates (default is 6).
    """
    with open(input_path, 'r') as file:
        lines = file.readlines()
    num_points = None
    start_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("internalField"):
            num_points = int(lines[i + 1].strip())
            start_index = i + 3
            break
    coords = []
    for line in lines[start_index:start_index + num_points]:
        x, y, z = map(float, line.strip("()\n").split())
        coords.append((x, y, z))
    coords = np.array(coords)

    # Round x and y values to the specified precision
    coords[:, 0] = np.round(coords[:, 0], precision)
    coords[:, 1] = np.round(coords[:, 1], precision)

    # Extract unique x and y values and count occurrences
    x_unique, x_counts = np.unique(coords[:, 0], return_counts=True)
    y_unique, y_counts = np.unique(coords[:, 1], return_counts=True)
    # print(x_unique, x_counts, y_unique, y_counts)

    # Expected count for a regular grid
    expected_count = len(y_unique)
    # print(expected_count)

    # Identify missing x and y values
    missing_x = x_unique[x_counts < expected_count]
    missing_y = y_unique[y_counts < expected_count]

    # Sanity check: Ensure we have 8 missing x and 8 missing y values
    assert len(missing_x) == 8, f"Unexpected missing x count: {len(missing_x)}"
    assert len(missing_y) == 8, f"Unexpected missing y count: {len(missing_y)}"

    # Generate hole centers
    z_value = coords[0, 2]  # All z-values are the same
    hole_centers = [(x, y, z_value) for x in missing_x for y in missing_y]

    # Append hole centers to the original list
    all_coords = list(coords) + hole_centers

    # Write the updated C file
    with open(output_path, 'w') as file:
        # Update the count
        file.write(f"{len(all_coords)}\n(\n")

        # Write the original coordinates first
        for coord in coords:
            file.write(f"({coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f})\n")

        # Write the hole centers at the end
        for coord in hole_centers:
            file.write(f"({coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f})\n")

        # Close the list
        file.write(")\n")
    # # Optional: Visualize the updated mesh
    # plt.figure(figsize=(10, 10))
    # all_coords = np.array(all_coords)
    # plt.scatter(all_coords[:, 0], all_coords[:, 1], s=2, c="blue", label="Cell Centers")
    # plt.title("Updated Mesh with Hole Centers")
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.axis("equal")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

# Load the updated C file
def load_coordinates(C_file_path):
    """
    Load x and y coordinates from the updated C file.

    Args:
        C_file_path (str): Path to the updated C file.

    Returns:
        tuple: (x_coords, y_coords) as 2D arrays reshaped to (128, 128).
    """
    with open(C_file_path, "r") as file:
        lines = file.readlines()

    # Extract the internalField coordinates
    start_index = 2
    num_points = int(lines[0].strip())

    coords = []
    for line in lines[start_index : start_index + num_points]:
        x, y, _ = map(float, line.strip("()\n").split())
        coords.append((x, y))

    coords = np.array(coords)
    x_coords = coords[:, 0].reshape(128, 128)
    y_coords = coords[:, 1].reshape(128, 128)

    return x_coords, y_coords

def masked_mse_loss(outputs, targets, mask):
    """
    Computes MSE loss excluding the hole regions based on the mask.

    Args:
        outputs: Predicted values (batch_size, x, y, 2).
        targets: Ground truth values (batch_size, x, y, 2).
        mask: Binary mask (batch_size, x, y), where 1 = valid, 0 = hole.

    Returns:
        Masked MSE loss.
    """
    # Apply the mask to outputs and targets
    mask = mask.unsqueeze(-1)  # Shape: (batch_size, x, y, 1), to match outputs/targets
    valid_outputs = outputs * mask
    valid_targets = targets * mask

    # Calculate MSE only for valid regions
    mse_loss = ((valid_outputs - valid_targets) ** 2).sum()  # Sum over all dimensions

    # Normalize by the number of valid cells
    num_valid_cells = mask.sum()  # Total number of valid (non-hole) cells
    return mse_loss / num_valid_cells

# def plot_predictions_with_centers(outputs, targets, mask, C_file_path, output_folder, epoch):
#     """
#     Plot predictions and ground truth using irregular grid coordinates from C_updated.
#     This version avoids NaN handling by directly using mask-based indexing.

#     Args:
#         outputs: Predicted values (batchsize, 128, 128, 2).
#         targets: Ground truth values (batchsize, 128, 128, 2).
#         mask: Binary mask (batchsize, 128, 128), where 1 = valid, 0 = hole.
#         C_file_path: Path to the updated C file containing grid coordinates.
#         output_folder: Folder to save plots.
#         epoch: Current epoch number.
#     """
#     # Load the updated C file
#     with open(C_file_path, "r") as file:
#         lines = file.readlines()

#     start_index = 2
#     num_points = int(lines[0].strip())

#     coords = []
#     for line in lines[start_index : start_index + num_points]:
#         x, y, _ = map(float, line.strip("()\n").split())
#         coords.append((x, y))
#     coords = np.array(coords)  # Shape: (16384, 2)

#     # Extract predictions, ground truth, and mask
#     outputs = outputs[0].detach().cpu().numpy()  # Take the first example from the batch
#     targets = targets[0].detach().cpu().numpy()
#     mask = mask[0].detach().cpu().numpy()

#     # Filter valid points based on the mask
#     valid_indices = mask.flatten() == 1  # Boolean array of valid points
#     valid_coords = coords[valid_indices]  # Filtered coordinates for valid points
#     valid_truth = targets.reshape(-1, 2)[valid_indices]  # Filtered ground truth velocities
#     valid_pred = outputs.reshape(-1, 2)[valid_indices]  # Filtered predicted velocities

#     # Prepare figure
#     fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # 1x2 grid for ground truth and prediction
#     channels = ["Horizontal Velocity (u)", "Vertical Velocity (v)"]
#     cmap = "gist_ncar"

#     for i, (data, title) in enumerate(zip([valid_truth, valid_pred], ["Ground Truth", "Prediction"])):
#         # Flatten the velocities for plotting
#         velocity_u = data[:, 0]  # Horizontal velocity (u)
#         velocity_v = data[:, 1]  # Vertical velocity (v)

#         # Plot horizontal velocity
#         sc = ax[i].scatter(valid_coords[:, 0], valid_coords[:, 1], c=velocity_u, cmap=cmap)
#         ax[i].set_title(f"{title} - Horizontal Velocity (u)")
#         ax[i].axis("off")
#         fig.colorbar(sc, ax=ax[i], orientation="vertical", fraction=0.046, pad=0.04)

#     # Save and log the plot
#     plt.tight_layout()
#     plot_path = os.path.join(output_folder, f"epoch_{epoch}.png")
#     plt.savefig(plot_path)
#     plt.close(fig)


def plot_predictions_with_centers(outputs, targets, mask, C_file_path, output_folder, epoch):
    """
    Plot predictions and ground truth excluding hole centers using the mask.

    Args:
        outputs: Predicted values (batchsize, 128, 128, 2).
        targets: Ground truth values (batchsize, 128, 128, 2).
        mask: Binary mask (batchsize, 128, 128), where 1 = valid, 0 = hole.
        C_file_path: Path to the updated C file containing grid coordinates.
        output_folder: Folder to save plots.
        epoch: Current epoch number.
    """
    with open(C_file_path, 'r') as file:
        lines = file.readlines()

    start_index = 2
    num_points = int(lines[0].strip())

    coords = []
    for line in lines[start_index : start_index + num_points]:
        x, y, _ = map(float, line.strip("()\n").split())
        coords.append((x, y))
    coords = np.array(coords) #shape: 16384, 2
    coords = np.round(coords, 6)

    outputs = outputs[0].detach().cpu().numpy()
    targets = targets[0].detach().cpu().numpy()
    mask = mask[0].detach().cpu().numpy()

    mask_flat = mask.flatten() #shape (16384,)
    valid_indices = mask_flat == 1
    hole_indices = np.where(mask_flat == 0)[0]
    hole_coords = coords[hole_indices]
    # print("Number of hole points:", len(hole_coords))
    # print("Hole coordinates:", hole_coords)
    # print("valid indices", valid_indices[valid_indices==False])
    # print("Valid indices", valid_indices.shape)
    coords_filtered = coords[valid_indices]
    # print("Coords filtered", coords_filtered.shape)
    outputs_flat = outputs.reshape(-1, 2)[valid_indices]
    targets_flat = targets.reshape(-1, 2)[valid_indices]
    # print("Output shate", outputs_flat.shape)
    fig, ax = plt.subplots(2, 2, figsize=(14, 10)) #2x2 grid for both channels
    channels = ["Horizontal Velocity (u)", "Vertical Velocity (v)"]
    cmap = "gist_ncar"

    for ch in range(len(channels)):
        truth = targets_flat[:, ch]
        pred = outputs_flat[:, ch]

        vmin = min(truth.min(), pred.min())
        vmax = max(truth.max(), pred.max())
        # Define the number of ticks you want (must be odd for symmetry, e.g., 5)
        num_ticks = 5  

        # Determine the maximum absolute value of vmin and vmax
        max_abs = max(abs(vmin), abs(vmax))

        # Create symmetric ticks around 0
        symmetric_ticks = np.linspace(-max_abs, max_abs, num_ticks)

        #plot GT
        sc1 = ax[ch, 0].scatter(coords_filtered[:, 0], coords_filtered[:, 1], c=truth, cmap=cmap, vmin = vmin, vmax=vmax, s=10)
        ax[ch, 0].set_title(f"Ground Truth - {channels[ch]}")
        ax[ch, 0].axis("off")
        cb1 = fig.colorbar(sc1, ax=ax[ch, 0], orientation = "vertical", fraction=0.046, pad=0.01, shrink=0.95)
        cb1.formatter = FormatStrFormatter('%.1f')  # Format labels with 1 decimal point
        cb1.locator = FixedLocator(symmetric_ticks)
        cb1.update_ticks()  # Update the color bar with the new format

        #plot pred
        sc2 = ax[ch, 1].scatter(coords_filtered[:, 0], coords_filtered[:, 1], c=pred, cmap=cmap, vmin=vmin, vmax=vmax, s=10)
        ax[ch, 1].set_title(f"Prediction - {channels[ch]}")
        ax[ch, 1].axis("off")
        cb2 = fig.colorbar(sc2, ax=ax[ch, 1], orientation="vertical", fraction=0.046, pad=0.01, shrink=0.95)
        cb2.formatter = FormatStrFormatter('%.1f')
        cb2.locator = FixedLocator(symmetric_ticks)
        cb2.update_ticks()

    
    plt.tight_layout()
    plot_path=os.path.join(output_folder, f"epoch_{epoch}_final.png")
    plt.savefig(plot_path)
    plt.close(fig)






# def _plot_predictions_with_centers(outputs, targets, C_file_path, output_folder, epoch):
#     """
#     Plot predictions and ground truth using all centers, including hole centers.

#     Args:
#         outputs: Predicted values (batchsize, 128, 128, 2).
#         targets: Ground truth values (batchsize, 128, 128, 2).
#         C_file_path: Path to the updated C file containing grid coordinates.
#         output_folder: Folder to save plots.
#         epoch: Current epoch number.
#     """
#     # Load the updated C file
#     with open(C_file_path, "r") as file:
#         lines = file.readlines()

#     start_index = 2
#     num_points = int(lines[0].strip())

#     coords = []
#     for line in lines[start_index : start_index + num_points]:
#         x, y, _ = map(float, line.strip("()\n").split())
#         coords.append((x, y))
#     coords = np.array(coords)  # Shape: (16384, 2)
#     coords = np.round(coords, 6)

#     # Extract predictions and ground truth
#     outputs = outputs[0].detach().cpu().numpy()  # Take the first example from the batch
#     targets = targets[0].detach().cpu().numpy()

#     # Reshape outputs and targets to align with the coordinates
#     outputs_flat = outputs.reshape(-1, 2)  # Shape: (16384, 2)
#     targets_flat = targets.reshape(-1, 2)  # Shape: (16384, 2)

#     # Prepare figure
#     fig, ax = plt.subplots(2, 2, figsize=(14, 10))  # 2x2 grid for horizontal and vertical velocity
#     channels = ["Horizontal Velocity (u)", "Vertical Velocity (v)"]
#     cmap = "gist_ncar"

#     for ch in range(2):  # Loop over horizontal and vertical velocity channels
#         vmin = min(targets_flat[:, ch].min(), outputs_flat[:, ch].min())
#         vmax = max(targets_flat[:, ch].max(), outputs_flat[:, ch].max())

#         # Plot ground truth
#         sc1 = ax[ch, 0].scatter(coords[:, 0], coords[:, 1], c=targets_flat[:, ch], cmap=cmap, vmin=-0.8, vmax=0.8, s=10)
#         ax[ch, 0].set_title(f"Ground Truth - {channels[ch]}")
#         ax[ch, 0].axis("off")
#         fig.colorbar(sc1, ax=ax[ch, 0], orientation="vertical", fraction=0.046, pad=0.04)

#         # Plot predictions
#         sc2 = ax[ch, 1].scatter(coords[:, 0], coords[:, 1], c=outputs_flat[:, ch], cmap=cmap, vmin=-0.8, vmax=0.8)
#         ax[ch, 1].set_title(f"Prediction - {channels[ch]}")
#         ax[ch, 1].axis("off")
#         fig.colorbar(sc2, ax=ax[ch, 1], orientation="vertical", fraction=0.046, pad=0.04)

#     # Save and log the plot
#     plt.tight_layout()
#     plot_path = os.path.join(output_folder, f"epoch_{epoch}_all.png")
#     plt.savefig(plot_path)
#     plt.close(fig)



# # def plot_predictions_with_centers(outputs, targets, mask, C_file_path, output_folder, epoch):
# #     """
# #     Plot predictions and ground truth using irregular grid coordinates from C_updated.

# #     Args:
# #         outputs: Predicted values (batchsize, 128, 128, 2).
# #         targets: Ground truth values (batchsize, 128, 128, 2).
# #         mask: Binary mask (batchsize, 128, 128), where 1 = valid, 0 = hole.
# #         C_file_path: Path to the updated C file containing grid coordinates.
# #         output_folder: Folder to save plots.
# #         epoch: Current epoch number.
# #     """
# #     # Load the updated C file
# #     with open(C_file_path, "r") as file:
# #         lines = file.readlines()

# #     start_index = 2
# #     num_points = int(lines[0].strip())

# #     coords = []
# #     for line in lines[start_index : start_index + num_points]:
# #         x, y, _ = map(float, line.strip("()\n").split())
# #         coords.append((x, y))
# #     coords = np.array(coords)  # Shape: (16384, 2)

# #     # Extract predictions, ground truth, and mask
# #     outputs = outputs[0].detach().cpu().numpy()  # Take the first example from the batch
# #     targets = targets[0].detach().cpu().numpy()
# #     mask = mask[0].detach().cpu().numpy()

# #     # Split coordinates into x and y
# #     x_coords = coords[:, 0]
# #     y_coords = coords[:, 1]

# #     # Prepare figure
# #     fig, ax = plt.subplots(2, 2, figsize=(14, 10))  # 2x2 grid for horizontal and vertical velocity
# #     channels = ["Horizontal Velocity (u)", "Vertical Velocity (v)"]
# #     cmap = "gist_ncar"

# #     for ch in range(2):  # Loop over horizontal and vertical velocity channels
# #         truth = np.where(mask == 1, targets[:, :, ch], np.nan)  # Mask invalid regions in ground truth
# #         pred = np.where(mask == 1, outputs[:, :, ch], np.nan)  # Mask invalid regions in predictions

# #         vmin = np.nanmin([truth.min(), pred.min()])
# #         vmax = np.nanmax([truth.max(), pred.max()])

# #         # Plot ground truth
# #         sc1 = ax[ch, 0].scatter(x_coords, y_coords, c=truth.flatten(), cmap=cmap, vmin=vmin, vmax=vmax)
# #         ax[ch, 0].set_title(f"Ground Truth - {channels[ch]}")
# #         ax[ch, 0].axis("off")
# #         fig.colorbar(sc1, ax=ax[ch, 0], orientation="vertical", fraction=0.046, pad=0.04)

# #         # Plot prediction
# #         sc2 = ax[ch, 1].scatter(x_coords, y_coords, c=pred.flatten(), cmap=cmap, vmin=vmin, vmax=vmax)
# #         ax[ch, 1].set_title(f"Prediction - {channels[ch]}")
# #         ax[ch, 1].axis("off")
# #         fig.colorbar(sc2, ax=ax[ch, 1], orientation="vertical", fraction=0.046, pad=0.04)

# #     # Save and log the plot
# #     plt.tight_layout()
# #     plot_path = os.path.join(output_folder, f"epoch_{epoch}.png")
# #     plt.savefig(plot_path)
# #     plt.close(fig)
