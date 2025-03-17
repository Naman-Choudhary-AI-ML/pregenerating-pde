import torch
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def autoregressive_rollout(model, initial_input, target_sequence=None, teacher_forcing_ratio=0.0):
    """
    Autoregressively roll out model predictions.
    
    Args:
      model: the one-step prediction model.
      initial_input: Tensor of shape (batch, H, W, C) for the seed timestep.
      target_sequence: Tensor of shape (batch, T, H, W, teacher_channels) if available.
      teacher_forcing_ratio: probability to use ground truth instead of the model’s prediction.
    
    Returns:
      predictions: Tensor of shape (batch, T, H, W, teacher_channels).
    """
    predictions = []
    current_input = initial_input  # shape: (batch, H, W, C)
    T = target_sequence.shape[1] if target_sequence is not None else 10

    for t in range(T):
        # current_input already has the correct mask in the last channel, no flipping
        current_input = current_input.to(device)
        # Predict next timestep
        pred = model(current_input)  # shape: (batch, H, W, teacher_channels)
        predictions.append(pred)

        # Decide on teacher forcing
        if target_sequence is not None and random.random() < teacher_forcing_ratio:
            next_input = target_sequence[:, t, ...]  # shape: (batch, H, W, teacher_channels)
        else:
            next_input = pred
        
        # Update the channels 0..teacher_channels in current_input
        current_input = update_input_for_next_timestep(current_input, next_input)
    
    return torch.stack(predictions, dim=1)

def update_input_for_next_timestep(current_input, new_data):
    """
    Overwrite the first few channels of current_input with new_data.
    """
    updated_input = current_input.clone()
    updated_input[..., :new_data.shape[-1]] = new_data
    return updated_input

def masked_mse_loss_autoregressive(outputs, targets, mask):
    """
    Computes the mean squared error loss over valid elements for a sequence.
    
    Args:
        outputs: Tensor of shape (B, T, H, W, C) - predicted sequence.
        targets: Tensor of shape (B, T, H, W, C) - ground truth sequence.
        mask:    Tensor of shape (B, H, W) or (B, T, H, W) with 1 = valid, 0 = hole.
               If constant across timesteps, shape can be (B, H, W).
    
    Returns:
        Mean squared error averaged over all valid elements.
    """
    # If mask is (B, H, W), expand it to (B, T, H, W)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)  # now (B, 1, H, W)
    
    # Expand mask to match channels: final shape (B, T, H, W, 1)
    mask = mask.unsqueeze(-1)
    
    valid_outputs = outputs * mask
    valid_targets = targets * mask
    mse_loss = ((valid_outputs - valid_targets) ** 2).sum()
    
    num_valid_elements = mask.sum() * outputs.shape[-1]
    return mse_loss / num_valid_elements

def plot_autoregressive_sequence(
    prediction,
    truth,
    mask_array=None,
    sim_idx=0,
    epoch=0,
    save_dir="plots",
    output_dim=None,
    channel_names=None,
    tol=1e-5
):
    """
    Plots all timesteps for prediction and ground-truth arrays, displaying
    'output_dim' channels. Each timestep is shown in a pair of rows:
      - Top row for predicted values.
      - Bottom row for ground truth.
    Each row contains 'output_dim' columns (one per channel).
    
    For each channel and timestep, a common color scale is computed from both 
    prediction and ground truth, so that the colorbar is identical.
    
    Args:
        prediction (np.ndarray): shape (T, H, W, C_pred), predicted values.
        truth (np.ndarray): shape (T, H, W, C_truth), ground truth values.
        mask_array (np.ndarray, optional): A mask array for masking regions. If provided,
            it should be of shape (T, H, W) or broadcastable to that shape.
        sim_idx (int): Simulation index (for filename).
        epoch (int): Current epoch (for filename/logging).
        save_dir (str): Directory to save plots.
        output_dim (int, optional): Number of channels to plot. If None, uses prediction.shape[-1].
        channel_names (list, optional): List of channel names. If not provided or incomplete,
            generic names will be generated.
        tol (float): Masking threshold. Values with |val| < tol will appear white.
    """
    import os
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import wandb

    os.makedirs(save_dir, exist_ok=True)  # Ensure save directory exists

    T, H, W, n_channels = prediction.shape
    # Determine the number of channels to plot
    if output_dim is None:
        output_dim = n_channels

    # If channel_names is not provided or not long enough, generate default names
    if channel_names is None:
        channel_names = [f"Ch{ch}" for ch in range(output_dim)]
    elif len(channel_names) < output_dim:
        channel_names = channel_names + [f"Ch{ch}" for ch in range(len(channel_names), output_dim)]
    elif len(channel_names) > output_dim:
        channel_names = channel_names[:output_dim]

    n_plot_channels = output_dim

    # Create a custom colormap that shows near-zero (or masked) values as white.
    cmap = matplotlib.cm.get_cmap('gist_ncar').copy()
    cmap.set_bad(color='white')

    # Create figure with 2*T rows (for each timestep, one row for pred, one for GT)
    fig, axes = plt.subplots(nrows=2 * T, ncols=n_plot_channels,
                             figsize=(4 * n_plot_channels, 4 * T))

    for t in range(T):
        row_pred = 2 * t      # Row for prediction at timestep t
        row_truth = 2 * t + 1 # Row for ground truth at timestep t

        for ch in range(n_plot_channels):
            # Create masked arrays: use mask_array if provided,
            # otherwise mask values with absolute value < tol.
            if mask_array is not None:
                masked_pred = np.ma.masked_where(mask_array[t] < 0.5, prediction[t, :, :, ch])
                masked_truth = np.ma.masked_where(mask_array[t] < 0.5, truth[t, :, :, ch])
            else:
                masked_pred = np.ma.masked_where(np.abs(prediction[t, :, :, ch]) < tol, prediction[t, :, :, ch])
                masked_truth = np.ma.masked_where(np.abs(truth[t, :, :, ch]) < tol, truth[t, :, :, ch])
            
            # Compute a common color scale for this channel at timestep t
            combined_vals = np.ma.concatenate([masked_pred.compressed(), masked_truth.compressed()])
            if combined_vals.size > 0:
                vmin, vmax = combined_vals.min(), combined_vals.max()
            else:
                vmin, vmax = 0, 1

            # Plot prediction image
            im_pred = axes[row_pred, ch].imshow(masked_pred, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
            axes[row_pred, ch].set_title(f"Pred {channel_names[ch]} (t={t})")
            axes[row_pred, ch].axis("off")
            fig.colorbar(im_pred, ax=axes[row_pred, ch], fraction=0.046, pad=0.04)

            # Plot ground truth image
            im_truth = axes[row_truth, ch].imshow(masked_truth, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
            axes[row_truth, ch].set_title(f"GT {channel_names[ch]} (t={t})")
            axes[row_truth, ch].axis("off")
            fig.colorbar(im_truth, ax=axes[row_truth, ch], fraction=0.046, pad=0.04)

    plt.suptitle(f"Simulation {sim_idx} - All Timesteps (Epoch {epoch})")
    plt.tight_layout()

    # Save the plot locally
    save_path = os.path.join(save_dir, f"epoch_{epoch}_sim_{sim_idx}.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved plot: {save_path}")

    # Log the image to wandb (if using wandb)
    wandb.log({f"Prediction_vs_GT_Epoch_{epoch}": wandb.Image(save_path)})
