import torch
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import yaml
import torch.nn.functional as F
with open("config/config.yaml", "r") as f:
    base_config = yaml.safe_load(f)
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
        if base_config["model"]["model_type"] in ["FNO", "FFNO"]:
            pred = model(current_input)  # (B, H, W, C)
        else:
            # scOT style
            times = torch.full((current_input.shape[0], 1),t,device=device,dtype=current_input.dtype)
            scot_out = model(current_input, time=times)
            pred = scot_out.output  # also (B, H, W, C) by design  # shape: (batch, H, W, teacher_channels)
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
    model_name = base_config["model"]["model_type"]
    
    updated_input = current_input.clone()

    if model_name in ["FNO", "FFNO"]:
        updated_input[..., :new_data.shape[-1]] = new_data
    elif model_name == "scOT":
        updated_input[:, :new_data.shape[1], :, :] = new_data
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    return updated_input

import torch
import torch.nn.functional as F

def scot_autoregressive_loss(sequence_preds, 
                             target_sequences, 
                             p=1, 
                             channel_slice_list_normalized_loss=None, 
                             epsilon=1e-10):
    """
    sequence_preds, target_sequences: shape (B, T, H, W, C) (or any shape 
        as long as they match), with the last dimension being channels.
    p: Whether to use L1 loss (p=1) or MSE (p=2).
    channel_slice_list_normalized_loss: list of integers specifying the start 
        and end indices of channel slices, e.g. [0, 4, 6, 10]. If None, compute 
        the standard scOT loss (i.e., L1 or MSE over all channels).
    epsilon: small constant to avoid division-by-zero.

    Returns:
        A scalar tensor (the average loss).
    """
    # ------------------------------------------------------------------
    # 1) Choose the correct base loss criterion from p
    # ------------------------------------------------------------------
    if p == 1:
        criterion = F.l1_loss
    elif p == 2:
        criterion = F.mse_loss
    else:
        raise ValueError(f"Invalid value for p={p}. Must be 1 or 2.")
    
    # ------------------------------------------------------------------
    # 2) If channel-slice normalization is specified, do it slice-by-slice
    # ------------------------------------------------------------------
    if channel_slice_list_normalized_loss is not None:
        slice_losses = []
        
        # We loop from i=0 up to one less than the last index 
        # so we can do i and i+1 in pairs.
        for i in range(len(channel_slice_list_normalized_loss) - 1):
            start_idx = channel_slice_list_normalized_loss[i]
            end_idx   = channel_slice_list_normalized_loss[i+1]
            
            # Predicted slice and target slice along the last dimension (channels)
            pred_slice = sequence_preds[..., start_idx:end_idx]
            tgt_slice  = target_sequences[..., start_idx:end_idx]

            # Numerator: L1 or MSE of pred_slice vs. tgt_slice
            slice_loss = criterion(pred_slice, tgt_slice, reduction='mean')

            # Denominator: L1 or MSE of tgt_slice vs. 0, plus epsilon
            zero_slice = torch.zeros_like(tgt_slice)
            denom = criterion(tgt_slice, zero_slice, reduction='mean') + epsilon

            # Normalized slice loss
            slice_losses.append(slice_loss / denom)

        # Final slice-normalized loss is the mean of all slice losses
        loss = torch.stack(slice_losses).mean()

    else:
        # ------------------------------------------------------------------
        # 3) If no slices, compute standard L1 or MSE over all channels
        # ------------------------------------------------------------------
        loss = criterion(sequence_preds, target_sequences, reduction='mean')
    
    return loss


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

import numpy as np
import torch
from types import SimpleNamespace

def run_autoregressive_inference(model, data_loader, device, update_input_for_next_timestep):
    """
    Runs autoregressive rollout on the entire data_loader and accumulates predictions & labels.
    Returns (all_preds, all_labels) as numpy arrays of shape (N, C, H, W).
    N = total # of samples across the entire dataset * T (if flattening the time dimension).
    """
    model.eval()
    preds_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in data_loader:
            initial_inputs = batch["pixel_values"].to(device)  # (B, H, W, C)
            target_sequences = batch["labels"].to(device)      # (B, T, H, W, C)

            # Autoregressive rollout
            sequence_preds = []
            current_input = initial_inputs
            T = target_sequences.shape[1]

            for t in range(T):
                # scOT style
                times = torch.full(
                    (current_input.shape[0], 1), t,
                    device=device, dtype=current_input.dtype
                )
                scot_out = model(current_input, time=times)
                pred_t = scot_out.output  # (B, H, W, C)
                
                sequence_preds.append(pred_t)

                # teacher forcing or not, but for eval we often just do ground‐truth forcing or no forcing
                # e.g., next_data = target_sequences[:, t]
                # or if you want purely autoregressive: next_data = pred_t
                next_data = target_sequences[:, t]  # <--- often for eval, use ground-truth
                current_input = update_input_for_next_timestep(current_input, next_data)

            # Convert from list => (B, T, H, W, C)
            sequence_preds = torch.stack(sequence_preds, dim=1)

            # Reshape to the format your compute_metrics function wants, i.e. (N, C, H, W).
            # 
            # If `compute_metrics` expects shape (N, C, H, W), 
            #   and you want to combine B and T into one dimension => (B*T, C, H, W).
            sequence_preds_ = sequence_preds.permute(0, 1, 4, 2, 3).contiguous()  
            # shape => (B, T, C, H, W)
            B, T, C, H, W = sequence_preds_.shape
            sequence_preds_ = sequence_preds_.view(B*T, C, H, W)  # (B*T, C, H, W)

            # Same transformation for the ground truth:
            target_sequences_ = target_sequences.permute(0, 1, 4, 2, 3).contiguous()
            target_sequences_ = target_sequences_.view(B*T, C, H, W)

            # Convert to numpy for your compute_metrics
            preds_list.append(sequence_preds_.cpu().numpy())
            labels_list.append(target_sequences_.cpu().numpy())

    # Concatenate across batches
    all_preds  = np.concatenate(preds_list,  axis=0)  # shape => (N, C, H, W)
    all_labels = np.concatenate(labels_list, axis=0)  # shape => (N, C, H, W)
    
    return all_preds, all_labels

def evaluate_scOT(model, data_loader, device, compute_metrics, update_input_for_next_timestep):
    # 1) get predictions and labels from the entire dataset
    all_preds, all_labels = run_autoregressive_inference(
        model, data_loader, device, update_input_for_next_timestep
    )

    # 2) Build a simple "eval_preds" object for your compute_metrics function
    eval_preds = SimpleNamespace(
        predictions=all_preds,
        label_ids=all_labels
    )

    # 3) Compute the metrics
    metrics_dict = compute_metrics(
    eval_preds,
    model_type="scOT",
    output_dim=3,  # or whatever channels you truly have
    channel_slice_list=[0,1,2,3]
)
    
    return metrics_dict

def evaluate_fno_ffno(model, loader, device, compute_metrics, update_input_for_next_timestep):
    """
    Evaluation function specialized for FNO or FFNO.

    1. Collects predictions in channel-last format (B, T, H, W, C).
    2. Permutes to (B, T, C, H, W).
    3. Flattens (B*T) into one dimension => (N, C, H, W).
    4. Converts to NumPy and calls compute_metrics.
    """
    model.eval()
    import numpy as np
    import torch
    from types import SimpleNamespace

    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            # initial_inputs => (B, H, W, C)
            # target_sequences => (B, T, H, W, C)
            initial_inputs = batch["pixel_values"].to(device).float()
            target_sequences = batch["labels"].to(device).float()

            sequence_preds = []
            current_input = initial_inputs
            T = target_sequences.shape[1]

            for t in range(T):
                # Forward pass => returns (B, H, W, C) in channel-last form
                pred_t = model(current_input)
                sequence_preds.append(pred_t)
                
                # In eval, no teacher forcing
                current_input = update_input_for_next_timestep(current_input, pred_t)

            # Now shape => (B, T, H, W, C)
            sequence_preds = torch.stack(sequence_preds, dim=1)
            
            # Collect predictions & targets across batches
            all_preds.append(sequence_preds.cpu())
            all_targets.append(target_sequences.cpu())
    
    # 1) Concatenate across all batches => (N, T, H, W, C)
    all_preds = torch.cat(all_preds, dim=0)      # channel-last
    all_targets = torch.cat(all_targets, dim=0)  # channel-last

    # 2) Permute so the channel dimension is second => (N, T, C, H, W)
    #    This step reorders from channel-last to channel-first.
    all_preds = all_preds.permute(0, 1, 4, 2, 3).contiguous()
    all_targets = all_targets.permute(0, 1, 4, 2, 3).contiguous()

    # 3) Flatten N*T => shape (N*T, C, H, W)
    N, T, C, H, W = all_preds.shape
    all_preds = all_preds.view(N * T, C, H, W)
    all_targets = all_targets.view(N * T, C, H, W)

    # 4) Convert to NumPy so that your compute_metrics & lp_error (which use np.sum) won't fail
    all_preds_np = all_preds.numpy()
    all_targets_np = all_targets.numpy()

    # Create eval_preds object
    eval_preds_fno = SimpleNamespace(
        predictions=all_preds_np,    # shape => (N*T, C, H, W)
        label_ids=all_targets_np
    )

    # Now compute your metrics 
    metrics_fno = compute_metrics(
    eval_preds_fno,
    model_type="FNO",
    output_dim=3,
    channel_slice_list=[0,3]
)
    return metrics_fno


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

def plot_autoregressive_sequence_channel_first(
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
    Plots all timesteps for prediction and ground-truth arrays (channel-first),
    i.e., arrays of shape (T, C, H, W).

    If mask_array is None but truth has one more channel than prediction,
    we assume the last channel of truth is the mask.
    Where mask>0.5 => white in the plots.
    """
    import os
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import wandb

    os.makedirs(save_dir, exist_ok=True)

    # Debug prints: check shapes
    print(f"[DEBUG] prediction.shape = {prediction.shape} (expected T,C,H,W)")
    print(f"[DEBUG] truth.shape      = {truth.shape} (expected T,C,H,W)")
    if mask_array is not None:
        print(f"[DEBUG] mask_array.shape = {mask_array.shape}")
    else:
        print("[DEBUG] mask_array not provided at function call.")

    T_pred, C_pred, H, W = prediction.shape
    T_truth, C_truth, Ht, Wt = truth.shape

    # Basic shape consistency checks
    if not (T_pred == T_truth and H == Ht and W == Wt):
        print("[DEBUG] WARNING: prediction and truth spatiotemporal dims do not match exactly!")
    
    # -- If we do NOT have a mask_array but ground truth has an extra channel, treat that as mask
    if mask_array is None and (C_truth == C_pred + 1):
        print("[DEBUG] Attempting to auto-detect mask as last channel of truth.")
        # Last channel is mask
        mask_array = truth[:, -1, :, :]
        # Now remove the mask channel from truth so it doesn't get plotted
        truth = truth[:, :-1, :, :]
        C_truth = truth.shape[1]
        print(f"[DEBUG] Detected mask shape = {mask_array.shape}")
    else:
        # If not triggered, print why
        if mask_array is None:
            print("[DEBUG] No auto-mask: either truth doesn't have an extra channel, or user didn't provide it.")
        else:
            print("[DEBUG] mask_array was explicitly provided. Not auto-detecting mask from truth.")

    # Another debug check: do we still have the same # of channels in pred and truth?
    if C_pred != C_truth:
        print(f"[DEBUG] WARNING: # of channels differ! pred={C_pred}, truth={C_truth}")

    # Decide how many channels to plot
    if output_dim is None:
        output_dim = min(C_pred, C_truth)

    # Prepare or fix channel_names
    if channel_names is None:
        channel_names = [f"Ch{ch}" for ch in range(output_dim)]
    else:
        if len(channel_names) < output_dim:
            channel_names += [f"Ch{ch}" for ch in range(len(channel_names), output_dim)]
        else:
            channel_names = channel_names[:output_dim]

    # Print min/max for each channel for debugging
    for ch in range(output_dim):
        print(f"[DEBUG] pred channel {ch} min={prediction[..., ch].min():.3g}, max={prediction[..., ch].max():.3g}")
        print(f"[DEBUG] true channel {ch} min={truth[..., ch].min():.3g}, max={truth[..., ch].max():.3g}")

    # Colormap that shows masked as white
    cmap = matplotlib.cm.get_cmap("gist_ncar").copy()
    cmap.set_bad(color="white")

    fig, axes = plt.subplots(
        nrows=2 * T_pred,
        ncols=output_dim,
        figsize=(4 * output_dim, 4 * T_pred),
        squeeze=False
    )

    for t in range(T_pred):
        row_pred = 2 * t
        row_truth = 2 * t + 1

        for ch in range(output_dim):
            # 2D slices
            pred_slice = prediction[t, ch, :, :]
            truth_slice = truth[t, ch, :, :]

            if mask_array is not None:
                # Mask out where mask>0.5 => domain/hole is white
                masked_pred  = np.ma.masked_where(mask_array[t] > 0.5, pred_slice)
                masked_truth = np.ma.masked_where(mask_array[t] > 0.5, truth_slice)
            else:
                # fallback: mask near-zero
                masked_pred  = np.ma.masked_where(np.abs(pred_slice - 1.0)  < tol, pred_slice)
                masked_truth = np.ma.masked_where(np.abs(truth_slice - 1.0) < tol, truth_slice)

            # Common color scale
            combined_vals = np.ma.concatenate([masked_pred.compressed(), masked_truth.compressed()])
            if combined_vals.size > 0:
                vmin, vmax = combined_vals.min(), combined_vals.max()
            else:
                vmin, vmax = 0, 1

            im_pred = axes[row_pred, ch].imshow(masked_pred, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
            axes[row_pred, ch].set_title(f"Pred {channel_names[ch]} (t={t})")
            axes[row_pred, ch].axis("off")
            fig.colorbar(im_pred, ax=axes[row_pred, ch], fraction=0.046, pad=0.04)

            im_truth = axes[row_truth, ch].imshow(masked_truth, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
            axes[row_truth, ch].set_title(f"GT {channel_names[ch]} (t={t})")
            axes[row_truth, ch].axis("off")
            fig.colorbar(im_truth, ax=axes[row_truth, ch], fraction=0.046, pad=0.04)

    plt.suptitle(f"Simulation {sim_idx} - All Timesteps (Epoch {epoch})")
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"epoch_{epoch}_sim_{sim_idx}.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"[DEBUG] Saved plot (channel-first): {save_path}")

    import wandb
    wandb.log({f"Prediction_vs_GT_Epoch_{epoch}": wandb.Image(save_path)})



