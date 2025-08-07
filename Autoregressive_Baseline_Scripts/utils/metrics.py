import numpy as np
import yaml
from types import SimpleNamespace

# Load config for output_dim
with open("config/config.yaml", "r") as f:
    base_config = yaml.safe_load(f)
output_dim = base_config["model"]["output_dim"]

# Based on output_dim, define channel labels (for logging)
if output_dim == 1:
    printable_channel_description = ["u"]
elif output_dim == 2:
    printable_channel_description = ["u", "v"]
elif output_dim == 3:
    printable_channel_description = ["u", "v", "p"]
else:
    # e.g., 4 => ["u", "v", "p", "d"]
    printable_channel_description = ["u", "v", "p", "d"][:output_dim]


def lp_error(preds: np.ndarray, targets: np.ndarray, p=1):
    """
    Standard L^p error across (N, C, H, W). Returns array of shape (N,),
    the error per sample. 
    """
    num_samples, num_channels, _, _ = preds.shape
    preds_resh = preds.reshape(num_samples, num_channels, -1)
    targets_resh = targets.reshape(num_samples, num_channels, -1)
    errors = np.sum(np.abs(preds_resh - targets_resh) ** p, axis=-1)  # shape (N, C)
    lp_per_sample = np.sum(errors, axis=-1) ** (1 / p)                # shape (N,)
    return lp_per_sample


def relative_lp_error(preds: np.ndarray, targets: np.ndarray, p=1, return_percent=True):
    """
    Relative L^p error, per sample, across all channels, H, W.
    Returns shape (N,).
    """
    num_samples, num_channels, _, _ = preds.shape
    preds_resh = preds.reshape(num_samples, num_channels, -1)
    targets_resh = targets.reshape(num_samples, num_channels, -1)

    diff = np.abs(preds_resh - targets_resh) ** p  # shape (N, C, H*W)
    errors = np.sum(diff, axis=-1)                 # shape (N, C)
    errors = np.sum(errors, axis=-1)               # shape (N,)

    denom = np.abs(targets_resh) ** p
    denom = np.sum(denom, axis=-1)                 # shape (N, C)
    denom = np.sum(denom, axis=-1)                 # shape (N,)
    denom = np.where(denom == 0, 1e-10, denom)

    rel = (errors / denom) ** (1 / p)
    if return_percent:
        rel *= 100
    return rel  # shape (N,)


def get_statistics(errors, metric_type):
    """
    errors: 1D array of per-sample errors, shape (N,).
    Returns a dictionary with median, mean, std, min, max.
    """
    median_error = np.median(errors, axis=0)
    mean_error = np.mean(errors, axis=0)
    std_error = np.std(errors, axis=0)
    min_error = np.min(errors, axis=0)
    max_error = np.max(errors, axis=0)
    return {
        f"median_relative_{metric_type}_error": median_error,
        f"mean_relative_{metric_type}_error": mean_error,
        f"std_relative_{metric_type}_error": std_error,
        f"min_relative_{metric_type}_error": min_error,
        f"max_relative_{metric_type}_error": max_error,
    }


def compute_metrics(
    eval_preds,
    model_type="scOT",
    output_dim=3,
    channel_slice_list=None
):
    """
    A unified metrics function that:
      - scOT: uses a channel_slice_list that slices each channel (e.g. [0,1,2,3]).
      - FNO/FFNO: can treat all channels at once, or define channel_slice_list=[0, output_dim].

    Expects:
      eval_preds.predictions -> np.array of shape (N, C, H, W)
      eval_preds.label_ids   -> np.array of shape (N, C, H, W)
    """

    preds = eval_preds.predictions  # shape (N, C, H, W)
    labels = eval_preds.label_ids   # shape (N, C, H, W)

    # If not given a slice list, define defaults
    if channel_slice_list is None:
        if model_type == "scOT":
            # For scOT, we typically slice each channel individually:
            # e.g. output_dim=3 => [0,1,2,3]
            channel_slice_list = list(range(output_dim + 1))
        else:
            # For FNO/FFNO, maybe treat all channels at once
            # e.g. output_dim=3 => [0,3]
            channel_slice_list = [0, output_dim]

    # We'll compute L1, L2, and L∞ for each slice in channel_slice_list.
    # E.g., if scOT => slices: [0:1], [1:2], [2:3]
    # If FNO => slices: [0:3]
    # Then we do aggregator logic at the end.

    # L1
    error_statistics_l1 = []
    for i in range(len(channel_slice_list) - 1):
        c_start = channel_slice_list[i]
        c_end   = channel_slice_list[i+1]
        # Relative L1
        rel_l1_per_sample = relative_lp_error(
            preds[:, c_start:c_end],
            labels[:, c_start:c_end],
            p=1,
            return_percent=True
        )
        stats_l1 = get_statistics(rel_l1_per_sample, "l1")
        error_statistics_l1.append(stats_l1)

    # L2
    error_statistics_l2 = []
    for i in range(len(channel_slice_list) - 1):
        c_start = channel_slice_list[i]
        c_end   = channel_slice_list[i+1]
        # Relative L2
        rel_l2_per_sample = relative_lp_error(
            preds[:, c_start:c_end],
            labels[:, c_start:c_end],
            p=2,
            return_percent=True
        )
        stats_l2 = get_statistics(rel_l2_per_sample, "l2")
        error_statistics_l2.append(stats_l2)

    # L∞ (max norm)
    error_statistics_linf = []
    for i in range(len(channel_slice_list) - 1):
        c_start = channel_slice_list[i]
        c_end   = channel_slice_list[i+1]
        diff = np.abs(preds[:, c_start:c_end] - labels[:, c_start:c_end])
        # shape => (N, sliceC, H, W)
        # We want the maximum over (channels in slice + spatial)
        # => shape (N,)
        linf_per_sample = diff.max(axis=(1,2,3))
        stats_linf = get_statistics(linf_per_sample, "linf")
        error_statistics_linf.append(stats_linf)

    # Now aggregate
    if output_dim == 1:
        # If there's truly only 1 channel total, there's only 1 slice anyway
        return {
            **error_statistics_l1[0],
            **error_statistics_l2[0],
            **error_statistics_linf[0]
        }
    else:
        # If multiple channels, your existing aggregator logic to combine them:
        # We can compute "mean over means" or other combos as you did in your snippet

        # For demonstration: let's do a "mean over means" for L1, L2, L∞:
        mean_over_means_l1 = np.mean(
            [stats["mean_relative_l1_error"] for stats in error_statistics_l1]
        )
        mean_over_means_l2 = np.mean(
            [stats["mean_relative_l2_error"] for stats in error_statistics_l2]
        )
        mean_over_means_linf = np.mean(
            [stats["mean_relative_linf_error"] for stats in error_statistics_linf]
        )
        mean_over_medians_l1 = np.mean(
            [stats["median_relative_l1_error"] for stats in error_statistics_l1]
        )
        mean_over_medians_l2 = np.mean(
            [stats["median_relative_l2_error"] for stats in error_statistics_l2]
        )
        mean_over_medians_linf = np.mean(
            [stats["median_relative_linf_error"] for stats in error_statistics_linf]
        )

        # Build the final dict
        error_statistics_ = {
            "mean_relative_l1_error": mean_over_means_l1,
            "mean_over_median_relative_l1_error": mean_over_medians_l1,
            "mean_relative_l2_error": mean_over_means_l2,
            "mean_over_median_relative_l2_error": mean_over_medians_l2,
            "mean_relative_linf_error": mean_over_means_linf,
            "mean_over_median_relative_linf_error": mean_over_medians_linf,
        }

        # Also store per-slice metrics with channel labels
        # e.g. "u/median_relative_l1_error", etc.
        for i, (stats_l1, stats_l2, stats_linf) in enumerate(
            zip(error_statistics_l1, error_statistics_l2, error_statistics_linf)
        ):
            # In scOT mode, each slice is a single channel
            # In FNO mode (single slice), i=0 => entire set of channels
            channel_name = printable_channel_description[i] if i < len(printable_channel_description) else f"ch{i}"
            for key, value in stats_l1.items():
                error_statistics_[f"{channel_name}/{key}"] = value
            for key, value in stats_l2.items():
                error_statistics_[f"{channel_name}/{key}"] = value
            for key, value in stats_linf.items():
                error_statistics_[f"{channel_name}/{key}"] = value

        return error_statistics_
