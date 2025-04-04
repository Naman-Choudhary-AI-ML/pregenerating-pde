import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import os
from utils.poseidon_dataloader import CEDataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_channel_layout(num_channels):
    """
    Infers how to treat each channel (physical, coordinates, mask) based on the
    total number of channels in the final data.

    Returns:
      physical_channels: list of channel indices for physical variables
      coord_channels: list of channel indices for x, y coordinates
      mask_channel: single integer for the mask channel
    """
    # Example logic:
    #   - 8-channel final layout => [0..4] = physical, [5..6] = coords, 7 = mask
    #   - 9-channel final layout => [0..5] = physical, [6..7] = coords, 8 = mask
    #   - fallback (7-channel)   => [0..3] = physical, [4..5] = coords, 6 = mask

    if num_channels == 8:
        physical_channels = [0, 1, 2]
        coord_channels = [5, 6]
        mask_channel = 7
    elif num_channels == 9:
        physical_channels = [0, 1, 2, 3]
        coord_channels = [6, 7]
        mask_channel = 8
    elif num_channels == 6:
        physical_channels = [0, 1, 2]
        coord_channels = [3, 4]
        mask_channel = [5]
    else:
        # fallback, e.g., 7-channel layout
        physical_channels = [0, 1, 2, 3]
        coord_channels = [4, 5]
        mask_channel = 6

    return physical_channels, coord_channels, mask_channel

# Normalization functions
def normalize_sequence(x, phys_stats, coord_stats):
    """
    Normalize a single sequence (timesteps, H, W, C).
    - Physical channels => (x - mean) / std
    - Coordinate channels => (x - min) / (max - min)
    - Mask remains unchanged
    """
    x_norm = x.clone().float()
    C = x_norm.shape[-1]

    # Figure out which channels are physical vs. coordinates vs. mask
    physical_channels, coord_channels, mask_channel = get_channel_layout(C)

    # --- Normalize physical channels ---
    for ch in physical_channels:
        mean, std = phys_stats[ch]
        x_norm[..., ch] = (x_norm[..., ch] - mean) / std

    # --- Scale coordinate channels to [0, 1] ---
    for ch in coord_channels:
        min_val, max_val = coord_stats[ch]
        rng = max_val - min_val if max_val != 0 else 1e-8
        x_norm[..., ch] = (x_norm[..., ch] - min_val) / (max_val - min_val + 1e-12)

    # Mask channel is left as-is
    return x_norm

def normalize_target_sequence(y, phys_stats):
    """
    Normalize the target sequence (timesteps, H, W, C).
    Typically, the target includes only physical channels.
    """
    y_norm = y.clone().float()
    C = y_norm.shape[-1]

    # If the target is purely physical, we can simply assume all channels are physical.
    # Or you can re-infer layout if your target includes coords or mask, but usually it does not.
    # Here, we just say "all channels in the target are physical."
    for ch in range(C):
        mean, std = phys_stats[ch]
        y_norm[..., ch] = (y_norm[..., ch] - mean) / std

    return y_norm

# Custom Dataset for Autoregressive Training
class SimulationDataset(Dataset):
    def __init__(
        self,
        npy_file,  # This can now be a file path (str) or an ndarray.
        phys_stats,
        coord_stats,
        teacher_channels=None,  # If None, automatically use all physical channels.
        transform_input=normalize_sequence,
        transform_target=normalize_target_sequence,
        debug=False
    ):
        """
        Args:
            npy_file: path to the .npy file or a pre-loaded simulation data array.
            phys_stats, coord_stats: normalization stats (dicts).
            teacher_channels: number of channels to use for the target. If None,
                              we use all physical channels.
            transform_input, transform_target: normalization callables.
            debug: if True, prints debug info.
        """
        self.debug = debug
        self.phys_stats = phys_stats
        self.coord_stats = coord_stats
        self.transform_input = transform_input
        self.transform_target = transform_target

        # Check if npy_file is already an array or a path.
        if isinstance(npy_file, (str, bytes, os.PathLike)):
            try:
                with open(npy_file, "rb") as f:
                    self.data = np.load(f, allow_pickle=True)
                if isinstance(self.data, np.ndarray) and self.data.dtype == np.object_:
                    self.data = self.data.tolist()
                logger.info(f"Loaded simulation data from {npy_file} with shape: {np.shape(self.data)}")
            except Exception as e:
                logger.error(f"Error loading npy file: {e}")
                raise e
        else:
            # Assume npy_file is already an array
            self.data = npy_file
            logger.info(f"Using pre-loaded simulation data with shape: {np.shape(self.data)}")

        # Determine the final channel layout from the first simulation's shape
        # (timesteps, H, W, C)
        example_sim = self.data[0]
        self.num_channels = example_sim.shape[-1]
        physical_channels, coord_channels, mask_channel = get_channel_layout(self.num_channels)
        
        # Automatically set teacher_channels if not provided
        if teacher_channels is None:
            self.teacher_channels = len(physical_channels)
            logger.info(f"Auto-setting teacher_channels to {self.teacher_channels} (all physical channels).")
        else:
            self.teacher_channels = teacher_channels
            logger.info(f"Using user-specified teacher_channels = {self.teacher_channels}.")

        self.physical_channels = physical_channels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # sim shape: (timesteps, H, W, C)
        sim = self.data[idx]
        sim = torch.tensor(sim).float()

        # Use first timestep as seed_input and remaining as target sequence.
        seed_input = sim[0]  # shape: (H, W, C)
        selected_phys = self.physical_channels[:self.teacher_channels]
        target_sequence = sim[1:, :, :, selected_phys]  # shape: (timesteps-1, H, W, teacher_channels)

        if self.transform_input:
            # Add a dummy time dimension to seed_input for transformation
            seed_input = self.transform_input(seed_input.unsqueeze(0), self.phys_stats, self.coord_stats)[0]
            if self.debug:
                for ch in range(min(5, seed_input.shape[-1])):
                    ch_data = seed_input[..., ch]
                    logger.info(
                        f"[DEBUG] Seed input channel {ch}: min={ch_data.min().item()}, max={ch_data.max().item()}, mean={ch_data.mean().item()}"
                    )
        if self.transform_target:
            target_sequence = self.transform_target(target_sequence, self.phys_stats)
            if self.debug:
                T = target_sequence.shape[0]
                for t in range(min(3, T)):
                    for ch in range(min(self.teacher_channels, target_sequence.shape[-1])):
                        ch_data = target_sequence[t, :, :, ch]
                        logger.info(
                            f"[DEBUG] Timestep {t+1} target channel {ch}: min={ch_data.min().item()}, max={ch_data.max().item()}, mean={ch_data.mean().item()}"
                        )
        return {
            "pixel_values": seed_input,
            "labels": target_sequence
        }

def compute_normalization_stats(tensor_data):
    """
    Compute normalization stats for the entire dataset of shape
    (num_sims, timesteps, H, W, C).

    - Figures out which channels are physical vs. coordinates vs. mask
      based on the total channel count.
    - Returns two dicts: phys_stats, coord_stats
    """
    C = tensor_data.shape[-1]
    physical_channels, coord_channels, mask_channel = get_channel_layout(C)

    # --- Physical stats (mean, std) ---
    phys_stats = {}
    for ch in physical_channels:
        channel_data = tensor_data[..., ch].float()
        mean = channel_data.mean().item()
        std = channel_data.std().item()
        if std < 1e-12:
            std = 1e-12  # avoid divide-by-zero
        phys_stats[ch] = (mean, std)
        logger.info(f"Physical channel {ch}: mean={mean:.4f}, std={std:.4f}")

    # --- Coordinate stats (min, max) ---
    coord_stats = {}
    for ch in coord_channels:
        channel_data = tensor_data[..., ch].float()
        min_val = channel_data.min().item()
        max_val = channel_data.max().item()
        coord_stats[ch] = (min_val, max_val)
        logger.info(f"Coord channel {ch}: min={min_val:.4f}, max={max_val:.4f}")

    return phys_stats, coord_stats

def get_data_loaders(config):
    """
    Returns training and test DataLoaders using parameters from config.

    Expects the following keys in config["data"]:
      - data_path: path to the .npy file containing simulation data.
      - batch_size: batch size for DataLoader.
      - train_split: fraction of data to use for training.
      - debug: (optional) flag to enable debug logging.
      - domain_range: (optional) tuple (min, max) for the x and y coordinates (default: (0, 2)).

    Raw data shape is assumed to be (T, H, W, 6) with channels:
        0: Ux, 1: Uy, 2: P, 3: Mask, 4: Re, 5: SDF

    Final data shape will be (T, H, W, 8) with channels:
        0: Ux, 1: Uy, 2: P, 3: Re, 4: SDF, 5: x, 6: y, 7: Mask
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset

    # Retrieve config parameters
    data_path = config["data"]["data_path"]
    batch_size = config["data"]["batch_size"]
    train_split = config["data"]["train_split"]
    debug_flag = config["data"].get("debug", False)
    domain_range = config["data"].get("domain_range", (0, 2))  # default (0..2)
    input_dim = config["model"]["input_dim"]

    # Load raw simulation data
    try:
        simulation_data = np.load(data_path, allow_pickle=True)
        if isinstance(simulation_data, np.ndarray) and simulation_data.dtype == np.object_:
            simulation_data = simulation_data.tolist()
        logger.info(f"Loaded simulation data with shape: {np.shape(simulation_data)}")
    except Exception as e:
        logger.error(f"Error loading npy file: {e}")
        raise e

    # Handle single simulation (4D) vs. multiple simulations (5D)
    # if simulation_data.ndim == 4:
    #     # Expand to (1, T, H, W, 6)
    #     simulation_data = np.expand_dims(simulation_data, axis=0)
    
    # Now we expect (num_sims, T, H, W, 6)
    # if simulation_data.ndim != 4 or simulation_data.shape[-1] != 6:
    #     raise ValueError("Expected data shape (num_sims, T, H, W, 6)")

    num_sims, T, H, W, _ = simulation_data.shape

    # ----------------------------------------------------------------
    # 1) Reorder channels from [Ux, Uy, P, Mask, Re, SDF]
    #    to [Ux, Uy, P, Re, SDF, Mask]
    # ----------------------------------------------------------------
    # The new order of indices: [0, 1, 2, 4, 5, 3]
    #   0->Ux
    #   1->Uy
    #   2->P
    #   4->Re
    #   5->SDF
    #   3->Mask
    if input_dim == 6:
        simulation_data = simulation_data[..., [0, 1, 2, 4, 5, 3]]
        logger.info("Reordered channels to [Ux, Uy, P, Re, SDF, Mask].")
    elif input_dim == 4:
        simulation_data = simulation_data
    else:
        simulation_data = simulation_data[..., [0, 1, 2, 3, 4, 6, 5]]
        logger.info("Reordered channels to [Ux, Uy, P, Re, SDF, Mask].")

    # ----------------------------------------------------------------
    # 2) Create x and y coordinate channels
    # ----------------------------------------------------------------
    x_min, x_max = domain_range
    y_min, y_max = domain_range  # adjust if y-range is different
    x_vals = np.linspace(x_min, x_max, W)  # shape (W,)
    y_vals = np.linspace(y_min, y_max, H)  # shape (H,)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)  # shape (H, W)

    # Broadcast to (num_sims, T, H, W, 1)
    x_channel = np.broadcast_to(x_grid, (num_sims, T, H, W))[..., np.newaxis]
    y_channel = np.broadcast_to(y_grid, (num_sims, T, H, W))[..., np.newaxis]

    # ----------------------------------------------------------------
    # 3) Concatenate channels => [Ux, Uy, P, Re, SDF, x, y, Mask]
    # ----------------------------------------------------------------
    # Right now: simulation_data shape => (num_sims, T, H, W, 6)
    # We want final => (num_sims, T, H, W, 8)
    #   indices:
    #       0->Ux, 1->Uy, 2->P, 3->Re, 4->SDF, 5->x, 6->y, 7->Mask
    if input_dim == 6:
        simulation_data = np.concatenate([
            simulation_data[..., :5],  # [Ux, Uy, P, Re, SDF]
            x_channel,                 # channel 5
            y_channel,                 # channel 6
            simulation_data[..., 5:6]  # channel 7 = Mask
        ], axis=-1)
    elif input_dim == 4:
        simulation_data = np.concatenate([
            simulation_data[..., :3],  # [Ux, Uy, P]
            x_channel,                 # channel 4
            y_channel,                 # channel 5
            simulation_data[..., 3:4]  # channel 6 = Mask
        ], axis=-1)
    elif input_dim == 7:
        simulation_data = np.concatenate([
            simulation_data[..., :6],  # [Ux, Uy, P, rho, Re, SDF]
            x_channel,                 # channel 6
            y_channel,                 # channel 7
            simulation_data[..., 6:7]  # channel 8 = Mask
        ], axis=-1)
    else:
        raise ValueError(f"Input dimensions not specified correctly")
    logger.info(f"Final data shape: {simulation_data.shape}")
    if input_dim == 6:
        simulation_data[..., 7] = 1 - simulation_data[..., 7]
    elif input_dim == 4:
        simulation_data[..., 5] = 1 - simulation_data[..., 5]

    # Convert to torch tensor
    tensor_data = torch.tensor(simulation_data)
    num_samples = tensor_data.shape[0]

    # ----------------------------------------------------------------
    # 4) Train/Test split
    # ----------------------------------------------------------------
    train_size = int(train_split * num_samples)
    train_indices = list(range(train_size))
    test_indices = list(range(train_size, num_samples))

    # ----------------------------------------------------------------
    # 5) Compute normalization stats using only training data
    # ----------------------------------------------------------------
    train_data = tensor_data[train_indices]
    phys_stats, coord_stats = compute_normalization_stats(train_data)

    # ----------------------------------------------------------------
    # 6) Create dataset & data loaders
    # ----------------------------------------------------------------
    # We pass 'simulation_data' (the np.array) directly, so SimulationDataset
    # must accept a pre-loaded array.
    if config["model"]["model_type"] == "FNO" or config["model"]["model_type"] == "FFNO":
        full_dataset = SimulationDataset(
            npy_file=simulation_data,
            phys_stats=phys_stats,
            coord_stats=coord_stats,
            teacher_channels=None,  # auto-detect # of physical channels
            debug=debug_flag
        )
    else:
        full_dataset = CEDataset(
            file_path=config["data"]["data_path"],
            max_num_time_steps=config["data"].get("max_num_time_steps", 20),
            time_step_size=config["data"].get("time_step_size", 1),
            fix_input_to_time_step=config["data"].get("fix_input_to_time_step", None),
            allowed_time_transitions=config["data"].get("allowed_time_transitions", [1]),
            resolution=config["data"].get("resolution", 128),
            num_trajectories=config["data"].get("num_trajectories", 600),
            N_val=config["data"].get("N_val", 100),
            N_test=config["data"].get("N_test", 80),
            which="train",
            use_all_channels=config["data"].get("use_all_channels", False)
        )

    # Define the desired split sizes
    num_train = config["data"].get("num_trajectories", 400)
    num_val   = config["data"].get("N_val", 100)
    num_test  = config["data"].get("N_test", 80)
    total_required = num_train + num_val + num_test

    # Option 1: If your full_dataset has exactly or more than total_required samples,
    # you can either select a subset or ensure it is exactly the size you need.
    if len(full_dataset) >= total_required:
        # Optionally, limit the full_dataset to the required size:
        indices = list(range(total_required))
        full_dataset = torch.utils.data.Subset(full_dataset, indices)
    else:
        raise ValueError("The full dataset has fewer samples than required for the split.")

    # Randomly split the dataset into train, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [num_train, num_val, num_test])

    # Create DataLoaders for each subset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader



