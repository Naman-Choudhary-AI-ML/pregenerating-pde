import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Normalization functions
def normalize_sequence(x, phys_stats, coord_stats):
    """
    x: tensor of shape (timesteps, H, W, 7)
    - Normalize physical channels (0-3) using (x - mean) / std.
    - Scale coordinate channels (4-5) to [0, 1] using (x - min) / (max - min).
    - Leaves any additional channels (e.g., mask) unchanged.
    """
    x_norm = x.clone().float()
    for ch in range(4):
        mean, std = phys_stats[ch]
        x_norm[..., ch] = (x_norm[..., ch] - mean) / std
    for ch in [4, 5]:
        min_val, max_val = coord_stats[ch]
        x_norm[..., ch] = (x_norm[..., ch] - min_val) / (max_val - min_val)
    return x_norm

def normalize_target_sequence(y, phys_stats):
    """
    y: tensor of shape (timesteps-1, H, W, 4)
    - Normalize each physical channel using (x - mean) / std.
    """
    y_norm = y.clone().float()
    for ch in range(4):
        mean, std = phys_stats[ch]
        y_norm[..., ch] = (y_norm[..., ch] - mean) / std
    return y_norm

# Custom Dataset for Autoregressive Training
class SimulationDataset(Dataset):
    def __init__(self, npy_file, phys_stats, coord_stats, teacher_channels=4, 
                 transform_input=normalize_sequence, transform_target=normalize_target_sequence, debug=False):
        """
        Args:
            npy_file: path to the .npy file.
            phys_stats, coord_stats: normalization stats.
            teacher_channels: number of channels to use for target.
            transform_input, transform_target: transformation functions.
            debug: prints debug info if True.
        """
        self.npy_file = npy_file
        try:
            with open(npy_file, "rb") as f:
                self.data = np.load(f, allow_pickle=True)
            if isinstance(self.data, np.ndarray) and self.data.dtype == np.object_:
                self.data = self.data.tolist()
            logger.info(f"Loaded simulation data from {npy_file} with shape: {np.shape(self.data)}")
        except Exception as e:
            logger.error(f"Error loading npy file: {e}")
            raise e
        
        self.teacher_channels = teacher_channels
        self.phys_stats = phys_stats
        self.coord_stats = coord_stats
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.debug = debug

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get one simulation: shape (timesteps, H, W, C)
        sim = self.data[idx]
        sim = torch.tensor(sim).float()
        # Use the first timestep as seed input and the remaining as target sequence
        seed_input = sim[0]
        target_sequence = sim[1:, :, :, :self.teacher_channels]

        if self.transform_input:
            # Add batch dimension and then remove it after transformation
            seed_input = self.transform_input(seed_input.unsqueeze(0), self.phys_stats, self.coord_stats)[0]
            if self.debug:
                for ch in range(self.teacher_channels):
                    ch_data = seed_input[..., ch]
                    logger.info(f"[DEBUG] Seed input channel {ch}: min={ch_data.min().item()}, max={ch_data.max().item()}, mean={ch_data.mean().item()}")
        if self.transform_target:
            target_sequence = self.transform_target(target_sequence, self.phys_stats)
            if self.debug:
                T = target_sequence.shape[0]
                for t in range(min(3, T)):
                    for ch in range(self.teacher_channels):
                        ch_data = target_sequence[t, :, :, ch]
                        logger.info(f"[DEBUG] Timestep {t+1} target channel {ch}: min={ch_data.min().item()}, max={ch_data.max().item()}, mean={ch_data.mean().item()}")
        return seed_input, target_sequence

def compute_normalization_stats(tensor_data):
    """
    Computes normalization stats for physical channels (0-3) and coordinate channels (4-5)
    from tensor_data of shape (num_simulations, timesteps, H, W, C).
    """
    phys_stats = {}
    for ch in range(4):
        channel_data = tensor_data[..., ch].float()
        mean = channel_data.mean().item()
        std = channel_data.std().item()
        phys_stats[ch] = (mean, std)
        logger.info(f"Physical channel {ch}: mean = {mean:.4f}, std = {std:.4f}")
    coord_stats = {}
    for ch in [4, 5]:
        channel_data = tensor_data[..., ch].float()
        min_val = channel_data.min().item()
        max_val = channel_data.max().item()
        coord_stats[ch] = (min_val, max_val)
        logger.info(f"Coordinate channel {ch}: min = {min_val:.4f}, max = {max_val:.4f}")
    return phys_stats, coord_stats

def get_data_loaders(config):
    """
    Returns training and test DataLoaders using parameters from config.
    Expects the following keys in config["data"]:
      - data_path: path to the .npy file.
      - batch_size: batch size for DataLoader.
      - train_split: fraction of data to use for training.
      - debug: (optional) flag to enable debug logging.
    """
    data_path = config["data"]["data_path"]
    batch_size = config["data"]["batch_size"]
    train_split = config["data"]["train_split"]
    debug_flag = config["data"].get("debug", False)

    # Load raw data for computing stats
    try:
        simulation_data = np.load(data_path, allow_pickle=True)
        if isinstance(simulation_data, np.ndarray) and simulation_data.dtype == np.object_:
            simulation_data = simulation_data.tolist()
        logger.info(f"Loaded simulation data with shape: {np.shape(simulation_data)}")
    except Exception as e:
        logger.error(f"Error loading npy file: {e}")
        raise e

    tensor_data = torch.tensor(simulation_data)
    num_simulations = tensor_data.shape[0]

    # Split indices for train and test sets
    train_size = int(train_split * num_simulations)
    train_indices = list(range(train_size))
    test_indices = list(range(train_size, num_simulations))

    # Compute normalization stats using training data
    train_data = tensor_data[train_indices]
    phys_stats, coord_stats = compute_normalization_stats(train_data)

    # Create full dataset instance
    full_dataset = SimulationDataset(data_path, phys_stats, coord_stats, teacher_channels=4, debug=debug_flag)

    # Create train and test subsets
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
