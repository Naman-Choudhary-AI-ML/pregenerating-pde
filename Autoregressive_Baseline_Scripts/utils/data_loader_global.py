# utils/data_loader_global.py
# -----------------------------------------------------------------------------
# PURPOSE
#   This is a modified version that implements GLOBAL NORMALIZATION.
#   
#   Key differences from original:
#   - Computes dataset-wide statistics (mean/std) across all samples
#   - Applies global z-score normalization: (x - global_mean) / global_std
#   - Configurable which channels to normalize globally
#   - Can normalize physical channels only OR include Reynolds number
#   - Statistics computed only from fluid regions (using ValidMask)
#
#   GLOBAL NORMALIZATION STRATEGY:
#   - Pass 1: Scan dataset to compute global mean/std per channel
#   - Pass 2: Apply normalization during data loading
#   - Statistics cached to avoid recomputation
#
# RAW ORDER  in files: [Ux, Uy, P, Re, Mask, SDF]
# OUTPUT to model     : [Ux_norm, Uy_norm, P_norm, Re_norm?, SDF, ValidMask]
# -----------------------------------------------------------------------------

from __future__ import annotations
import os
import logging
import pickle
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# indices (raw)
UX, UY, P, RE, MASK, SDF = 0, 1, 2, 3, 4, 5
# indices (model out)
OUT_UX, OUT_UY, OUT_P, OUT_RE, OUT_SDF, OUT_VMASK = 0, 1, 2, 3, 4, 5
OUT_C = 6

# ------------------------------ env controls ---------------------------------
DL_DEBUG        = int(os.getenv("DL_DEBUG", "0")) == 1
DL_DEBUG_EVERY  = int(os.getenv("DL_DEBUG_EVERY", "200"))
DL_MIN_VALID    = float(os.getenv("DL_MIN_VALID_FRAC", "0.0"))   # 0 → no filtering
DL_CLIP_ABS     = float(os.getenv("DL_CLIP_ABS", "0.0"))         # 0 → no clipping
DL_MASK_ALL     = int(os.getenv("DL_MASK_ALL", "0")) == 1        # also zero Re/SDF in obstacles
DL_SCAN_LIMIT   = int(os.getenv("DL_SCAN_LIMIT", "0"))           # 0 → scan all
DL_NORM_EPS     = float(os.getenv("DL_NORM_EPS", "1e-8"))        # epsilon for normalization

# ------------------------------- utilities -----------------------------------
def _valid_fraction_seed(raw: np.ndarray) -> float:
    """ valid (fluid) fraction at t=0 based on MASK channel. """
    seed_mask = raw[0, ..., MASK]  # (H,W)
    obs = (seed_mask > 0.5).astype(np.float32)
    vmask = 1.0 - obs
    return float(vmask.mean())

def _compute_global_stats(datasets: List[np.ndarray], 
                         normalize_channels: List[int],
                         cache_file: Optional[str] = None) -> Dict[int, Dict[str, float]]:
    """
    Compute global mean and std for specified channels across all datasets.
    Only uses fluid regions for statistics computation.
    
    Args:
        datasets: List of memory-mapped arrays [(N,T,H,W,6), ...]
        normalize_channels: List of channel indices to compute stats for
        cache_file: Optional file to cache/load statistics
    
    Returns:
        Dict mapping channel_idx -> {'mean': float, 'std': float}
    """
    
    # Try to load cached stats
    if cache_file and os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                stats = pickle.load(f)
            log.info(f"Loaded cached global statistics from {cache_file}")
            return stats
        except Exception as e:
            log.warning(f"Failed to load cached stats: {e}. Recomputing...")
    
    log.info(f"Computing global statistics for channels {normalize_channels}...")
    
    # Initialize accumulators
    channel_stats = {}
    for ch in normalize_channels:
        channel_stats[ch] = {'sum': 0.0, 'sum_sq': 0.0, 'count': 0}
    
    total_samples = 0
    for dataset_idx, dataset in enumerate(datasets):
        log.info(f"Processing dataset {dataset_idx+1}/{len(datasets)} with shape {dataset.shape}")
        
        if dataset.ndim == 5:  # (N,T,H,W,6)
            N, T, H, W, C = dataset.shape
        elif dataset.ndim == 4:  # (T,H,W,6)
            N, T, H, W, C = 1, *dataset.shape
            dataset = dataset[None, ...]
        else:
            raise ValueError(f"Expected dataset shape (N,T,H,W,6) or (T,H,W,6), got {dataset.shape}")
        
        # Process each sample
        for i in range(N):
            if (i + 1) % 100 == 0:
                log.info(f"  Processed {i+1}/{N} samples from dataset {dataset_idx+1}")
            
            sim = dataset[i]  # (T,H,W,6)
            
            # Get valid mask (fluid regions)
            obs = (sim[..., MASK] > 0.5).astype(np.float32)
            vmask = (1.0 - obs).astype(np.float32)
            
            # Process each timestep
            for t in range(T):
                mask_t = vmask[t]  # (H,W)
                fluid_indices = mask_t > 0.5
                
                if not fluid_indices.any():
                    continue  # Skip if no fluid regions
                
                # Accumulate stats for each channel
                for ch in normalize_channels:
                    data = sim[t, :, :, ch]  # (H,W)
                    fluid_data = data[fluid_indices]
                    
                    if len(fluid_data) > 0:
                        channel_stats[ch]['sum'] += float(np.sum(fluid_data))
                        channel_stats[ch]['sum_sq'] += float(np.sum(fluid_data ** 2))
                        channel_stats[ch]['count'] += len(fluid_data)
        
        total_samples += N
    
    # Compute final mean and std
    global_stats = {}
    for ch in normalize_channels:
        if channel_stats[ch]['count'] == 0:
            log.warning(f"No valid data found for channel {ch}. Using default stats.")
            global_stats[ch] = {'mean': 0.0, 'std': 1.0}
        else:
            count = channel_stats[ch]['count']
            mean = channel_stats[ch]['sum'] / count
            var = (channel_stats[ch]['sum_sq'] / count) - (mean ** 2)
            std = max(np.sqrt(var), DL_NORM_EPS)  # Ensure std > 0
            
            global_stats[ch] = {'mean': float(mean), 'std': float(std)}
            log.info(f"Channel {ch}: mean={mean:.6f}, std={std:.6f} (from {count} points)")
    
    # Cache the stats
    if cache_file:
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(global_stats, f)
            log.info(f"Cached global statistics to {cache_file}")
        except Exception as e:
            log.warning(f"Failed to cache stats: {e}")
    
    log.info(f"Global statistics computed from {total_samples} samples")
    return global_stats

def _global_normalize_channels(sim: np.ndarray, vmask: np.ndarray,
                              global_stats: Dict[int, Dict[str, float]],
                              normalize_channels: List[int]) -> np.ndarray:
    """
    Apply global normalization to specified channels.
    
    Args:
        sim: (T,H,W,6) simulation data in model order
        vmask: (T,H,W) valid mask (1=fluid, 0=obstacle)
        global_stats: Global statistics per channel
        normalize_channels: Which channels to normalize
    
    Returns:
        sim: normalized simulation data
    """
    sim_norm = sim.copy()
    
    for ch in normalize_channels:
        if ch not in global_stats:
            continue
            
        mean = global_stats[ch]['mean']
        std = global_stats[ch]['std']
        
        # Normalize all timesteps for this channel
        for t in range(sim.shape[0]):
            data = sim_norm[t, :, :, ch]
            sim_norm[t, :, :, ch] = (data - mean) / std
            
            # Apply mask to ensure obstacles remain zero
            sim_norm[t, :, :, ch] *= vmask[t, :, :]
    
    return sim_norm

def _sanitize_and_layout(sim: np.ndarray, 
                        global_stats: Optional[Dict[int, Dict[str, float]]] = None,
                        normalize_channels: Optional[List[int]] = None) -> np.ndarray:
    """
    sim (T,H,W,6) → out (T,H,W,6) in model order, fully finite, masked, and globally normalized.
    """
    assert sim.ndim == 4 and sim.shape[-1] == 6, f"Expected (...,6) got {sim.shape}"
    sim = np.nan_to_num(sim.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # ValidMask (1=fluid)
    obs = (sim[..., MASK] > 0.5).astype(np.float32)
    vmask = (1.0 - obs).astype(np.float32)

    # Arrange outputs
    out = np.empty_like(sim, dtype=np.float32)
    out[..., OUT_UX]    = sim[..., UX]
    out[..., OUT_UY]    = sim[..., UY]
    out[..., OUT_P]     = sim[..., P]
    out[..., OUT_RE]    = sim[..., RE]
    out[..., OUT_SDF]   = sim[..., SDF]
    out[..., OUT_VMASK] = vmask

    # Zero physical channels in obstacles across ALL timesteps
    out[..., :3] *= vmask[..., None]
    if DL_MASK_ALL:
        out[..., 3:5] *= vmask[..., None]  # also zero Re, SDF inside obstacles

    # Apply global normalization if specified
    if global_stats is not None and normalize_channels is not None:
        out = _global_normalize_channels(out, vmask, global_stats, normalize_channels)

    # Optional clipping of normalized values
    if DL_CLIP_ABS > 0 and normalize_channels is not None:
        for ch in normalize_channels:
            np.clip(out[..., ch], -DL_CLIP_ABS, DL_CLIP_ABS, out=out[..., ch])

    # Final guard: ensure finite
    if not np.isfinite(out).all():
        bad = np.where(~np.isfinite(out))
        raise RuntimeError(f"[DataLoader] non‑finite values after sanitize at indices (first 5): {list(zip(*(b[:5] for b in bad)))}")
    return out

# ------------------------------- dataset -------------------------------------
class _SingleFileDS(Dataset):
    """
    Memmap dataset for one .npy file (N,T,H,W,6) or (T,H,W,6).
    Features GLOBAL NORMALIZATION of specified channels.
    """
    def __init__(self,
                 path: str,
                 output_dim: int,
                 global_stats: Optional[Dict[int, Dict[str, float]]] = None,
                 normalize_channels: Optional[List[int]] = None,
                 min_valid_frac: float = DL_MIN_VALID,
                 scan_limit: int = DL_SCAN_LIMIT):
        mm = np.load(path, mmap_mode="r")
        if mm.ndim == 5 and mm.shape[-1] == 6:
            self.mm = mm
        elif mm.ndim == 4 and mm.shape[-1] == 6:
            self.mm = mm[None, ...]
        else:
            raise ValueError(f"{path}: expected (N,T,H,W,6) or (T,H,W,6); got {mm.shape}")

        self.N = self.mm.shape[0]
        self.output_dim = output_dim
        self.teacher_ch = min(self.output_dim, 3)  # U,V,P typically
        self.global_stats = global_stats
        self.normalize_channels = normalize_channels or []

        # Filter indices by valid fraction if requested
        self.indices = list(range(self.N))
        if min_valid_frac > 0.0:
            good = []
            limit = self.N if scan_limit <= 0 else min(scan_limit, self.N)
            for i in range(limit):
                vf = _valid_fraction_seed(self.mm[i])
                if vf >= min_valid_frac:
                    good.append(i)
            if scan_limit > 0 and limit < self.N:
                good += list(range(limit, self.N))
            self.indices = good
            if not self.indices:
                raise RuntimeError(f"{path}: all samples failed valid‑fraction ≥ {min_valid_frac}")

        # Quick seed sanity
        sample = _sanitize_and_layout(self.mm[self.indices[0]], 
                                    global_stats=self.global_stats,
                                    normalize_channels=self.normalize_channels)
        ms = float(sample[0, ..., OUT_VMASK].mean())
        norm_status = f"GLOBAL norm on channels {self.normalize_channels}" if self.normalize_channels else "NO normalization"
        log.info(f"{path}: seed valid‑mask mean ~ {ms:.4f} (1=fluid). kept={len(self.indices)}/{self.N} indices. {norm_status}.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        mm_idx = self.indices[i]
        sim = _sanitize_and_layout(self.mm[mm_idx], 
                                 global_stats=self.global_stats,
                                 normalize_channels=self.normalize_channels)  # (T,H,W,6)
        seed = torch.from_numpy(sim[0])                              # (H,W,6)
        tgt  = torch.from_numpy(sim[1:, :, :, :self.teacher_ch])     # (T-1,H,W,C_out)
        return {"pixel_values": seed, "labels": tgt}

# ----------------------- optional batch‑level telemetry -----------------------
def _batch_stats(name: str, batch: dict, step: int):
    """ Lightweight per‑batch stats (only when DL_DEBUG=1). """
    x = batch["pixel_values"].float()
    y = batch["labels"].float()

    x_finite = torch.isfinite(x).all().item()
    y_finite = torch.isfinite(y).all().item()
    vmask = x[..., OUT_VMASK]
    valid_frac = float(vmask.mean().item())

    # numeric ranges
    px = x[..., :3]   # Ux,Uy,P (potentially normalized)
    px_min = float(px.min().item())
    px_max = float(px.max().item())
    px_mean = float(px.mean().item())
    px_std = float(px.std().item())
    
    re_min = float(x[..., OUT_RE].min().item())
    re_max = float(x[..., OUT_RE].max().item())
    sdf_min = float(x[..., OUT_SDF].min().item())
    sdf_max = float(x[..., OUT_SDF].max().item())

    # label stats
    y_min = float(y.min().item())
    y_max = float(y.max().item())
    y_mean = float(y.mean().item())
    y_std = float(y.std().item())

    log.info(
        f"[DL_GLOBAL][{name}] step={step} finite(x)={x_finite} finite(y)={y_finite} "
        f"valid_frac={valid_frac:.4f} U,V,P∈[{px_min:.3e},{px_max:.3e}] μ={px_mean:.3e} σ={px_std:.3e} "
        f"Re∈[{re_min:.3e},{re_max:.3e}] SDF∈[{sdf_min:.3e},{sdf_max:.3e}] "
        f"labels∈[{y_min:.3e},{y_max:.3e}] μ={y_mean:.3e} σ={y_std:.3e}  x.shape={tuple(x.shape)} y.shape={tuple(y.shape)}"
    )

class _DebugWrapper:
    """ Wrap any DataLoader to emit _batch_stats every K steps when DL_DEBUG=1. """
    def __init__(self, loader: DataLoader, name: str, every: int = DL_DEBUG_EVERY):
        self.loader, self.name, self.every = loader, name, max(1, every)
    def __iter__(self):
        for step, batch in enumerate(self.loader):
            if DL_DEBUG and (step % self.every == 0):
                try:
                    _batch_stats(self.name, batch, step)
                except Exception as e:
                    log.error(f"[DL_GLOBAL][{self.name}] stats error: {e}")
            yield batch
    def __len__(self): 
        return len(self.loader)

# ------------------------------ public factory --------------------------------
def get_data_loaders(*, config: dict,
                     easy_train: int, hard_train: int,
                     easy_path: str, hard_path: str,
                     normalize_channels: Optional[List[int]] = None,
                     cache_stats: bool = True):
    """
    Returns five loaders with GLOBAL NORMALIZATION:
      train_loader (mixed easy+hard),
      val_easy_loader, val_hard_loader,
      test_easy_loader, test_hard_loader.
      
    Args:
        normalize_channels: List of channel indices to apply global normalization to.
                           Default None means no normalization.
                           Examples:
                           - [0, 1, 2] = normalize Ux, Uy, P only
                           - [0, 1, 2, 3] = normalize Ux, Uy, P, Re
                           - [3] = normalize only Reynolds number
        cache_stats: Whether to cache computed global statistics
    """
    batch = config["data"]["batch_size"]
    output_dim = config["model"]["output_dim"]

    # Load datasets for statistics computation
    easy_mm = np.load(easy_path, mmap_mode="r")
    hard_mm = np.load(hard_path, mmap_mode="r")
    
    global_stats = None
    if normalize_channels is not None and len(normalize_channels) > 0:
        # Generate cache filename based on data paths and channels
        cache_dir = os.path.join(os.path.dirname(easy_path), ".cache")
        easy_name = os.path.splitext(os.path.basename(easy_path))[0]
        hard_name = os.path.splitext(os.path.basename(hard_path))[0]
        cache_file = os.path.join(cache_dir, f"global_stats_{easy_name}_{hard_name}_ch{'_'.join(map(str, normalize_channels))}.pkl")
        
        if not cache_stats:
            cache_file = None
        
        # Compute global statistics
        global_stats = _compute_global_stats([easy_mm, hard_mm], normalize_channels, cache_file)
        
        log.info("Global normalization will be applied to channels: " + 
                ", ".join([f"{ch} ({'Ux' if ch==0 else 'Uy' if ch==1 else 'P' if ch==2 else 'Re' if ch==3 else 'SDF' if ch==4 else 'Mask'})" 
                          for ch in normalize_channels]))

    ds_easy = _SingleFileDS(easy_path, output_dim=output_dim, 
                           global_stats=global_stats, normalize_channels=normalize_channels)
    ds_hard = _SingleFileDS(hard_path, output_dim=output_dim,
                           global_stats=global_stats, normalize_channels=normalize_channels)

    # Deterministic shuffle
    g = torch.Generator().manual_seed(42)
    idx_easy = torch.randperm(len(ds_easy), generator=g).tolist()
    idx_hard = torch.randperm(len(ds_hard), generator=g).tolist()

    if easy_train > len(idx_easy) or hard_train > len(idx_hard):
        raise ValueError("Requested more train sims than available.")

    # TRAIN
    trE = idx_easy[:easy_train]
    trH = idx_hard[:hard_train]

    # Leftovers for VAL/TEST
    leftE = idx_easy[easy_train:]
    leftH = idx_hard[hard_train:]

    VAL_E, VAL_H, TEST_E, TEST_H = 50, 50, 40, 40
    needE, needH = VAL_E + TEST_E, VAL_H + TEST_H
    if len(leftE) < needE or len(leftH) < needH:
        raise ValueError(
            f"Not enough sims left for val/test. "
            f"Left easy={len(leftE)} (need {needE}), left hard={len(leftH)} (need {needH}). "
            f"Reduce easy_train/hard_train."
        )

    valE, testE = leftE[:VAL_E], leftE[VAL_E:VAL_E + TEST_E]
    valH, testH = leftH[:VAL_H], leftH[VAL_H:VAL_H + TEST_H]

    Sub = Subset
    train_ds     = Sub(ds_easy, trE) + Sub(ds_hard, trH)
    val_easy_ds  = Sub(ds_easy, valE)
    val_hard_ds  = Sub(ds_hard, valH)
    test_easy_ds = Sub(ds_easy, testE)
    test_hard_ds = Sub(ds_hard, testH)

    def _dl(dset, shuffle):
        return DataLoader(dset, batch_size=batch, shuffle=shuffle,
                          num_workers=4, pin_memory=True, drop_last=False)

    train_loader     = _dl(train_ds, True)
    val_easy_loader  = _dl(val_easy_ds, False)
    val_hard_loader  = _dl(val_hard_ds, False)
    test_easy_loader = _dl(test_easy_ds, False)
    test_hard_loader = _dl(test_hard_ds, False)

    # Optional debug wrappers (no‑op if DL_DEBUG=0)
    train_loader     = _DebugWrapper(train_loader, "train")
    val_easy_loader  = _DebugWrapper(val_easy_loader, "val_easy")
    val_hard_loader  = _DebugWrapper(val_hard_loader, "val_hard")
    test_easy_loader = _DebugWrapper(test_easy_loader, "test_easy")
    test_hard_loader = _DebugWrapper(test_hard_loader, "test_hard")

    norm_status = f"WITH global normalization on channels {normalize_channels}" if normalize_channels else "WITHOUT normalization"
    log.info(f"Created data loaders {norm_status}.")
    
    if global_stats:
        for ch, stats in global_stats.items():
            ch_name = ['Ux', 'Uy', 'P', 'Re', 'SDF', 'Mask'][ch] if ch < 6 else f'Ch{ch}'
            log.info(f"Global stats for {ch_name} (ch{ch}): mean={stats['mean']:.6f}, std={stats['std']:.6f}")

    return (train_loader, val_easy_loader, val_hard_loader, test_easy_loader, test_hard_loader)

# ------------------------------ convenience functions -------------------------
def get_physical_channels_normalized_loaders(*, config: dict,
                                            easy_train: int, hard_train: int,
                                            easy_path: str, hard_path: str):
    """Convenience function that applies global normalization to Ux, Uy, P only."""
    return get_data_loaders(
        config=config, easy_train=easy_train, hard_train=hard_train,
        easy_path=easy_path, hard_path=hard_path, 
        normalize_channels=[0, 1, 2]  # Ux, Uy, P
    )

def get_physical_and_re_normalized_loaders(*, config: dict,
                                         easy_train: int, hard_train: int,
                                         easy_path: str, hard_path: str):
    """Convenience function that applies global normalization to Ux, Uy, P, Re."""
    return get_data_loaders(
        config=config, easy_train=easy_train, hard_train=hard_train,
        easy_path=easy_path, hard_path=hard_path, 
        normalize_channels=[0, 1, 2, 3]  # Ux, Uy, P, Re
    )

def get_re_only_normalized_loaders(*, config: dict,
                                 easy_train: int, hard_train: int,
                                 easy_path: str, hard_path: str):
    """Convenience function that applies global normalization to Reynolds number only."""
    return get_data_loaders(
        config=config, easy_train=easy_train, hard_train=hard_train,
        easy_path=easy_path, hard_path=hard_path, 
        normalize_channels=[3]  # Re only
    )


