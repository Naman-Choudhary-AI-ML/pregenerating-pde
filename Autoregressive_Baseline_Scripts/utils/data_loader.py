# utils/data_loader.py
# -----------------------------------------------------------------------------
# PURPOSE
#   You asked whether the DataLoader could be the root cause of NaNs.
#   Weight‑norm in FFNO *was* the direct NaN source in your trace, but the
#   loader can still feed bad values that *trigger* it. This version adds:
#
#   1) Strong SANITIZATION:
#        • np.nan_to_num on the raw arrays (no NaNs/Infs enter the model)
#        • Zero physical channels inside obstacles for ALL timesteps
#        • (optional) clamp physical magnitudes to avoid crazy outliers
#
#   2) SAMPLE FILTERING (optional):
#        • Drop samples whose valid (fluid) fraction at t=0 is too small
#
#   3) DEBUG TELEMETRY (opt‑in via env):
#        export DL_DEBUG=1                # batch‑level stats every DL_DEBUG_EVERY
#        export DL_DEBUG_EVERY=200
#        export DL_MIN_VALID_FRAC=0.0005  # drop samples with <0.05% valid area
#        export DL_CLIP_ABS=0             # if >0, clip |Ux,Uy,P| to this value
#        export DL_MASK_ALL=0             # if 1, also mask Re/SDF in obstacles
#        export DL_SCAN_LIMIT=0           # if >0, scan at most N sims for filter
#
# RAW ORDER  in files: [Ux, Uy, P, Re, Mask, SDF]
# OUTPUT to model     : [Ux, Uy, P, Re, SDF, ValidMask]   (ValidMask: 1=fluid)
# -----------------------------------------------------------------------------

from __future__ import annotations
import os
import logging
from typing import List, Tuple
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

# ------------------------------- utilities -----------------------------------
def _valid_fraction_seed(raw: np.ndarray) -> float:
    """ valid (fluid) fraction at t=0 based on MASK channel. """
    seed_mask = raw[0, ..., MASK]  # (H,W)
    obs = (seed_mask > 0.5).astype(np.float32)
    vmask = 1.0 - obs
    return float(vmask.mean())

def _sanitize_and_layout(sim: np.ndarray) -> np.ndarray:
    """
    sim (T,H,W,6) → out (T,H,W,6) in model order, fully finite, masked.
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

    # Optional clipping of physical magnitudes
    if DL_CLIP_ABS > 0:
        np.clip(out[..., :3], -DL_CLIP_ABS, DL_CLIP_ABS, out=out[..., :3])

    # Final guard: ensure finite
    if not np.isfinite(out).all():
        bad = np.where(~np.isfinite(out))
        raise RuntimeError(f"[DataLoader] non‑finite values after sanitize at indices (first 5): {list(zip(*(b[:5] for b in bad)))}")
    return out

# ------------------------------- dataset -------------------------------------
class _SingleFileDS(Dataset):
    """
    Memmap dataset for one .npy file (N,T,H,W,6) or (T,H,W,6).
    Optional filtering by valid fraction at seed.
    """
    def __init__(self,
                 path: str,
                 output_dim: int,
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
                # Keep the rest unfiltered (to avoid bias), but log the stats
                log.warning(f"{path}: scanned only first {limit}/{self.N} sims for valid‑fraction filter; kept {len(good)} as 'good'; "
                            f"remaining {self.N - limit} left unfiltered.")
                good += list(range(limit, self.N))
            self.indices = good
            if not self.indices:
                raise RuntimeError(f"{path}: all samples failed valid‑fraction ≥ {min_valid_frac}")

        # Quick seed sanity
        sample = _sanitize_and_layout(self.mm[self.indices[0]])
        ms = float(sample[0, ..., OUT_VMASK].mean())
        log.info(f"{path}: seed valid‑mask mean ~ {ms:.4f} (1=fluid). kept={len(self.indices)}/{self.N} indices.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        mm_idx = self.indices[i]
        sim = _sanitize_and_layout(self.mm[mm_idx])                 # (T,H,W,6)
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
    px = x[..., :3]   # Ux,Uy,P
    px_min = float(px.min().item())
    px_max = float(px.max().item())
    re_min = float(x[..., OUT_RE].min().item())
    re_max = float(x[..., OUT_RE].max().item())
    sdf_min = float(x[..., OUT_SDF].min().item())
    sdf_max = float(x[..., OUT_SDF].max().item())

    # label stats
    y_min = float(y.min().item())
    y_max = float(y.max().item())

    log.info(
        f"[DL][{name}] step={step} finite(x)={x_finite} finite(y)={y_finite} "
        f"valid_frac={valid_frac:.4f} U,V,P∈[{px_min:.3e},{px_max:.3e}] "
        f"Re∈[{re_min:.3e},{re_max:.3e}] SDF∈[{sdf_min:.3e},{sdf_max:.3e}] "
        f"labels∈[{y_min:.3e},{y_max:.3e}]  x.shape={tuple(x.shape)} y.shape={tuple(y.shape)}"
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
                    log.error(f"[DL][{self.name}] stats error: {e}")
            yield batch
    def __len__(self): 
        return len(self.loader)

# ------------------------------ public factory --------------------------------
def get_data_loaders(*, config: dict,
                     easy_train: int, hard_train: int,
                     easy_path: str, hard_path: str):
    """
    Returns five loaders:
      train_loader (mixed easy+hard),
      val_easy_loader, val_hard_loader,
      test_easy_loader, test_hard_loader.
    """
    batch = config["data"]["batch_size"]
    output_dim = config["model"]["output_dim"]

    ds_easy = _SingleFileDS(easy_path, output_dim=output_dim)
    ds_hard = _SingleFileDS(hard_path, output_dim=output_dim)

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

    return (train_loader, val_easy_loader, val_hard_loader, test_easy_loader, test_hard_loader)
