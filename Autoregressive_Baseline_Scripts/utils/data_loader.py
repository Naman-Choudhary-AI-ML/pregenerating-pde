# utils/data_loader.py
# -----------------------------------------------------------------------------
# Loader for two .npy files (easy=no-hole, hard=hole)
# RAW ORDER in files (per your note): [Ux, Uy, P, Re, Mask, SDF]
# OUTPUT to the model:                [Ux, Uy, P, Re, SDF, ValidMask]
# IMPORTANT FIXES FOR NaNs:
#   • Convert any NaNs/Infs in the arrays to 0.0 (np.nan_to_num)
#   • Zero-out physical channels inside obstacles across ALL timesteps
#       (so targets are finite even under the mask)
# -----------------------------------------------------------------------------

from __future__ import annotations
import logging
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

UX, UY, P, RE, MASK, SDF = 0, 1, 2, 3, 4, 5
OUT_UX, OUT_UY, OUT_P, OUT_RE, OUT_SDF, OUT_VMASK = 0, 1, 2, 3, 4, 5
OUT_C = 6

def _to_model_layout(sim: np.ndarray) -> np.ndarray:
    """
    sim: (T,H,W,6) raw -> out: (T,H,W,6) model layout
    • Replace NaN/Inf with 0.0
    • Make ValidMask = 1 - (Mask>0.5)
    • Zero physical channels inside obstacles for ALL timesteps
    """
    assert sim.ndim == 4 and sim.shape[-1] == 6, f"Expected (...,6) got {sim.shape}"
    sim = np.nan_to_num(sim.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    obs = (sim[..., MASK] > 0.5).astype(np.float32)     # 1=obstacle
    vmask = (1.0 - obs).astype(np.float32)              # 1=fluid

    out = np.empty_like(sim, dtype=np.float32)
    out[..., OUT_UX]    = sim[..., UX]
    out[..., OUT_UY]    = sim[..., UY]
    out[..., OUT_P]     = sim[..., P]
    out[..., OUT_RE]    = sim[..., RE]
    out[..., OUT_SDF]   = sim[..., SDF]
    out[..., OUT_VMASK] = vmask

    # Zero physical channels in obstacles for *all* timesteps to avoid NaNs
    out[..., :3] *= vmask[..., None]   # broadcast mask into [Ux,Uy,P]

    return out

class _SingleFileDS(Dataset):
    def __init__(self, path: str, output_dim: int):
        mm = np.load(path, mmap_mode="r")
        if mm.ndim == 5 and mm.shape[-1] == 6:
            self.mm = mm
        elif mm.ndim == 4 and mm.shape[-1] == 6:
            self.mm = mm[None, ...]
        else:
            raise ValueError(f"{path}: expected (N,T,H,W,6) or (T,H,W,6); got {mm.shape}")

        self.N = self.mm.shape[0]
        self.output_dim = output_dim
        self.teacher_ch = min(self.output_dim, 3)  # first channels U,V,P

        # quick mask sanity (seed)
        sample = _to_model_layout(self.mm[0])
        ms = float(sample[0, ..., OUT_VMASK].mean())
        log.info(f"{path}: seed valid-mask mean ~ {ms:.4f} (1=fluid). shape={self.mm.shape}")

    def __len__(self): return self.N

    def __getitem__(self, idx):
        sim = _to_model_layout(self.mm[idx])                 # (T,H,W,6)
        seed = torch.from_numpy(sim[0])                      # (H,W,6)
        tgt  = torch.from_numpy(sim[1:, :, :, :self.teacher_ch])  # (T-1,H,W,C_out)
        return {"pixel_values": seed, "labels": tgt}

def get_data_loaders(*, config: dict,
                     easy_train: int, hard_train: int,
                     easy_path: str, hard_path: str):
    batch = config["data"]["batch_size"]
    output_dim = config["model"]["output_dim"]

    ds_easy = _SingleFileDS(easy_path, output_dim=output_dim)
    ds_hard = _SingleFileDS(hard_path, output_dim=output_dim)

    # deterministic shuffle
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

    return (_dl(train_ds, True),
            _dl(val_easy_ds, False),  _dl(val_hard_ds, False),
            _dl(test_easy_ds, False), _dl(test_hard_ds, False))
