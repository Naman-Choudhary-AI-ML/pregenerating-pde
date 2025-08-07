# utils/data_loader.py
# -----------------------------------------------------------------------------
# Memory‑friendly loader that
#   • ingests TWO .npy files   [easy (no‑hole) first, hard (hole) second]
#   • computes a single (mean, std) over the UNION   (streaming, O(1) RAM)
#   • builds five DataLoaders:
#         train_loader              # mixed easy+hard, user‑controlled counts
#         val_easy_loader           # only easy  (50 sims)
#         val_hard_loader           # only hard  (50 sims)
#         test_easy_loader          # only easy  (40 sims)
#         test_hard_loader          # only hard  (40 sims)
#
# Called from train.py as:
#   get_data_loaders(
#       config, easy_train=N1, hard_train=N2,
#       easy_path="...NoHole.npy", hard_path="...Hole.npy"
#   )
# -----------------------------------------------------------------------------

from __future__ import annotations
import logging, numpy as np, torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple, List

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ────────────────────────── helper: final‑layout semantics ───────────────────
def channel_layout(C: int):
    if C == 8:   return [0, 1, 2], [5, 6], 7        # [Ux,Uy,P,Re,SDF,x,y,Mask]
    if C == 6:   return [0, 1, 2], [3, 4], 5        # [Ux,Uy,P,x,y,Mask]
    if C == 9:   return [0,1,2,3], [6,7], 8
    phys_last = C - 3
    return list(range(phys_last)), [phys_last, phys_last+1], phys_last+2

# ────────────── Welford streaming mean/std (per physical channel) ────────────
class _RunningStat:
    def __init__(self): self.n = 0 ; self.mu = 0. ; self.m2 = 0.
    def update(self, x):
        x = x.astype(np.float64).ravel()
        n  = x.size
        if n == 0: return
        mu = x.mean()
        m2 = ((x - mu) ** 2).sum()

        delta = mu - self.mu
        tot   = self.n + n
        self.mu += delta * n / tot
        self.m2 += m2 + delta**2 * self.n * n / tot
        self.n   = tot
    @property
    def std(self): return float(np.sqrt(self.m2 / max(self.n, 1)))

# ───────────────────────── preprocessing (reorder + coords) ──────────────────
def _preprocess(sim: np.ndarray, input_dim: int,
                dom: Tuple[float,float]) -> np.ndarray:
    # 1) channel reorder
    if input_dim == 6: sim = sim[..., [0,1,2,4,5,3]]
    elif input_dim == 4: pass
    elif input_dim == 7: sim = sim[..., [0,1,2,3,4,5,6]]
    else: raise ValueError("input_dim must be 4,6,7")

    T,H,W,_ = sim.shape
    xmin,xmax = dom
    xv = np.linspace(xmin, xmax, W, dtype=sim.dtype)
    yv = np.linspace(xmin, xmax, H, dtype=sim.dtype)
    xg, yg = np.meshgrid(xv, yv)
    xp = np.broadcast_to(xg, (T,H,W))[...,None]
    yp = np.broadcast_to(yg, (T,H,W))[...,None]

    if input_dim == 6:
        out = np.concatenate([sim[...,:5], xp, yp, sim[...,5:6]], -1)
        out[...,7] = 1. - out[...,7]                 # invert mask
    elif input_dim == 4:
        out = np.concatenate([sim[...,:3], xp, yp, sim[...,3:4]], -1)
        out[...,5] = 1. - out[...,5]
    else:                                            # input_dim == 7
        out = np.concatenate([sim[...,:6], xp, yp, sim[...,6:7]], -1)

    return out.astype(np.float32)

# ───────────────────────────── Dataset (one file) ────────────────────────────
class _SingleFileDS(Dataset):
    def __init__(self, path: str, input_dim: int, dom_range,
                 phys_stats, coord_stats, teacher_ch: int | None):
        self.mm  = np.load(path, mmap_mode="r")
        self.dim = input_dim
        self.dom = dom_range
        self.phys_stats, self.coord_stats = phys_stats, coord_stats

        tmp = _preprocess(self.mm[0], input_dim, dom_range)
        self.C   = tmp.shape[-1]
        self.phys, self.coord, _ = channel_layout(self.C)
        self.teacher_ch = teacher_ch or len(self.phys)

    def __len__(self): return self.mm.shape[0]

    def __getitem__(self, idx):
        sim  = torch.from_numpy(_preprocess(self.mm[idx], self.dim, self.dom))  # (T,H,W,C)
        seed = sim[0]
        tgt  = sim[1:, :, :, self.phys[:self.teacher_ch]]

        seed = self._norm(seed.unsqueeze(0))[0]
        tgt  = self._norm_tgt(tgt)
        return {"pixel_values": seed, "labels": tgt}

    # ―― helpers ――――――――――――――――――――――――――――――――――――――――――――――――――――――――
    def _norm(self, x):
        x = x.float()
        for ch in self.phys:
            m,s = self.phys_stats[ch]; x[...,ch] = (x[...,ch]-m) / s
        for ch in self.coord:
            lo,hi = self.coord_stats[ch]; x[...,ch]=(x[...,ch]-lo)/(hi-lo+1e-12)
        return x
    def _norm_tgt(self, y):
        y = y.float()
        for k,ch in enumerate(self.phys[:self.teacher_ch]):
            m,s = self.phys_stats[ch]; y[...,k]=(y[...,k]-m)/s
        return y

# ─────────────────────────────── stats over union ────────────────────────────
def _compute_stats(paths: List[str], input_dim:int, dom):
    first = _preprocess(np.load(paths[0], mmap_mode="r")[0], input_dim, dom)
    C = first.shape[-1]
    phys, coord, _ = channel_layout(C)
    RS = {ch:_RunningStat() for ch in phys}

    for p in paths:
        mm = np.load(p, mmap_mode="r")
        for sim in mm:
            proc = _preprocess(sim, input_dim, dom)
            for ch in phys: RS[ch].update(proc[...,ch])

    phys_stats  = {ch:(RS[ch].mu, max(RS[ch].std,1e-12)) for ch in phys}
    coord_stats = {ch:(first[...,ch].min(), first[...,ch].max()) for ch in coord}
    return phys_stats, coord_stats

# ────────────────────────── public DataLoader factory ────────────────────────
def get_data_loaders(*, config:dict,
                     easy_train:int, hard_train:int,
                     easy_path:str, hard_path:str):

    # sanity: easy first, hard second
    paths = [easy_path, hard_path]

    batch      = config["data"]["batch_size"]
    dom_range  = config["data"].get("domain_range", (0,2))
    input_dim  = config["model"]["input_dim"]

    # 1) global mean/std
    phys_stats, coord_stats = _compute_stats(paths, input_dim, dom_range)

    # 2) datasets
    ds_easy = _SingleFileDS(easy_path, input_dim, dom_range,
                            phys_stats, coord_stats, teacher_ch=None)
    ds_hard = _SingleFileDS(hard_path, input_dim, dom_range,
                            phys_stats, coord_stats, teacher_ch=None)

    rng = torch.Generator().manual_seed(42)
    idx_easy = torch.randperm(len(ds_easy), generator=rng).tolist()
    idx_hard = torch.randperm(len(ds_hard), generator=rng).tolist()

    if easy_train > len(idx_easy) or hard_train > len(idx_hard):
        raise ValueError("Requested more train sims than available.")

    # 3) build splits
    train_idx_easy = idx_easy[:easy_train]
    train_idx_hard = idx_hard[:hard_train]

    left_easy = idx_easy[easy_train:]
    left_hard = idx_hard[hard_train:]

    VAL_E, VAL_H, TEST_E, TEST_H = 50, 50, 40, 40
    assert len(left_easy) >= VAL_E+TEST_E and len(left_hard) >= VAL_H+TEST_H, \
        "Not enough sims for val/test splits."

    val_idx_easy,  test_idx_easy  = left_easy[:VAL_E],  left_easy[VAL_E:VAL_E+TEST_E]
    val_idx_hard,  test_idx_hard  = left_hard[:VAL_H],  left_hard[VAL_H:VAL_H+TEST_H]

    # 4) Subsets
    Sub = Subset
    train_ds = Sub(ds_easy, train_idx_easy) + Sub(ds_hard, train_idx_hard)

    val_easy_ds,  val_hard_ds  = Sub(ds_easy, val_idx_easy),  Sub(ds_hard, val_idx_hard)
    test_easy_ds, test_hard_ds = Sub(ds_easy, test_idx_easy), Sub(ds_hard, test_idx_hard)

    # 5) DataLoaders
    def _dl(d, sh): return DataLoader(d, batch_size=batch,
                                      shuffle=sh, num_workers=4, pin_memory=True)
    return (_dl(train_ds, True),
            _dl(val_easy_ds, False),  _dl(val_hard_ds, False),
            _dl(test_easy_ds, False), _dl(test_hard_ds, False))
