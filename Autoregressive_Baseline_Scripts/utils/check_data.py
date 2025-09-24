# scripts/check_data.py
# -----------------------------------------------------------------------------
# Standalone preflight checker for *just the data*. Run this BEFORE training if
# you suspect loader issues. It does not import your model and won’t OOM.
# -----------------------------------------------------------------------------
import os, sys, numpy as np
from data_loader import _sanitize_and_layout, _valid_fraction_seed

def inspect(path: str, max_samples: int = 20):
    mm = np.load(path, mmap_mode="r")
    if mm.ndim == 4: mm = mm[None, ...]
    N = mm.shape[0]
    k = min(max_samples, N)
    print(f"[{os.path.basename(path)}] N={N}, inspecting first {k} samples")
    vf = []
    px_min, px_max = [], []
    any_nonfinite = False
    for i in range(k):
        raw = mm[i]
        vf.append(_valid_fraction_seed(raw))
        out = _sanitize_and_layout(raw)
        px = out[..., :3]
        px_min.append(px.min())
        px_max.append(px.max())
        if not np.isfinite(out).all(): any_nonfinite = True
    print(f"  valid_frac (seed) avg={np.mean(vf):.4f} min={np.min(vf):.4f} max={np.max(vf):.4f}")
    print(f"  U,V,P range over {k} sims: [{np.min(px_min):.3e}, {np.max(px_max):.3e}]")
    print(f"  non‑finite detected after sanitize? {any_nonfinite}")

if __name__ == "__main__":
    easy = sys.argv[1]
    hard = sys.argv[2]
    inspect(easy, max_samples=50)
    inspect(hard, max_samples=50)
