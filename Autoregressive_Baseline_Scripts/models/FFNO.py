# models/FFNO.py
# -----------------------------------------------------------------------------
# Strategic NaN/Inf instrumentation inside SpectralConv2d and FFNO.forward.
# Guarded by attributes:
#   self.debug_nan (bool)         -> set by train.py from $DEBUG_NAN
#   self.debug_threshold (float)  -> warn if |activation| exceeds this
# -----------------------------------------------------------------------------
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from utils.feedforward import FeedForward
from utils.linear      import WNLinear

def _fin_ratio(x: torch.Tensor) -> float:
    x = x.detach()
    return float(torch.isfinite(x).sum().item()) / float(x.numel()) if x.numel() else 1.0

def _stat_str(tag: str, x: torch.Tensor) -> str:
    y = torch.nan_to_num(x.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    return (f"{tag}: min={y.min().item():.3e} max={y.max().item():.3e} "
            f"mean={y.mean().item():.3e} std={y.std().item():.3e} "
            f"finite={_fin_ratio(x):.6f}")

class SpectralConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, modes_x, modes_y, mode="full",
                 n_ff_layers=2, factor=4, dropout=0.1, ff_weight_norm=True,
                 use_fork=False, layer_norm=False, fourier_weight=None,
                 forecast_ff=None, backcast_ff=None):
        super().__init__()
        assert mode in ['no-fourier', 'full', 'low-pass']
        self.in_dim, self.out_dim = in_dim, out_dim
        self.modes_x, self.modes_y = modes_x, modes_y
        self.mode = mode
        self.use_fork = use_fork
        self.fourier_weight = fourier_weight
        self.debug_nan = False
        self.debug_threshold = 1e6

        if not self.fourier_weight:
            self.fourier_weight = nn.ParameterList([])
            for n_modes in [modes_x, modes_y]:
                weight = torch.empty(in_dim, out_dim, n_modes, 2)
                nn.init.xavier_normal_(weight)
                self.fourier_weight.append(nn.Parameter(weight))

        if use_fork:
            self.forecast_ff = forecast_ff or FeedForward(out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)
        self.backcast_ff = backcast_ff or FeedForward(out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

    def _check(self, name, x):
        if not self.debug_nan: return
        if not torch.isfinite(x).all():
            raise RuntimeError(f"[SpectralConv2d] non‑finite after {name}: " + _stat_str(name, x))
        if x.abs().max() > self.debug_threshold:
            # don't crash; just warn once per call-site
            print(f"[SpectralConv2d][WARN] large activation after {name}: " + _stat_str(name, x))

    def forward(self, x):
        if self.mode != 'no-fourier':
            x = self.forward_fourier(x)
        self._check("fourier", x)

        b = self.backcast_ff(x)
        self._check("backcast_ff", b)

        f = self.forecast_ff(x) if self.use_fork else None
        if f is not None: self._check("forecast_ff", f)
        return b, f

    def forward_fourier(self, x):
        # x: [B, M, N, I]
        x = rearrange(x, 'b m n i -> b i m n')  # [B, I, M, N]
        self._check("rearrange_in", x)

        B, I, M, N = x.shape

        # Y dimension
        x_fty = torch.fft.rfft(x, dim=-1, norm='ortho')                 # complex
        self._check("rfft_y", torch.view_as_real(x_fty))

        out_ft = x_fty.new_zeros(B, I, M, N // 2 + 1)                   # complex
        if self.mode == 'full':
            wy = torch.view_as_complex(self.fourier_weight[1])          # [I,O,My]
            out_ft[:, :, :, :self.modes_y] = torch.einsum(
                "bixy,ioy->boxy", x_fty[:, :, :, :self.modes_y], wy)
        elif self.mode == 'low-pass':
            out_ft[:, :, :, :self.modes_y] = x_fty[:, :, :, :self.modes_y]
        xy = torch.fft.irfft(out_ft, n=N, dim=-1, norm='ortho').real    # [B,I,M,N]
        self._check("irfft_y", xy)

        # X dimension
        x_ftx = torch.fft.rfft(x, dim=-2, norm='ortho')
        self._check("rfft_x", torch.view_as_real(x_ftx))

        out_ft = x_ftx.new_zeros(B, I, M // 2 + 1, N)                   # complex
        if self.mode == 'full':
            wx = torch.view_as_complex(self.fourier_weight[0])          # [I,O,Mx]
            out_ft[:, :, :self.modes_x, :] = torch.einsum(
                "bixy,iox->boxy", x_ftx[:, :, :self.modes_x, :], wx)
        elif self.mode == 'low-pass':
            out_ft[:, :, :self.modes_x, :] = x_ftx[:, :, :self.modes_x, :]

        xx = torch.fft.irfft(out_ft, n=M, dim=-2, norm='ortho').real
        self._check("irfft_x", xx)

        x = xx + xy
        self._check("sum_xy", x)

        x = rearrange(x, 'b i m n -> b m n i')                          # [B,M,N,I]
        self._check("rearrange_out", x)
        return x

class FFNO(nn.Module):
    def __init__(self, input_dim, output_dim, modes_x, modes_y, width,
                 n_layers=4, factor=4, n_ff_layers=2, share_weight=True,
                 ff_weight_norm=True, layer_norm=False):
        super().__init__()
        self.padding   = 8
        self.modes_x   = modes_x
        self.modes_y   = modes_y
        self.width     = width
        self.input_dim = input_dim
        self.n_layers  = n_layers
        self.debug_nan = False
        self.debug_threshold = 1e6

        if input_dim != 6:
            raise ValueError("FFNO expects 6 input channels [Ux,Uy,P,Re,SDF,ValidMask]")

        self.in_channels_physical = 5   # Ux, Uy, P, Re, SDF
        self.in_channels_coords   = 2   # (x, y)

        self.in_proj = WNLinear(self.in_channels_physical + self.in_channels_coords,
                                self.width, wnorm=ff_weight_norm)

        self.fourier_weight = None
        if share_weight:
            self.fourier_weight = nn.ParameterList([])
            for n_modes in [modes_x, modes_y]:
                weight = torch.empty(width, width, n_modes, 2)
                nn.init.xavier_normal_(weight)
                self.fourier_weight.append(nn.Parameter(weight))

        self.spectral_layers = nn.ModuleList([])
        for _ in range(n_layers):
            sc = SpectralConv2d(in_dim=width, out_dim=width,
                                modes_x=modes_x, modes_y=modes_y,
                                forecast_ff=None, backcast_ff=None,
                                fourier_weight=self.fourier_weight,
                                factor=factor, ff_weight_norm=ff_weight_norm,
                                n_ff_layers=n_ff_layers, layer_norm=layer_norm,
                                use_fork=False, dropout=0.1, mode='full')
            self.spectral_layers.append(sc)

        self.out = nn.Sequential(
            WNLinear(self.width, 128, wnorm=ff_weight_norm),
            WNLinear(128, output_dim, wnorm=ff_weight_norm)
        )

    def _check(self, name, x):
        if not self.debug_nan: return
        if not torch.isfinite(x).all():
            raise RuntimeError(f"[FFNO] non‑finite after {name} | " + _stat_str(name, x))
        if x.abs().max() > self.debug_threshold:
            print(f"[FFNO][WARN] large activation after {name} | " + _stat_str(name, x))

    def forward(self, x):
        # x: [B, X, Y, 6] = [Ux,Uy,P,Re,SDF,ValidMask]
        if x.shape[-1] != 6:
            raise RuntimeError(f"Expected input with 6 channels [Ux,Uy,P,Re,SDF,ValidMask], got {x.shape}")

        mask     = x[..., -1]
        physical = x[..., :5] * mask.unsqueeze(-1)    # zero holes
        self._check("masking", physical)

        grid = self.get_grid(x.shape, x.device)       # [B,X,Y,2] in [0,1]
        x = torch.cat((physical, grid), dim=-1)       # [B,X,Y,7]
        self._check("concat_grid", x)

        x = self.in_proj(x)                           # [B,X,Y,H]
        self._check("in_proj", x)

        x = x.permute(0, 3, 1, 2)                     # [B,H,X,Y]
        x = F.pad(x, [0, self.padding, 0, self.padding])
        x = x.permute(0, 2, 3, 1)                     # [B,X+pad,Y+pad,H]
        self._check("pad", x)

        for i, layer in enumerate(self.spectral_layers):
            # propagate debug flags into spectral blocks
            if hasattr(layer, "debug_nan"):
                layer.debug_nan = self.debug_nan
                layer.debug_threshold = self.debug_threshold
            b, _ = layer(x)
            self._check(f"spectral_{i}/backcast", b)
            x = x + b
            self._check(f"spectral_{i}/residual", x)

        b = b[..., :-self.padding, :-self.padding, :] # unpad
        self._check("unpad", b)

        output = self.out(b)                          # [B,X,Y,C_out]
        self._check("out", output)

        output = output * mask.unsqueeze(-1)          # mask outside‑fluid
        self._check("mask_out", output)
        return output

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x, device=device, dtype=torch.float32).view(1, size_x, 1, 1).repeat(batchsize, 1, size_y, 1)
        gridy = torch.linspace(0, 1, size_y, device=device, dtype=torch.float32).view(1, 1, size_y, 1).repeat(batchsize, size_x, 1, 1)
        g = torch.cat((gridx, gridy), dim=-1)
        self._check("grid", g)
        return g
