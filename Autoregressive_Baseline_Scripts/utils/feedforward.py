import torch.nn as nn

from .linear import WNLinear
import os
import torch


class FeedForward(nn.Module):
    def __init__(self,
                 dim: int,
                 factor: int = 4,
                 ff_weight_norm: bool = True,
                 n_ff_layers: int = 2,
                 layer_norm: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        self.debug = int(os.getenv("DEBUG_NAN", "0")) == 1

        hidden = dim * factor
        layers = []
        for i in range(n_ff_layers):
            in_d  = dim if i == 0 else hidden
            out_d = dim if i == n_ff_layers - 1 else hidden
            layers.append(WNLinear(in_d, out_d, wnorm=ff_weight_norm))
            if i < n_ff_layers - 1:
                layers.append(nn.GELU())
                if layer_norm:
                    layers.append(nn.LayerNorm(out_d))
                if dropout and dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if self.debug:
            if not torch.isfinite(y).all():
                raise RuntimeError("[FeedForward] non‑finite output")
        return y
