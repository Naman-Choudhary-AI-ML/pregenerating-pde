import copy
import logging
import math
import torch
import os

import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn.utils.weight_norm import WeightNorm
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GehringLinear(nn.Linear):
    """A linear layer with Gehring initialization and weight normalization."""

    def __init__(self, in_features, out_features, dropout=0, bias=True,
                 weight_norm=True):
        self.dropout = dropout
        self.weight_norm = weight_norm
        super().__init__(in_features, out_features, bias)

    def reset_parameters(self):
        # One problem with initialization from the uniform distribution is that
        # the distribution of the outputs has a variance that grows with the
        # number of inputs. It turns out that we can normalize the variance of
        # each neuron’s output to 1 by scaling its weight vector by the square
        # root of its fan-in (i.e. its number of inputs). Dropout further
        # increases the variance of each input, so we need to scale down std.
        # See A.3. in Gehring et al (2017): https://arxiv.org/pdf/1705.03122.
        std = math.sqrt((1 - self.dropout) / self.in_features)
        self.weight.data.normal_(mean=0, std=std)
        if self.bias is not None:
            self.bias.data.fill_(0)

        # Weight normalization is a reparameterization that decouples the
        # magnitude of a weight tensor from its direction. See Salimans and
        # Kingma (2016): https://arxiv.org/abs/1602.07868.
        if self.weight_norm:
            nn.utils.weight_norm(self)


class WNLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 wnorm: bool = True,
                 bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.debug        = int(os.getenv("DEBUG_NAN", "0")) == 1
        self.eps          = float(os.getenv("SAFE_WN_EPS", "1e-6"))
        self.wnorm        = wnorm and (int(os.getenv("DISABLE_WN", "0")) == 0)

        if not self.wnorm:
            # Plain Linear
            self.lin = nn.Linear(in_features, out_features, bias=bias)
        else:
            # Safe weight-norm (manual g/v with epsilon)
            # weight_v: (out, in),  weight_g: (out, 1)
            self.weight_v = nn.Parameter(torch.empty(out_features, in_features))
            self.weight_g = nn.Parameter(torch.empty(out_features, 1))
            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter("bias", None)

            # Kaiming init for v, g = ||v|| so that initial weight == v
            nn.init.kaiming_uniform_(self.weight_v, a=math.sqrt(5))
            with torch.no_grad():
                vnorm = self.weight_v.norm(dim=1, keepdim=True).clamp_min(self.eps)
                self.weight_g.copy_(vnorm)

    # ----- helpers -----------------------------------------------------------
    def _make_weight(self) -> torch.Tensor:
        # w = g * v / (||v|| + eps)
        v = self.weight_v
        g = self.weight_g
        vnorm = v.norm(dim=1, keepdim=True).clamp_min(self.eps)
        w = g * (v / vnorm)
        if self.debug:
            if not torch.isfinite(w).all():
                raise RuntimeError("[WNLinear] non-finite weight computed")
        return w

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"wnorm={self.wnorm}, eps={self.eps}")

    # ----- forward -----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.wnorm:
            return self.lin(x)

        w = self._make_weight()
        if self.debug:
            # light‑weight stats (no huge prints)
            if not torch.isfinite(x).all():
                raise RuntimeError("[WNLinear] non-finite input")
            wabs = w.abs().max().item()
            if wabs > 1e6:
                print(f"[WNLinear][WARN] |w|max={wabs:.3e}")
        return F.linear(x, w, self.bias)
