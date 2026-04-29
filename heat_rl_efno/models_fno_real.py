"""
Actor and critic for heat RL using classical FNO (real frequency) backbones.
State encoding matches `models.py` (4-channel field + optional action broadcast for Q).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .fno2d_real import FNO2dRealFreq
from .models import broadcast_action


class ActorFNORealFreq(nn.Module):
    """Maps state field (4 ch) to control field (1 ch), scaled and tanh-bounded."""

    def __init__(
        self,
        nx: int,
        nt: int,
        k_max: int = 8,
        width: int = 32,
        n_layers: int = 4,
        a_max: float = 2.0,
        use_channel_mlp: bool = True,
    ):
        super().__init__()
        self.nx, self.nt = nx, nt
        self.a_max = a_max
        self.net = FNO2dRealFreq(4, 1, width, nx, nt, k_max, n_layers, use_channel_mlp=use_channel_mlp)

    def forward(self, state_bchw: torch.Tensor) -> torch.Tensor:
        out = self.net(state_bchw)
        return self.a_max * torch.tanh(out)


class CriticQFNORealFreq(nn.Module):
    """Q(s, a): concatenate state (4 ch) with broadcast a (1 ch)."""

    def __init__(
        self,
        nx: int,
        nt: int,
        k_max: int = 8,
        width: int = 32,
        n_layers: int = 4,
        use_channel_mlp: bool = True,
    ):
        super().__init__()
        self.trunk = FNO2dRealFreq(5, width, width, nx, nt, k_max, n_layers, use_channel_mlp=use_channel_mlp)
        self.head = nn.Sequential(
            nn.Conv2d(width, 16, 1),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, state_bchw: torch.Tensor, a_vec: torch.Tensor) -> torch.Tensor:
        a_b = broadcast_action(a_vec, state_bchw.shape[-1])
        x = torch.cat([state_bchw, a_b], dim=1)
        h = self.trunk(x)
        q_local = self.head(h)
        return q_local.mean(dim=(2, 3))
