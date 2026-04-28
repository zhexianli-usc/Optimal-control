"""
Neural operators on (x, t) for RL:
- Actor: causal state field -> full control field a(x, t); action at step n is slice [:, :, n].
- Critic Q: state + broadcast action -> scalar (mean over x,t after EFNO trunk).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .efno import EFNO2d


def causal_u_grid(u_traj: torch.Tensor, n: int) -> torch.Tensor:
    """
    u_traj: (B, nx, nt) full trajectory known up to time n (inclusive).
    Build (B, nx, nt) causal copy: column j shows u(·, min(j, n)).
    """
    B, nx, nt = u_traj.shape
    idx = torch.arange(nt, device=u_traj.device).view(1, 1, nt).expand(B, nx, nt)
    n_t = torch.full((1, 1, nt), n, device=u_traj.device, dtype=torch.long).expand(B, nx, nt)
    take = torch.minimum(idx, n_t)
    return torch.gather(u_traj, 2, take)


def state_tensor(
    u_traj: torch.Tensor,
    n: int,
    x_norm: torch.Tensor,
    t_norm: torch.Tensor,
    nt_total: int,
) -> torch.Tensor:
    """
    Returns (B, 4, nx, nt): [causal_u, x broadcast, t broadcast, normalized_decision_time].
    x_norm: (1, nx, 1), t_norm: (1, 1, nt)
    """
    B = u_traj.shape[0]
    u_c = causal_u_grid(u_traj, n)
    xb = x_norm.expand(B, -1, u_traj.shape[2])
    tb = t_norm.expand(B, u_traj.shape[1], -1)
    n_norm = float(n) / max(nt_total - 1, 1)
    ch_n = torch.full((B, u_traj.shape[1], u_traj.shape[2]), n_norm, device=u_traj.device, dtype=u_traj.dtype)
    return torch.stack([u_c, xb, tb, ch_n], dim=1)


def gather_action_time(a_full: torch.Tensor, n_col: torch.Tensor) -> torch.Tensor:
    """a_full: (B, 1, nx, nt), n_col: (B,) long -> (B, nx)."""
    B, _, nx, nt = a_full.shape
    idx = n_col.clamp(0, nt - 1).long().view(B, 1, 1, 1).expand(B, 1, nx, 1)
    return a_full.gather(3, idx).squeeze(1).squeeze(-1)


def broadcast_action(a_vec: torch.Tensor, nt: int) -> torch.Tensor:
    """a_vec: (B, nx) -> (B, 1, nx, nt) constant along t."""
    return a_vec.unsqueeze(-1).expand(-1, -1, nt).unsqueeze(1)


class ActorEFNO(nn.Module):
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
        self.net = EFNO2d(4, 1, width, nx, nt, k_max, n_layers, use_channel_mlp=use_channel_mlp)

    def forward(self, state_bchw: torch.Tensor) -> torch.Tensor:
        out = self.net(state_bchw)
        return self.a_max * torch.tanh(out)


class CriticQEFNO(nn.Module):
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
        # in_ch=5, out_ch=width (feature maps before pooling), spectral width=width
        self.trunk = EFNO2d(5, width, width, nx, nt, k_max, n_layers, use_channel_mlp=use_channel_mlp)
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
