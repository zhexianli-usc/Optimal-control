"""Closed-loop rollouts of the trained EFNO policy (deterministic, no exploration noise)."""
from __future__ import annotations

import torch

from .heat_env import HeatEnv
from .models import gather_action_time, state_tensor


@torch.no_grad()
def rollout_policy(
    actor: torch.nn.Module,
    env: HeatEnv,
    u0: torch.Tensor,
    x_norm: torch.Tensor,
    t_norm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        u_traj: (B, nx, nt)
        a_seq: (B, nx, nt-1) controls used at each step
        total_return: (B,) sum of env.step_reward
    """
    B, nx = u0.shape
    nt = env.cfg.nt
    device = u0.device
    dtype = u0.dtype
    u = torch.zeros(B, nx, nt, device=device, dtype=dtype)
    u[:, :, 0] = u0
    u[:, 0, :] = 0.0
    u[:, -1, :] = 0.0
    a_seq = torch.zeros(B, nx, nt - 1, device=device, dtype=dtype)
    total_r = torch.zeros(B, device=device, dtype=dtype)
    for n in range(nt - 1):
        s = state_tensor(u, n, x_norm, t_norm, nt)
        a_full = actor(s)
        a = gather_action_time(a_full, torch.full((B,), n, device=device, dtype=torch.long))
        a[:, 0] = 0.0
        a[:, -1] = 0.0
        a_seq[:, :, n] = a
        u_next = env.step(u[:, :, n], a)
        u[:, :, n + 1] = u_next
        total_r = total_r + env.step_reward(a, u_next)
    return u, a_seq, total_r


def total_return_to_cost(total_return: torch.Tensor) -> torch.Tensor:
    """Cost J = - sum r (same units as discrete adjoint objective mean)."""
    return -total_return
