"""
Open-loop optimal control baseline using gradients through the same Crank–Nicolson
semi-discretization as `HeatEnv`. Backpropagation through the rollout implements the
**discrete adjoint** of the CN scheme (equivalent to an implicit adjoint solve).

Minimizes total cost J = sum_k step_cost(a_k, u^{k+1}) with the same weights as RL
(control L2 + soft state constraint), subject to the discrete heat dynamics.
"""
from __future__ import annotations

import torch

from .heat_env import CNContext, HeatEnvConfig, cn_context_from_cfg, cn_step, step_cost


def differentiable_rollout(
    u0: torch.Tensor,
    a_xt: torch.Tensor,
    ctx: CNContext,
    cfg: HeatEnvConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    u0: (B, nx), a_xt: (B, nx, nt-1) with boundary control rows zero.
    Returns u (B, nx, nt), total_cost (B,).
    """
    B, nx, nt1 = a_xt.shape
    nt = nt1 + 1
    device, dtype = u0.device, u0.dtype
    u = torch.zeros(B, nx, nt, device=device, dtype=dtype)
    u[:, :, 0] = u0
    u[:, 0, :] = 0.0
    u[:, -1, :] = 0.0
    total = torch.zeros(B, device=device, dtype=dtype)
    for k in range(nt - 1):
        ak = a_xt[:, :, k]
        ak = ak.clone()
        ak[:, 0] = 0.0
        ak[:, -1] = 0.0
        u_next = cn_step(u[:, :, k], ak, ctx)
        u[:, :, k + 1] = u_next
        total = total + step_cost(ak, u_next, cfg)
    return u, total


def optimize_open_loop_discrete_adjoint(
    u0: torch.Tensor,
    cfg: HeatEnvConfig,
    ctx: CNContext,
    *,
    n_iters: int = 400,
    lr: float = 0.05,
    init_scale: float = 0.0,
    a_warm_start: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """
    Adam on full open-loop control a(x,t) with same CN dynamics and cost as RL env.

    If ``a_warm_start`` is given (B, nx, nt-1), optimization starts from it (e.g. RL rollout).
    Otherwise starts at zero or small random if ``init_scale`` > 0.

    Returns (a_opt (B,nx,nt-1), u_opt, loss_history).
    """
    B, nx = u0.shape
    nt = cfg.nt
    device = u0.device
    dtype = u0.dtype
    if a_warm_start is not None:
        a = a_warm_start.to(device=device, dtype=dtype).clone().detach().requires_grad_(True)
        with torch.no_grad():
            a[:, 0, :] = 0.0
            a[:, -1, :] = 0.0
    else:
        a = torch.zeros(B, nx, nt - 1, device=device, dtype=dtype, requires_grad=True)
        if init_scale != 0:
            with torch.no_grad():
                a.data[:, 1:-1, :] = init_scale * torch.randn(B, nx - 2, nt - 1, device=device, dtype=dtype)
                a.data[:, 0, :] = 0.0
                a.data[:, -1, :] = 0.0

    opt = torch.optim.Adam([a], lr=lr)
    hist: list[float] = []
    for _ in range(n_iters):
        opt.zero_grad(set_to_none=True)
        _, total = differentiable_rollout(u0, a, ctx, cfg)
        loss = total.mean()
        loss.backward()
        with torch.no_grad():
            a.grad[:, 0, :] = 0.0
            a.grad[:, -1, :] = 0.0
        opt.step()
        with torch.no_grad():
            a[:, 0, :] = 0.0
            a[:, -1, :] = 0.0
        hist.append(float(loss.item()))
    with torch.no_grad():
        u_fin, _ = differentiable_rollout(u0, a, ctx, cfg)
    return a.detach(), u_fin.detach(), hist
