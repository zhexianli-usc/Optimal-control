"""
1D heat equation environment: u_t = alpha * u_xx + a(x, t) on [0, L] x [0, T]
with Dirichlet u(0,t)=u(L,t)=0. Additive control a is applied as a forcing term.

Semi-discrete Crank–Nicolson for diffusion; control is added explicitly on the RHS.
Used for RL rollouts (no autograd through the solver by default).

`HeatEnvConfig` is the single source of default values for domain, discretization,
diffusivity, and reward weights. Training scripts should take CLI defaults from
`heat_env_defaults()` and build the live config with `HeatEnvConfig.from_rl_args()`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class HeatEnvConfig:
    L: float = 1.0
    T: float = 0.5
    alpha: float = 0.1
    nx: int = 20
    nt: int = 10
    """Spatial grid including boundaries; Nt time slices including t=0."""
    u_abs_max: float = 0.5
    """Soft constraint: penalize |u| above this threshold (interior)."""
    w_control: float = 1e-3
    w_constraint: float = 0.0
    """Weight on mean squared ReLU(|u| - u_abs_max)."""
    w_tracking: float = 1.0
    """Weight on mean squared tracking error to u_desired."""

    @classmethod
    def from_rl_args(cls, args: Any) -> HeatEnvConfig:
        """Build config from argparse Namespace (or any object with matching fields)."""
        return cls(
            L=float(args.L),
            T=float(args.T),
            alpha=float(args.alpha),
            nx=int(args.nx),
            nt=int(args.nt),
            u_abs_max=float(args.u_max),
            w_control=float(args.w_control),
            w_constraint=float(args.w_constraint),
            w_tracking=float(args.w_tracking),
        )


def heat_env_defaults() -> HeatEnvConfig:
    """Fresh default config; use for argparse defaults so they stay in sync."""
    return HeatEnvConfig()


@dataclass
class CNContext:
    """Crank–Nicolson matrices and scalars for one heat discretization (matches `HeatEnv`)."""

    A_cn: torch.Tensor
    main_r: torch.Tensor
    off_r: torch.Tensor
    dt: torch.Tensor
    nx: int
    n_int: int


def cn_context_from_cfg(cfg: HeatEnvConfig, device: torch.device, dtype: torch.dtype = torch.float32) -> CNContext:
    dx = cfg.L / (cfg.nx - 1)
    dt = cfg.T / (cfg.nt - 1)
    n_int = cfg.nx - 2
    kappa = cfg.alpha * dt / (2.0 * dx * dx)
    main = 1.0 + 2.0 * kappa
    off = -kappa
    A = torch.zeros(n_int, n_int, device=device, dtype=dtype)
    for i in range(n_int):
        A[i, i] = main
        if i > 0:
            A[i, i - 1] = off
        if i < n_int - 1:
            A[i, i + 1] = off
    main_r = torch.tensor(1.0 - 2.0 * kappa, device=device, dtype=dtype)
    off_r = torch.tensor(kappa, device=device, dtype=dtype)
    return CNContext(A_cn=A, main_r=main_r, off_r=off_r, dt=torch.tensor(dt, device=device, dtype=dtype), nx=cfg.nx, n_int=n_int)


def cn_step(u_n: torch.Tensor, a_n: torch.Tensor, ctx: CNContext) -> torch.Tensor:
    """
    One CN step; differentiable in u_n, a_n. u_n, a_n: (B, nx).
    Dirichlet: u_n[:,0] and u_n[:,-1] assumed zero; a_n boundaries forced zero in caller.
    """
    B, nx = u_n.shape
    n_int = ctx.n_int
    u_int = u_n[:, 1:nx - 1]
    rhs = torch.zeros_like(u_int)
    for i in range(n_int):
        gp = i + 1
        u_m1 = u_n[:, gp - 1]
        u_0 = u_n[:, gp]
        u_p1 = u_n[:, gp + 1]
        lap = ctx.off_r * (u_m1 + u_p1) + ctx.main_r * u_0
        rhs[:, i] = lap + ctx.dt * a_n[:, gp]
    u_next_int = torch.linalg.solve(ctx.A_cn.unsqueeze(0).expand(B, -1, -1), rhs.unsqueeze(-1)).squeeze(-1)
    out = u_n.clone()
    out[:, 1 : nx - 1] = u_next_int
    out[:, 0] = 0.0
    out[:, -1] = 0.0
    return out


def interior_mask_u(u: torch.Tensor) -> torch.Tensor:
    """Interior columns nx-2 for (B, nx)."""
    return u[:, 1:-1]


def constraint_penalty_from_u(u: torch.Tensor, u_abs_max: float) -> torch.Tensor:
    """(B,) mean squared ReLU(|u|-u_abs_max) on interior."""
    u_int = interior_mask_u(u).abs()
    excess = torch.relu(u_int - u_abs_max)
    return (excess**2).mean(dim=1)


def desired_u_profile(x: torch.Tensor, t_scalar: torch.Tensor | float, L: float) -> torch.Tensor:
    """u_desired(x,t) = 16 * x * (x-L) * sin(pi * t), shape-compatible with x."""
    if not torch.is_tensor(t_scalar):
        t_scalar = torch.tensor(t_scalar, dtype=x.dtype, device=x.device)
    return 16.0 * x * (x - L) * torch.sin(torch.pi * t_scalar)


def tracking_cost_from_u(
    u_next: torch.Tensor,
    t_next: torch.Tensor | float,
    cfg: HeatEnvConfig,
) -> torch.Tensor:
    """(B,) mean squared tracking error to desired profile on interior."""
    nx = u_next.shape[1]
    x = torch.linspace(0.0, cfg.L, nx, device=u_next.device, dtype=u_next.dtype)
    u_des = desired_u_profile(x, t_next, cfg.L).unsqueeze(0).expand_as(u_next)
    err = interior_mask_u(u_next - u_des)
    return (err**2).mean(dim=1)


def step_cost(
    a_n: torch.Tensor,
    u_next: torch.Tensor,
    cfg: HeatEnvConfig,
    t_next: torch.Tensor | float,
) -> torch.Tensor:
    """Scalar cost per step (B,): control + state-constraint + desired-state tracking."""
    a_int = interior_mask_u(a_n)
    c_ctrl = (a_int**2).mean(dim=1) * cfg.w_control
    c_pen = constraint_penalty_from_u(u_next, cfg.u_abs_max) * cfg.w_constraint
    c_track = tracking_cost_from_u(u_next, t_next, cfg) * cfg.w_tracking
    return c_ctrl + c_pen + c_track


class HeatEnv:
    """
    Batched heat solver. State at time index n is u[:, n] with shape (B, nx).
    One transition applies control a_n over [t_n, t_{n+1}] (piecewise constant in time).
    """

    def __init__(self, cfg: HeatEnvConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.dx = cfg.L / (cfg.nx - 1)
        self.dt = cfg.T / (cfg.nt - 1)
        self.x = torch.linspace(0.0, cfg.L, cfg.nx, device=device, dtype=torch.float32)
        self.t = torch.linspace(0.0, cfg.T, cfg.nt, device=device, dtype=torch.float32)
        self._ctx = cn_context_from_cfg(cfg, device, dtype=torch.float32)
        self.A_cn = self._ctx.A_cn
        self.main_r = float(self._ctx.main_r.item())
        self.off_r = float(self._ctx.off_r.item())
        self.n_int = self._ctx.n_int

    def _interior(self, u):
        return u[:, 1 : self.cfg.nx - 1]

    def step(self, u_n: torch.Tensor, a_n: torch.Tensor) -> torch.Tensor:
        """
        u_n, a_n: (B, nx) with boundary rows of u_n zero; a_n should be zero at boundaries.
        Returns u_{n+1}.
        """
        assert u_n.shape[1] == self.cfg.nx
        return cn_step(u_n, a_n, self._ctx)

    def constraint_penalty(self, u: torch.Tensor) -> torch.Tensor:
        """Mean squared violation of |u| <= u_abs_max on interior (B,)."""
        return constraint_penalty_from_u(u, self.cfg.u_abs_max)

    def step_reward(self, a_n: torch.Tensor, u_next: torch.Tensor, t_next: torch.Tensor | float) -> torch.Tensor:
        """Per-step reward (B,): negative of step_cost at time t_next."""
        return -step_cost(a_n, u_next, self.cfg, t_next)

    def rollout(self, u0: torch.Tensor, a_xt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        u0: (B, nx) Dirichlet-compatible.
        a_xt: (B, nx, nt-1) control at each subinterval [t_k, t_{k+1}] (or pad nt with last ignored).
        Returns u_traj (B, nx, nt), total_reward (B,) sum of step rewards.
        """
        B = u0.shape[0]
        nt = self.cfg.nt
        u = torch.zeros(B, self.cfg.nx, nt, device=self.device, dtype=u0.dtype)
        u[:, :, 0] = u0
        u[:, 0, :] = 0.0
        u[:, -1, :] = 0.0
        total_r = torch.zeros(B, device=self.device, dtype=u0.dtype)
        for k in range(nt - 1):
            a_k = a_xt[:, :, k]
            a_k = a_k.clone()
            a_k[:, 0] = 0.0
            a_k[:, -1] = 0.0
            u_k = u[:, :, k]
            u_next = self.step(u_k, a_k)
            u[:, :, k + 1] = u_next
            total_r = total_r + self.step_reward(a_k, u_next, self.t[k + 1])
        return u, total_r
