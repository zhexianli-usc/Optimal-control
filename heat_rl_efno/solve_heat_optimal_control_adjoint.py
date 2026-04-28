"""
Numerical optimal control for 1D heat equation via discrete adjoint + L-BFGS-B.

State dynamics (interior points only):
    A u^{k+1} = B u^k + dt * a^k
with Dirichlet boundaries fixed to zero.

Objective (matches heat_env step_cost aggregation):
    J = sum_k [
          w_control   * mean_x(a_k^2)
        + w_constraint* mean_x(ReLU(|u_{k+1}| - u_abs_max)^2)
        + w_tracking  * mean_x((u_{k+1} - u_des(x,t_{k+1}))^2)
    ]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from .heat_env import HeatEnvConfig


@dataclass
class HeatAdjointResult:
    a_opt: np.ndarray  # (nx, nt-1), full grid with zero boundaries
    u_opt: np.ndarray  # (nx, nt)
    objective: float
    success: bool
    message: str
    n_iter: int


def desired_u_np(x: np.ndarray, t_scalar: float, L: float) -> np.ndarray:
    return 16.0 * x * (x - L) * np.sin(np.pi * t_scalar)


def _build_cn_mats(cfg: HeatEnvConfig):
    nx, nt = cfg.nx, cfg.nt
    dx = cfg.L / (nx - 1)
    dt = cfg.T / (nt - 1)
    n = nx - 2
    kappa = cfg.alpha * dt / (2.0 * dx * dx)
    A = diags([-kappa * np.ones(n - 1), (1.0 + 2.0 * kappa) * np.ones(n), -kappa * np.ones(n - 1)], [-1, 0, 1]).tocsr()
    B = diags([kappa * np.ones(n - 1), (1.0 - 2.0 * kappa) * np.ones(n), kappa * np.ones(n - 1)], [-1, 0, 1]).tocsr()
    return A, B, dx, dt


def _constraint_cost_grad(u_int: np.ndarray, u_abs_max: float):
    excess = np.maximum(0.0, np.abs(u_int) - u_abs_max)
    c = np.mean(excess**2)
    # d/d u [ReLU(|u|-umax)^2] = 2*ReLU(|u|-umax)*sign(u)
    g = 2.0 * excess * np.sign(u_int) / u_int.size
    return c, g


def _tracking_cost_grad(u_int: np.ndarray, u_des_int: np.ndarray):
    diff = u_int - u_des_int
    c = np.mean(diff**2)
    g = 2.0 * diff / diff.size
    return c, g


def solve_optimal_control_numerical_adjoint(
    u0_full: np.ndarray,
    cfg: HeatEnvConfig,
    *,
    maxiter: int = 300,
    init_control_full: np.ndarray | None = None,
    verbose: bool = False,
) -> HeatAdjointResult:
    """
    Solve one open-loop optimal control problem for fixed initial state u0(x).
    """
    nx, nt = cfg.nx, cfg.nt
    n = nx - 2
    n_steps = nt - 1
    x = np.linspace(0.0, cfg.L, nx)
    t = np.linspace(0.0, cfg.T, nt)

    A, B, dx, dt = _build_cn_mats(cfg)

    u0 = np.asarray(u0_full, dtype=np.float64).copy()
    u0[0] = 0.0
    u0[-1] = 0.0
    u0_int = u0[1:-1]

    if init_control_full is None:
        a0_int = np.zeros((n, n_steps), dtype=np.float64)
    else:
        a0 = np.asarray(init_control_full, dtype=np.float64)
        a0_int = a0[1:-1, :n_steps]

    def forward(a_int: np.ndarray):
        u_int = np.zeros((n, nt), dtype=np.float64)
        u_int[:, 0] = u0_int
        for k in range(n_steps):
            rhs = B @ u_int[:, k] + dt * a_int[:, k]
            u_int[:, k + 1] = spsolve(A, rhs)
        return u_int

    def obj_and_grad(a_flat: np.ndarray):
        a_int = a_flat.reshape(n, n_steps)
        u_int = forward(a_int)

        J = 0.0
        # g_u[:,k] = dJ/du_k, only k>=1 have stage costs
        g_u = np.zeros((n, nt), dtype=np.float64)
        g_a = np.zeros((n, n_steps), dtype=np.float64)

        for k in range(n_steps):
            uk1 = u_int[:, k + 1]
            ak = a_int[:, k]
            u_des = desired_u_np(x, float(t[k + 1]), cfg.L)[1:-1]

            c_ctrl = cfg.w_control * np.mean(ak**2)
            g_a[:, k] += cfg.w_control * (2.0 * ak / ak.size)

            c_con, g_con = _constraint_cost_grad(uk1, cfg.u_abs_max)
            c_tr, g_tr = _tracking_cost_grad(uk1, u_des)
            J += c_ctrl + cfg.w_constraint * c_con + cfg.w_tracking * c_tr
            g_u[:, k + 1] += cfg.w_constraint * g_con + cfg.w_tracking * g_tr

        # Discrete adjoint for A u_{k+1} - B u_k - dt a_k = 0
        p = np.zeros((n, nt), dtype=np.float64)
        # Terminal k = n_steps: A^T p_N = -g_u_N
        p[:, n_steps] = spsolve(A.T, -g_u[:, n_steps])
        # Backward for k = n_steps-1 ... 1
        for k in range(n_steps - 1, 0, -1):
            rhs = B.T @ p[:, k + 1] - g_u[:, k]
            p[:, k] = spsolve(A.T, rhs)

        # Grad wrt controls: dJ/da_k = stage_grad - dt * p_{k+1}
        for k in range(n_steps):
            g_a[:, k] += -dt * p[:, k + 1]

        return float(J), g_a.ravel()

    def fun(a_flat):
        J, _ = obj_and_grad(a_flat)
        return J

    def jac(a_flat):
        _, g = obj_and_grad(a_flat)
        return g

    res = minimize(
        fun,
        a0_int.ravel(),
        jac=jac,
        method="L-BFGS-B",
        options={"maxiter": int(maxiter), "ftol": 1e-10, "gtol": 1e-6},
    )

    a_int_opt = res.x.reshape(n, n_steps)
    u_int_opt = forward(a_int_opt)

    a_full = np.zeros((nx, n_steps), dtype=np.float32)
    a_full[1:-1, :] = a_int_opt.astype(np.float32)
    u_full = np.zeros((nx, nt), dtype=np.float32)
    u_full[1:-1, :] = u_int_opt.astype(np.float32)
    u_full[0, :] = 0.0
    u_full[-1, :] = 0.0

    if verbose:
        print(f"  Numerical adjoint OC: J={res.fun:.6e}, iters={getattr(res, 'nit', -1)}, success={res.success}")

    return HeatAdjointResult(
        a_opt=a_full,
        u_opt=u_full,
        objective=float(res.fun),
        success=bool(res.success),
        message=str(res.message),
        n_iter=int(getattr(res, "nit", -1)),
    )
