"""
Generate optimal control data for 1D viscous Burgers equation with time-dependent
Dirichlet boundary conditions and distributed control m(x,t).

PDE:  u_t + u*u_x = nu*u_xx + m   on (0,L)x(0,T)
BCs:  u(0,t)=g_left(t), u(L,t)=g_right(t)
IC:   u(x,0) = sin(pi*x/L) with corners set to BC at t=0

Objective (same structure as heat optimal control):
  J = 0.5 * dx*dt * sum_{k,i} (u_{i,k} - u_{d,i,k})^2 + 0.5 * alpha * dx*dt * sum m_{i,k}^2

Forward (semi-implicit diffusion, explicit upwind advection — stable with moderate dt):
  (I - dt*nu*L) u^{k+1} = u^k + dt * ( -F(u^k) + m^{k+1} ),
  where F_i = u_i * (u_x)_i with first-order upwind on interior; F_0=F_{n-1}=0.

Adjoint (discretize-then-differentiate consistent with forward; cf. adjoint of conservation
laws / [Fikl et al., arxiv:2209.03270](https://arxiv.org/pdf/2209.03270) for discrete adjoint
principles):
  M^T p^k = p^{k+1} + dt * (u^k - u_d^k) + dt * J_F(u^k)^T p^{k+1},
  with M = I - dt*nu*L, p^N = 0, p=0 at Dirichlet boundaries.

Gradient w.r.t. control (matches heat):
  grad_m = (p + alpha * m) * dx * dt

Saves same format as heat_optimal_control_data.npz: bc_type, g_left, g_right, m_opts, x, t.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from heat_1d_numpy_scipy import (
    L,
    T,
    build_laplacian_1d,
    apply_bc_to_M,
    u_desired,
)

# Grid (match burgers_bc_data / heat data generation)
N_X_DATA = 24
N_T_DATA = 30
NU = 0.01
ALPHA_DATA = 0.5

GRF_ELL = 0.15
GRF_SIGMA = 1.0


def u0_burgers(x):
    return np.sin(np.pi * x / L)


def sample_grf_on_grid(t, ell=GRF_ELL, sigma=GRF_SIGMA):
    t = np.asarray(t, dtype=np.float64).ravel()
    n = len(t)
    d = (t[:, None] - t[None, :]) ** 2
    K = (sigma ** 2) * np.exp(-0.5 * d / (ell ** 2))
    K += 1e-8 * np.eye(n)
    L_chol = np.linalg.cholesky(K)
    z = L_chol @ np.random.randn(n)
    return z.astype(np.float32)


def upwind_advection_F(u, dx):
    """
    F_i = u_i * (u_x)_i with first-order upwind; interior i=1..n-2.
    F_0 = F_{n-1} = 0.
    """
    n = len(u)
    F = np.zeros_like(u)
    for i in range(1, n - 1):
        if u[i] >= 0.0:
            u_x = (u[i] - u[i - 1]) / dx
        else:
            u_x = (u[i + 1] - u[i]) / dx
        F[i] = u[i] * u_x
    return F


def advection_jacobian_transpose_apply(u, p, dx):
    """
    v = J_F(u)^T p where F is upwind advection F_i = u_i * (u_x)_i (interior).
    Boundaries v[0]=v[-1]=0.
    """
    n = len(u)
    v = np.zeros(n)
    for i in range(1, n - 1):
        pi = p[i]
        if pi == 0.0:
            continue
        if u[i] >= 0.0:
            # F_i = u_i * (u_i - u_{i-1}) / dx
            v[i - 1] += pi * (u[i] / dx)
            v[i] += pi * (-(2.0 * u[i] - u[i - 1]) / dx)
        else:
            # F_i = u_i * (u_{i+1} - u_i) / dx
            v[i] += pi * (-(u[i + 1] - 2.0 * u[i]) / dx)
            v[i + 1] += pi * (-u[i] / dx)
    v[0] = 0.0
    v[-1] = 0.0
    return v


def build_diffusion_operator_M(n_x, dx, dt, nu, bc_left, bc_right):
    """M = I - dt*nu*L with Dirichlet BC rows on Laplacian."""
    n = n_x + 1
    A = build_laplacian_1d(n_x, dx, bc_left, bc_right)
    I = diags(np.ones(n), 0).tocsr()
    M = (I - dt * nu * A).tolil()
    apply_bc_to_M(M, bc_left, bc_right)
    M[0, :] = 0.0
    M[0, 0] = 1.0
    M[-1, :] = 0.0
    M[-1, -1] = 1.0
    return M.tocsr()


def solve_forward_burgers_oc(m, x, t, M, dx, dt, nu, g_left_arr, g_right_arr, u0_func):
    """
    Semi-implicit: M u^{k+1} = u^k + dt * (-F(u^k) + m^{k+1}).
    m, u: shape (n, n_t+1).
    """
    n = len(x)
    n_t_steps = len(t) - 1
    u = np.zeros((n, n_t_steps + 1))
    u[:, 0] = u0_func(x)
    u[0, 0] = g_left_arr[0]
    u[-1, 0] = g_right_arr[0]

    for k in range(n_t_steps):
        Fk = upwind_advection_F(u[:, k], dx)
        rhs = u[:, k] + dt * (-Fk + m[:, k + 1])
        rhs[0] = g_left_arr[k + 1]
        rhs[-1] = g_right_arr[k + 1]
        u[:, k + 1] = spsolve(M, rhs)

    return u


def objective_and_gradient_burgers_oc(
    m_flat, x, t, M, dx, dt, nu, u_d, g_left_arr, g_right_arr, u0_func, alpha
):
    n = len(x)
    n_t_steps = len(t) - 1
    m = m_flat.reshape(n, n_t_steps + 1)

    u = solve_forward_burgers_oc(m, x, t, M, dx, dt, nu, g_left_arr, g_right_arr, u0_func)

    diff = u - u_d
    J_tracking = 0.5 * dx * dt * np.sum(diff ** 2)
    J_reg = 0.5 * alpha * dx * dt * np.sum(m ** 2)
    J = J_tracking + J_reg

    p = np.zeros((n, n_t_steps + 1))
    p[:, -1] = 0.0

    for k in range(n_t_steps - 1, -1, -1):
        adj_src = advection_jacobian_transpose_apply(u[:, k], p[:, k + 1], dx)
        rhs = p[:, k + 1] + dt * (u[:, k] - u_d[:, k]) + dt * adj_src
        rhs[0] = 0.0
        rhs[-1] = 0.0
        p[:, k] = spsolve(M.T, rhs)

    grad = (p + alpha * m) * (dx * dt)
    return J, grad.ravel()


def solve_optimal_control_one_burgers(g_left_arr, g_right_arr, n_x, n_t, alpha, nu=NU, verbose=False):
    dx = L / n_x
    dt = T / n_t
    x = np.linspace(0, L, n_x + 1)
    t = np.linspace(0, T, n_t + 1)
    X, T_grid = np.meshgrid(x, t, indexing="ij")
    u_d = u_desired(X, T_grid)

    bc_left = ("dirichlet", 0.0)
    bc_right = ("dirichlet", 0.0)
    M = build_diffusion_operator_M(n_x, dx, dt, nu, bc_left, bc_right)

    n = n_x + 1
    n_t_pts = n_t + 1
    m0 = np.random.randn(n * n_t_pts) * 0.01

    def J_func(m_flat):
        J, _ = objective_and_gradient_burgers_oc(
            m_flat, x, t, M, dx, dt, nu, u_d, g_left_arr, g_right_arr, u0_burgers, alpha
        )
        return J

    def grad_func(m_flat):
        _, g = objective_and_gradient_burgers_oc(
            m_flat, x, t, M, dx, dt, nu, u_d, g_left_arr, g_right_arr, u0_burgers, alpha
        )
        return g

    result = minimize(
        J_func,
        m0,
        method="L-BFGS-B",
        jac=grad_func,
        bounds=None,
        options={"maxiter": 2000, "ftol": 1e-9, "gtol": 1e-6},
    )
    m_opt = result.x.reshape(n, n_t_pts)
    if verbose:
        print(f"  Burgers OC -> J={result.fun:.2e}, success={result.success}")
    return x, t, m_opt


def generate_dataset(
    n_samples,
    n_x=N_X_DATA,
    n_t=N_T_DATA,
    nu=NU,
    alpha=ALPHA_DATA,
    seed=42,
    verbose=True,
    ell=GRF_ELL,
    sigma=GRF_SIGMA,
):
    np.random.seed(seed)
    t_grid = np.linspace(0, T, n_t + 1)
    g_left_list = []
    g_right_list = []
    m_opts = []
    x_ref, t_ref = None, None

    for i in range(n_samples):
        g_left_arr = sample_grf_on_grid(t_grid, ell=ell, sigma=sigma)
        g_right_arr = sample_grf_on_grid(t_grid, ell=ell, sigma=sigma)
        x, t, m_opt = solve_optimal_control_one_burgers(
            g_left_arr, g_right_arr, n_x, n_t, alpha, nu=nu, verbose=(verbose and i % max(1, n_samples // 5) == 0)
        )
        g_left_list.append(g_left_arr)
        g_right_list.append(g_right_arr)
        m_opts.append(m_opt)
        if x_ref is None:
            x_ref, t_ref = x, t

    g_left = np.stack(g_left_list, axis=0).astype(np.float32)
    g_right = np.stack(g_right_list, axis=0).astype(np.float32)
    m_opts = np.stack(m_opts, axis=0).astype(np.float32)
    return g_left, g_right, m_opts, x_ref, t_ref


def plot_verification_example(
    g_left, g_right, m_opts, x, t, sample_idx=0, out_path=None, nu=NU
):
    """Plot optimal m, forward state u with m_opt, desired u_d, and tracking error."""
    import matplotlib.pyplot as plt

    gl = np.asarray(g_left[sample_idx], dtype=np.float64)
    gr = np.asarray(g_right[sample_idx], dtype=np.float64)
    m = np.asarray(m_opts[sample_idx], dtype=np.float64)

    dx = x[1] - x[0]
    dt = t[1] - t[0]
    n_x = len(x) - 1
    bc_left = ("dirichlet", 0.0)
    bc_right = ("dirichlet", 0.0)
    M = build_diffusion_operator_M(n_x, dx, dt, nu, bc_left, bc_right)
    X, Tg = np.meshgrid(x, t, indexing="ij")
    u_d = u_desired(X, Tg)

    u = solve_forward_burgers_oc(m, x, t, M, dx, dt, nu, gl, gr, u0_burgers)
    err = u - u_d

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    im0 = axes[0, 0].imshow(
        m.T, extent=[x[0], x[-1], t[0], t[-1]], aspect="auto", origin="lower", cmap="viridis"
    )
    axes[0, 0].set_title(f"Optimal control m(x,t) — sample {sample_idx}")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("t")
    plt.colorbar(im0, ax=axes[0, 0], label="m")

    im1 = axes[0, 1].imshow(
        u.T, extent=[x[0], x[-1], t[0], t[-1]], aspect="auto", origin="lower", cmap="RdBu_r"
    )
    axes[0, 1].set_title("State u(x,t) with m_opt")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("t")
    plt.colorbar(im1, ax=axes[0, 1], label="u")

    im2 = axes[1, 0].imshow(
        u_d.T, extent=[x[0], x[-1], t[0], t[-1]], aspect="auto", origin="lower", cmap="RdBu_r"
    )
    axes[1, 0].set_title("Desired u_d(x,t)")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("t")
    plt.colorbar(im2, ax=axes[1, 0], label="u_d")

    im3 = axes[1, 1].imshow(
        err.T, extent=[x[0], x[-1], t[0], t[-1]], aspect="auto", origin="lower", cmap="RdBu_r"
    )
    axes[1, 1].set_title("Tracking error u - u_d")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("t")
    plt.colorbar(im3, ax=axes[1, 1], label="error")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved verification figure to {out_path}")
    plt.show()


def main():
    import argparse

    p = argparse.ArgumentParser(description="Burgers optimal control data (adjoint + L-BFGS-B).")
    p.add_argument("--samples", type=int, default=50)
    p.add_argument("--n_x", type=int, default=N_X_DATA)
    p.add_argument("--n_t", type=int, default=N_T_DATA)
    p.add_argument("--nu", type=float, default=NU)
    p.add_argument("--alpha", type=float, default=ALPHA_DATA)
    p.add_argument("--out", type=str, default="burgers_optimal_control_data.npz")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ell", type=float, default=GRF_ELL)
    p.add_argument("--sigma", type=float, default=GRF_SIGMA)
    p.add_argument("--plot", action="store_true", help="Plot one example after saving")
    args = p.parse_args()

    out_path = os.path.join(SCRIPT_DIR, args.out)
    print("Burgers optimal control data (semi-implicit FD + discrete adjoint).")
    print(f"  nu={args.nu}, alpha={args.alpha}, domain [0,{L}]x[0,{T}]")
    print("  Ref: discrete adjoint (arxiv:2209.03270)")

    g_left, g_right, m_opts, x, t = generate_dataset(
        args.samples,
        n_x=args.n_x,
        n_t=args.n_t,
        nu=args.nu,
        alpha=args.alpha,
        seed=args.seed,
        verbose=True,
        ell=args.ell,
        sigma=args.sigma,
    )
    # Compatibility mode: save control under both keys:
    # - m_opts: semantic key for optimal control
    # - u: key expected by train_fno_burgers_bc.py
    np.savez(
        out_path,
        bc_type="dirichlet",
        g_left=g_left,
        g_right=g_right,
        u=m_opts,
        m_opts=m_opts,
        x=x,
        t=t,
        nu=args.nu,
        alpha=args.alpha,
        equation="burgers_viscous_oc",
    )
    print(f"Saved {args.samples} samples to {out_path}")
    print(f"  g_left {g_left.shape}, u/m_opts {m_opts.shape}")

    if args.plot:
        fig_path = os.path.join(SCRIPT_DIR, "burgers_optimal_control_verification.png")
        plot_verification_example(
            g_left, g_right, m_opts, x, t, sample_idx=0, out_path=fig_path, nu=args.nu
        )


if __name__ == "__main__":
    main()
