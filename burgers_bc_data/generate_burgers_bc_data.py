"""
Generate training data for 1D Burgers equation with time-dependent Dirichlet boundary conditions.
PDE: u_t + u*u_x = nu*u_xx on (0, L) x (0, T), with u(0,t)=g_left(t), u(L,t)=g_right(t).
Boundary data g_left(t), g_right(t) are sampled from a Gaussian random field (GRF) in time.
Saves in same format as heat_optimal_control_data: bc_type, g_left, g_right, u (solution), x, t.
FNO input: boundary conditions (g_left, g_right). FNO output: solution u(x,t).

Solver follows explicit Euler finite-difference approach (see e.g. neural operator Burgers example):
  u^{n+1} = u^n + dt * (-u^n*u_x^n + nu*u_xx^n), with BCs enforced after each step.
"""
import numpy as np
import os

# Domain (match heat equation setup)
L = 1.0
T = 0.5

# Data generation grid
N_X_DATA = 24
N_T_DATA = 30
NU = 0.01   # viscosity
# Sub-steps per output interval for stability (explicit Euler: need CFL and diffusion limit)
N_SUBSTEPS = 80   # dt_inner small enough for CFL dt <= dx/max|u| and diffusion

# GRF kernel: K(t,t') = sigma^2 * exp(-|t-t'|^2 / (2*ell^2))
GRF_ELL = 0.15
GRF_SIGMA = 1.0

# Fixed initial condition used for all samples (same for every BC)
def u0_burgers(x):
    """Initial condition u(x, 0); same for all boundary condition samples."""
    return np.sin(np.pi * x / L)


def sample_grf_on_grid(t, ell=GRF_ELL, sigma=GRF_SIGMA):
    """Sample one realization of a Gaussian random field on time grid t. Returns array of shape (len(t),)."""
    t = np.asarray(t, dtype=np.float64).ravel()
    n = len(t)
    d = (t[:, None] - t[None, :]) ** 2
    K = (sigma ** 2) * np.exp(-0.5 * d / (ell ** 2))
    K += 1e-8 * np.eye(n)
    L_chol = np.linalg.cholesky(K)
    z = L_chol @ np.random.randn(n)
    return z.astype(np.float32)


def _bc_at_time(t_cur, t_out, g_left_arr, g_right_arr):
    """Linear interpolation of BCs at time t_cur from values on grid t_out."""
    idx = np.searchsorted(t_out, t_cur, side="right") - 1
    idx = max(0, min(idx, len(t_out) - 2))
    s = (t_cur - t_out[idx]) / (t_out[idx + 1] - t_out[idx])
    gl = (1 - s) * g_left_arr[idx] + s * g_left_arr[idx + 1]
    gr = (1 - s) * g_right_arr[idx] + s * g_right_arr[idx + 1]
    return gl, gr


def solve_burgers_1d_fd(g_left_arr, g_right_arr, x, t, nu=NU, n_substeps=N_SUBSTEPS, u0=None):
    """
    Solve u_t + u*u_x = nu*u_xx with Dirichlet BC u(0,t)=g_left(t), u(L,t)=g_right(t).
    Fully explicit Euler in time (as in neural operator Burgers example):
      u^{n+1} = u^n + dt * (-u^n*u_x^n + nu*u_xx^n).
    Central differences for u_x and u_xx; BCs applied after each sub-step.
    Sub-stepping between output times ensures stability (dt_inner small).
    g_left_arr, g_right_arr: (n_t+1,) each. t: output time grid.
    Returns u of shape (n_x+1, n_t+1).
    """
    n = len(x)
    n_t_out = len(t)
    dx = x[1] - x[0]
    dt_out = (t[1] - t[0]) if n_t_out > 1 else 0.0
    dt_inner = dt_out / n_substeps if n_substeps > 0 and dt_out > 0 else dt_out

    u = np.zeros((n, n_t_out))
    u_now = np.zeros(n)
    if u0 is not None:
        u_now[:] = u0
    else:
        u_now[:] = u0_burgers(x)
    # Enforce BC at t=0 (boundaries match g_left(0), g_right(0))
    u_now[0] = g_left_arr[0]
    u_now[-1] = g_right_arr[0]
    u[:, 0] = u_now.copy()

    # CFL / diffusion: keep u_now finite (clip to avoid blow-up from rounding)
    u_max_mag = 10.0
    for k in range(n_t_out - 1):
        t_start = t[k]
        for step in range(n_substeps):
            t_cur = t_start + (step + 0.5) * dt_inner
            gl, gr = _bc_at_time(t_cur, t, g_left_arr, g_right_arr)
            # Upwind difference for u_x (stable): u_x[i] = (u[i]-u[i-1])/dx if u[i]>=0 else (u[i+1]-u[i])/dx
            u_x = np.zeros_like(u_now)
            u_x[1:-1] = np.where(
                u_now[1:-1] >= 0,
                (u_now[1:-1] - u_now[:-2]) / dx,
                (u_now[2:] - u_now[1:-1]) / dx,
            )
            # Second derivative: u_xx[i] = (u[i+1] - 2*u[i] + u[i-1]) / dx^2
            u_xx = np.zeros_like(u_now)
            u_xx[1:-1] = (u_now[2:] - 2.0 * u_now[1:-1] + u_now[:-2]) / (dx ** 2)
            # Explicit Euler: u_t + u*u_x = nu*u_xx  =>  u_new = u + dt*(-u*u_x + nu*u_xx)
            u_now = u_now + dt_inner * (-u_now * u_x + nu * u_xx)
            u_now[0] = gl
            u_now[-1] = gr
            # Clip to avoid overflow (keeps solution bounded)
            np.clip(u_now, -u_max_mag, u_max_mag, out=u_now)
        u[:, k + 1] = u_now.copy()

    return u


def generate_dataset(
    n_samples,
    n_x=N_X_DATA,
    n_t=N_T_DATA,
    nu=NU,
    n_substeps=N_SUBSTEPS,
    seed=42,
    verbose=True,
    ell=GRF_ELL,
    sigma=GRF_SIGMA,
):
    """
    Generate Burgers equation dataset with Dirichlet BCs sampled from GRF in time.
    Returns g_left (n_samples, n_t+1), g_right (n_samples, n_t+1), u (n_samples, n_x+1, n_t+1), x, t.
    """
    np.random.seed(seed)
    x = np.linspace(0, L, n_x + 1)
    t = np.linspace(0, T, n_t + 1)
    t_grid = t

    # Same initial condition for all samples (interior profile; BC at t=0 applied in solver)
    u0_fixed = u0_burgers(x)
    g_left_list = []
    g_right_list = []
    u_list = []
    for i in range(n_samples):
        g_left_arr = sample_grf_on_grid(t_grid, ell=ell, sigma=sigma)
        g_right_arr = sample_grf_on_grid(t_grid, ell=ell, sigma=sigma)
        u_sol = solve_burgers_1d_fd(g_left_arr, g_right_arr, x, t, nu=nu, n_substeps=n_substeps, u0=u0_fixed)
        g_left_list.append(g_left_arr)
        g_right_list.append(g_right_arr)
        u_list.append(u_sol)
        if verbose and (i + 1) % 10 == 0:
            print(f"  Generated sample {i + 1}/{n_samples}")

    g_left = np.stack(g_left_list, axis=0).astype(np.float32)
    g_right = np.stack(g_right_list, axis=0).astype(np.float32)
    u = np.stack(u_list, axis=0).astype(np.float32)
    return g_left, g_right, u, x, t


def main():
    import argparse
    p = argparse.ArgumentParser(description="Generate 1D Burgers data with GRF boundary conditions for FNO.")
    p.add_argument("--samples", type=int, default=50, help="Number of BC samples")
    p.add_argument("--n_x", type=int, default=N_X_DATA)
    p.add_argument("--n_t", type=int, default=N_T_DATA)
    p.add_argument("--nu", type=float, default=NU, help="Viscosity")
    p.add_argument("--n_substeps", type=int, default=N_SUBSTEPS, help="Sub-steps per output interval (stability)")
    p.add_argument("--out", type=str, default="burgers_bc_data.npz", help="Output .npz file (saved in script dir)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ell", type=float, default=GRF_ELL, help="GRF length scale in time")
    p.add_argument("--sigma", type=float, default=GRF_SIGMA, help="GRF std")
    p.add_argument("--plot", action="store_true", help="Plot one example (x,t) heatmap after saving")
    args = p.parse_args()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    print("Generating 1D Burgers data (BC -> solution) for FNO.")
    print(f"  Domain [0, {L}] x [0, {T}], nu={args.nu}, n_x={args.n_x}, n_t={args.n_t}")
    print(f"  GRF ell={args.ell}, sigma={args.sigma}")
    g_left, g_right, u, x, t = generate_dataset(
        args.samples,
        n_x=args.n_x,
        n_t=args.n_t,
        nu=args.nu,
        n_substeps=args.n_substeps,
        verbose=True,
        ell=args.ell,
        sigma=args.sigma,
        seed=args.seed,
    )
    np.savez(
        out_path,
        bc_type="dirichlet",
        g_left=g_left,
        g_right=g_right,
        u=u,
        x=x,
        t=t,
    )
    print(f"Saved {args.samples} samples to {out_path}")
    print(f"  g_left: {g_left.shape}, g_right: {g_right.shape}, u: {u.shape}, x: {x.shape}, t: {t.shape}")

    if args.plot:
        plot_example_heatmap(u, g_left, g_right, x, t, sample_idx=0, out_path=out_path)


def plot_example_heatmap(u, g_left, g_right, x, t, sample_idx=0, out_path=None):
    """Plot one sample u(x,t) as (x,t) heatmap and BCs. u shape (n_samples, n_x+1, n_t+1)."""
    import matplotlib.pyplot as plt
    u_sample = u[sample_idx]  # (n_x+1, n_t+1)
    gl = g_left[sample_idx]
    gr = g_right[sample_idx]
    fig, axes = plt.subplots(2, 1, figsize=(6, 5), height_ratios=[1.5, 0.6])
    # Heatmap: x horizontal, t vertical (t=0 at bottom)
    im = axes[0].imshow(
        u_sample.T,
        extent=[x[0], x[-1], t[0], t[-1]],
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
    )
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    axes[0].set_title(f"Burgers solution u(x,t) — sample {sample_idx}")
    plt.colorbar(im, ax=axes[0], label="u")
    # Boundary conditions over time
    axes[1].plot(t, gl, label="g_left(t)", color="C0")
    axes[1].plot(t, gr, label="g_right(t)", color="C1")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("BC value")
    axes[1].set_title("Boundary conditions")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    if out_path:
        plot_path = os.path.join(os.path.dirname(out_path), "burgers_example_heatmap.png")
        plt.savefig(plot_path, dpi=150)
        print(f"Saved plot to {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
