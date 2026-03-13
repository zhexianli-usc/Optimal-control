"""
Generate optimal control data for 1D heat equation with time-dependent boundary conditions.
Boundary data g_left(t), g_right(t) are sampled from a Gaussian random field (GRF) in time.
Dataset is either all Dirichlet or all Neumann (no mixing).
Saves: bc_type, g_left, g_right, m_opts, x, t.
"""
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize

from heat_1d_numpy_scipy import (
    L, T,
    build_laplacian_1d,
    apply_bc_to_M,
    u0,
    u_desired,
)

# Data generation grid (coarser = faster)
N_X_DATA = 24
N_T_DATA = 30
ALPHA_DATA = 0.5

# GRF kernel: K(t,t') = sigma^2 * exp(-|t-t'|^2 / (2*ell^2))
GRF_ELL = 0.2
GRF_SIGMA = 2


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


def solve_forward_tdep(m, x, t, A, dx, dt, g_left_arr, g_right_arr, bc_type, u0_func):
    """
    Solve u_t = u_xx + m with time-dependent BC.
    g_left_arr, g_right_arr: arrays of length len(t). bc_type: "dirichlet" or "neumann".
    """
    n = len(x)
    n_t_steps = len(t) - 1
    u = np.zeros((n, n_t_steps + 1))
    u[:, 0] = u0_func(x)

    I = diags(np.ones(n), 0).tocsr()
    M = (I - dt * A).tolil()
    apply_bc_to_M(M, (bc_type, 0.0), (bc_type, 0.0))
    M = M.tocsr()

    for k in range(n_t_steps):
        rhs = u[:, k] + dt * m[:, k + 1]
        if bc_type == "dirichlet":
            rhs[0] = g_left_arr[k + 1]
            rhs[-1] = g_right_arr[k + 1]
        else:
            rhs[0] = -dx * g_left_arr[k + 1]
            rhs[-1] = dx * g_right_arr[k + 1]
        u[:, k + 1] = spsolve(M, rhs)

    return u


def objective_and_gradient_tdep(m_flat, x, t, A, dx, dt, u_d, g_left_arr, g_right_arr, bc_type, u0_func, alpha):
    """J(m) and gradient for time-dependent BC. Returns (J, grad_flat)."""
    n = len(x)
    n_t_steps = len(t) - 1
    m = m_flat.reshape(n, n_t_steps + 1)

    u = solve_forward_tdep(m, x, t, A, dx, dt, g_left_arr, g_right_arr, bc_type, u0_func)

    diff = u - u_d
    J_tracking = 0.5 * dx * dt * np.sum(diff ** 2)
    J_reg = 0.5 * alpha * dx * dt * np.sum(m ** 2)
    J = J_tracking + J_reg

    # Adjoint: same structure, BC values zero for adjoint
    I = diags(np.ones(n), 0).tocsr()
    M = (I - dt * A).tolil()
    apply_bc_to_M(M, (bc_type, 0.0), (bc_type, 0.0))
    M = M.tocsr()

    lam = np.zeros((n, n_t_steps + 1))
    lam[:, -1] = 0.0
    for k in range(n_t_steps - 1, -1, -1):
        rhs = lam[:, k + 1] + dt * (u[:, k] - u_d[:, k])
        rhs[0] = 0.0
        rhs[-1] = 0.0
        lam[:, k] = spsolve(M.T, rhs)

    grad = (lam + alpha * m) * (dx * dt)
    return J, grad.ravel()


def solve_optimal_control_one_tdep(g_left_arr, g_right_arr, bc_type, n_x, n_t, alpha, verbose=False):
    """
    Solve optimal control for one time-dependent BC (Dirichlet or Neumann).
    g_left_arr, g_right_arr: (n_t+1,) each. bc_type: "dirichlet" or "neumann".
    Returns (x, t, m_opt).
    """
    dx = L / n_x
    dt = T / n_t
    x = np.linspace(0, L, n_x + 1)
    t = np.linspace(0, T, n_t + 1)
    X, T_grid = np.meshgrid(x, t, indexing="ij")
    u_d = u_desired(X, T_grid)

    bc_left = (bc_type, 0.0)
    bc_right = (bc_type, 0.0)
    A = build_laplacian_1d(n_x, dx, bc_left, bc_right)

    n = n_x + 1
    n_t_pts = n_t + 1
    m0 = np.random.rand(n * n_t_pts)

    last_J = [None]
    step = [0]
    def callback(xk):
        step[0] += 1
        if last_J[0] is not None:
            print(f"    step {step[0]:4d}: J = {last_J[0]:.6e}")

    def J_func(m_flat):
        J, _ = objective_and_gradient_tdep(
            m_flat, x, t, A, dx, dt, u_d, g_left_arr, g_right_arr, bc_type, u0_func=u0, alpha=alpha
        )
        last_J[0] = J
        return J

    def grad_func(m_flat):
        _, g = objective_and_gradient_tdep(
            m_flat, x, t, A, dx, dt, u_d, g_left_arr, g_right_arr, bc_type, u0_func=u0, alpha=alpha
        )
        return g

    result = minimize(
        J_func, m0, method="L-BFGS-B", jac=grad_func, bounds=None,
        options={"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-7},
    )

    m_opt = result.x.reshape(n, n_t_pts)
    if verbose:
        print(f"  {bc_type} BC sample -> J={result.fun:.2e}, success={result.success}")

    if not result.success:
        print(m_opt[0,:])
    return x, t, m_opt


def generate_dataset(
    n_samples,
    bc_type,
    n_x=N_X_DATA,
    n_t=N_T_DATA,
    alpha=ALPHA_DATA,
    seed=42,
    verbose=True,
    ell=GRF_ELL,
    sigma=GRF_SIGMA,
):
    """
    Generate dataset with either all Dirichlet or all Neumann time-dependent BCs.
    bc_type: "dirichlet" or "neumann". Boundary functions are sampled from a GRF in time.
    Returns g_left (n_samples, n_t+1), g_right (n_samples, n_t+1), m_opts (n_samples, n_x+1, n_t+1), x, t.
    """
    if bc_type not in ("dirichlet", "neumann"):
        raise ValueError("bc_type must be 'dirichlet' or 'neumann' (no mixing).")

    np.random.seed(seed)
    t_grid = np.linspace(0, T, n_t + 1)
    g_left_list = []
    g_right_list = []
    m_opts = []
    x_ref, t_ref = None, None

    for i in range(n_samples):
        g_left_arr = sample_grf_on_grid(t_grid, ell=ell, sigma=sigma)
        g_right_arr = sample_grf_on_grid(t_grid, ell=ell, sigma=sigma)
        # print(g_left_arr)
        x, t, m_opt = solve_optimal_control_one_tdep(
            g_left_arr, g_right_arr, bc_type, n_x, n_t, alpha,
            verbose=(verbose and (i % 1 == 0)),
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


def main():
    import argparse
    p = argparse.ArgumentParser(description="Generate heat optimal control data with GRF boundary conditions.")
    p.add_argument("--samples", type=int, default=50, help="Number of BC samples")
    p.add_argument("--bc_type", type=str, default="dirichlet", choices=("dirichlet", "neumann"),
                   help="Boundary type: dirichlet or neumann (no mixing in one dataset)")
    p.add_argument("--n_x", type=int, default=N_X_DATA)
    p.add_argument("--n_t", type=int, default=N_T_DATA)
    p.add_argument("--out", type=str, default="heat_optimal_control_data.npz")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ell", type=float, default=GRF_ELL, help="GRF length scale in time")
    p.add_argument("--sigma", type=float, default=GRF_SIGMA, help="GRF std")
    args = p.parse_args()

    print(f"Generating optimal control data: {args.bc_type} BCs from Gaussian random field (no mixing).")
    print(f"  GRF ell={args.ell}, sigma={args.sigma}")
    g_left, g_right, m_opts, x, t = generate_dataset(
        args.samples,
        args.bc_type,
        n_x=args.n_x,
        n_t=args.n_t,
        verbose=True,
        ell=args.ell,
        sigma=args.sigma,
        seed=args.seed,
    )
    np.savez(
        args.out,
        bc_type=args.bc_type,
        g_left=g_left,
        g_right=g_right,
        m_opts=m_opts,
        x=x,
        t=t,
    )
    print(f"Saved {args.samples} samples to {args.out}")
    print(f"  bc_type: {args.bc_type}, g_left shape: {g_left.shape}, g_right shape: {g_right.shape}, m_opts shape: {m_opts.shape}")


if __name__ == "__main__":
    main()
