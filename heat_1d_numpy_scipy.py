"""
1D time-dependent heat equation solver (finite differences in NumPy) with
optimal control via forcing term, optimized with SciPy.
PDE: u_t = u_xx + m(x,t) on (0, L) x (0, T), with configurable BCs.
Control: m(x, t) = forcing term.
Objective: J = 0.5 * int (u - u_d)^2 dx dt + (alpha/2) * int m^2 dx dt.

With alpha=0 the minimizer of the tracking term can be non-unique (many m give
the same u); use alpha > 0 (e.g. 1e-10) so the optimal m is unique and all
initial guesses converge to the same control.
"""
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize

# -----------------------------------------------------------------------------
# Configuration: domain, grid, boundary conditions
# -----------------------------------------------------------------------------
L = 1.0           # domain (0, L)
T = 0.5         # time horizon (0, T)
n_x = 80          # number of spatial intervals (n_x + 1 points)
n_t = 100         # number of time steps (n_t + 1 points)

# Boundary conditions: ("dirichlet", value) or ("neumann", value)
#   dirichlet: u = value at that end
#   neumann:   u' = value at that end (flux)
# Examples: both Dirichlet zero; or left Neumann zero, right Dirichlet zero
bc_left  = ("dirichlet", 0.0)
bc_right = ("dirichlet", 0.0)

# Initial condition u(x, 0) = u0(x)
def u0(x):
    return np.sin(np.pi * x)

# Desired state u_d(x, t) (target for tracking)
def u_desired(x, t):
    return np.exp(-t) * np.sin(np.pi * x)

# Regularization on control. Use alpha > 0 (e.g. 1e-10) for unique optimal m so different
# initial guesses converge to the same control; alpha=0 can give numerically non-unique minima.
alpha = 1

# Control bounds (None = no bounds; use bounds to keep m in [m_min, m_max])
m_min, m_max = None, None


def build_laplacian_1d(n_x, dx, bc_left, bc_right):
    """
    Build the 1D discrete Laplacian (second derivative) with BCs.
    Returns a sparse matrix A such that A @ u ≈ u_xx at interior points.
    u has length n_x + 1 (full grid including boundaries).
    """
    n = n_x + 1
    # Standard 3-point stencil: (u_{i-1} - 2*u_i + u_{i+1}) / dx^2
    diag = -2.0 * np.ones(n)
    off = np.ones(n - 1)
    A = (1.0 / (dx ** 2)) * (diags(off, -1) + diags(diag, 0) + diags(off, 1))
    A = A.tolil()  # mutable format for row assignment

    # Apply BCs by overwriting rows
    type_left, val_left = bc_left
    type_right, val_right = bc_right

    if type_left == "dirichlet":
        A[0, :] = 0.0  # (I-dt*A)[0,0]=1; we set rhs[0]=val in bc_rhs
    elif type_left == "neumann":
        A[0, :] = 0.0  # row overwritten in solver with [1,-1,0,...] so u[0]-u[1]=-dx*val

    if type_right == "dirichlet":
        A[-1, :] = 0.0
    elif type_right == "neumann":
        A[-1, :] = 0.0  # row overwritten in solver with [0,...,-1,1]

    return A.tocsr()


def bc_rhs(n_x, dx, bc_left, bc_right, adjoint=False):
    """RHS vector to enforce BC values when solving (I - dt*A)*u = rhs.
    For adjoint: lambda = 0 on Dirichlet, zero flux on Neumann, so zeros.
    """
    n = n_x + 1
    b = np.zeros(n)
    if adjoint:
        return b
    type_left, val_left = bc_left
    type_right, val_right = bc_right
    if type_left == "dirichlet":
        b[0] = val_left
    elif type_left == "neumann":
        b[0] = -dx * val_left
    if type_right == "dirichlet":
        b[-1] = val_right
    elif type_right == "neumann":
        b[-1] = dx * val_right
    return b


def apply_bc_to_M(M, bc_left, bc_right):
    """Overwrite first/last rows of M for Neumann so that discrete u'(0)=g is enforced."""
    type_left, _ = bc_left
    type_right, _ = bc_right
    if type_left == "neumann":
        M[0, :] = 0.0
        M[0, 0] = 1.0
        M[0, 1] = -1.0
    if type_right == "neumann":
        M[-1, :] = 0.0
        M[-1, -2] = -1.0
        M[-1, -1] = 1.0
    return M


def solve_forward(m, x, t, A, dx, dt, bc_left, bc_right, u0_func):
    """
    Solve u_t = u_xx + m with given BCs and initial condition.
    m: shape (n_x+1, n_t+1), same as u.
    Returns u of shape (n_x+1, n_t+1).
    """
    n = len(x)
    n_t_steps = len(t) - 1
    u = np.zeros((n, n_t_steps + 1))
    u[:, 0] = u0_func(x)

    I = diags(np.ones(n), 0).tocsr()
    M = (I - dt * A).tolil()
    apply_bc_to_M(M, bc_left, bc_right)
    M = M.tocsr()
    bc_vec = bc_rhs(n_x, dx, bc_left, bc_right)

    for k in range(n_t_steps):
        rhs = u[:, k] + dt * m[:, k + 1]
        rhs[0] = bc_vec[0]
        rhs[-1] = bc_vec[-1]
        u[:, k + 1] = spsolve(M, rhs)

    return u


def objective_and_gradient(m_flat, x, t, A, dx, dt, u_d, bc_left, bc_right, u0_func, alpha, m_min, m_max):
    """
    J(m) and gradient. m_flat is the flattened control (n_x+1)*(n_t+1).
    Returns (J, grad_flat).
    """
    n = len(x)
    n_t_steps = len(t) - 1
    m = m_flat.reshape(n, n_t_steps + 1)

    u = solve_forward(m, x, t, A, dx, dt, bc_left, bc_right, u0_func)

    # Objective: J = 0.5 * int (u - u_d)^2 + (alpha/2) * int m^2
    diff = u - u_d
    J_tracking = 0.5 * dx * dt * np.sum(diff ** 2)
    J_reg = 0.5 * alpha * dx * dt * np.sum(m ** 2)
    J = J_tracking + J_reg

    # Adjoint: -lambda_t = lambda_xx + (u - u_d), lambda(T) = 0; lambda=0 on Dirichlet, zero flux on Neumann
    # Backward in time: (I - dt*A)*lambda^n = lambda^{n+1} + dt*(u^n - u_d^n)
    I = diags(np.ones(n), 0).tocsr()
    M = (I - dt * A).tolil()
    apply_bc_to_M(M, bc_left, bc_right)
    M = M.tocsr()
    bc_vec_adj = bc_rhs(n_x, dx, bc_left, bc_right, adjoint=True)

    lam = np.zeros((n, n_t_steps + 1))
    lam[:, -1] = 0.0
    for k in range(n_t_steps - 1, -1, -1):
        rhs = lam[:, k + 1] + dt * (u[:, k] - u_d[:, k])
        rhs[0] = bc_vec_adj[0]
        rhs[-1] = bc_vec_adj[-1]
        lam[:, k] = spsolve(M.T, rhs)

    # Gradient: dJ/dm = lambda + alpha*m (pointwise)
    grad = (lam + alpha * m) * (dx * dt)
    grad_flat = grad.ravel()
    return J, grad_flat


def main():
    dx = L / n_x
    dt = T / n_t
    x = np.linspace(0, L, n_x + 1)
    t = np.linspace(0, T, n_t + 1)

    # Desired state on grid
    X, T_grid = np.meshgrid(x, t, indexing="ij")
    u_d = u_desired(X, T_grid)

    A = build_laplacian_1d(n_x, dx, bc_left, bc_right)

    # Initial guess: zero forcing; m has one column per time point (0..n_t inclusive)
    n = n_x + 1
    n_t_pts = n_t + 1  # number of time points
    m0 = np.random.rand(n * n_t_pts)

    bounds = None
    if m_min is not None and m_max is not None:
        bounds = [(m_min, m_max)] * (n * n_t_pts)

    # Store last J so callback can print without extra cost
    last_J = [None]
    step = [0]

    def J_func(m_flat):
        J, _ = objective_and_gradient(
            m_flat, x, t, A, dx, dt, u_d, bc_left, bc_right, u0_func=u0, alpha=alpha,
            m_min=m_min, m_max=m_max
        )
        last_J[0] = J
        return J

    def grad_func(m_flat):
        J, g = objective_and_gradient(
            m_flat, x, t, A, dx, dt, u_d, bc_left, bc_right, u0_func=u0, alpha=alpha,
            m_min=m_min, m_max=m_max
        )
        last_J[0] = J
        return g

    def callback(xk):
        step[0] += 1
        if last_J[0] is not None:
            print(f"    step {step[0]:4d}: J = {last_J[0]:.6e}")

    print("1D time-dependent heat optimal control (NumPy + SciPy)")
    print(f"  BC left:  {bc_left[0]} = {bc_left[1]}")
    print(f"  BC right: {bc_right[0]} = {bc_right[1]}")
    print(f"  Grid: n_x={n_x}, n_t={n_t}, alpha={alpha}")
    # Stopping: ftol (relative cost change), gtol (gradient norm). With alpha=0, no bounds, use enough iterations to drive J ~ 0.
    # (disp/iprint removed: deprecated in SciPy, will be removed in 1.18)
    opt_options = {
        "maxiter": 3000,
        "ftol": 1e-12,
        "gtol": 1e-10,
    }
    print("  Running L-BFGS-B...")

    result = minimize(
        J_func,
        m0,
        method="L-BFGS-B",
        jac=grad_func,
        bounds=bounds,
        callback=callback,
        options=opt_options,
    )

    m_opt = result.x.reshape(n, n_t_pts)
    u_opt = solve_forward(m_opt, x, t, A, dx, dt, bc_left, bc_right, u0)

    J_final, _ = objective_and_gradient(
        result.x, x, t, A, dx, dt, u_d, bc_left, bc_right, u0_func=u0, alpha=alpha,
        m_min=m_min, m_max=m_max
    )
    print(f"  Success: {result.success}")
    print(f"  Final J: {J_final:.6e}")
    print("  Optimal state u and control m in (x, t) returned by solve_forward(m_opt, ...).")
    print(f"  Optimal control is {m_opt}")
    return result, x, t, u_opt, m_opt, u_d


if __name__ == "__main__":
    result, x, t, u_opt, m_opt, u_d = main()
