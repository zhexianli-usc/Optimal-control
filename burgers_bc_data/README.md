# 1D Burgers equation – BC → solution data for FNO

This folder contains a finite-difference data generator for the **1D Burgers equation** with time-dependent Dirichlet boundary conditions, in the same layout as the heat optimal-control data used for the Fourier Neural Operator (FNO).

## PDE

- **Equation:** \( u_t + u\,u_x = \nu\,u_{xx} \) on \( (0,L)\times(0,T) \)
- **BCs:** \( u(0,t) = g_{\mathrm{left}}(t) \), \( u(L,t) = g_{\mathrm{right}}(t) \)
- **IC:** \( u(x,0) = \sin(\pi x/L) \) for all samples (same initial condition for every boundary condition)

## Data format (same as heat `.npz`)

- `bc_type`: `"dirichlet"`
- `g_left`: `(n_samples, n_t+1)` – left boundary values in time
- `g_right`: `(n_samples, n_t+1)` – right boundary values in time
- `u`: `(n_samples, n_x+1, n_t+1)` – solution \( u(x,t) \) (FNO target; heat data uses `m_opts` for control)
- `x`: spatial grid
- `t`: time grid

## Usage (Conda)

From the **project root** (with conda installed), create the environment once:

```bash
conda env create -f environment.yml
conda activate optimal-control
```

Then generate data and plot an example:

```bash
cd burgers_bc_data
python generate_burgers_bc_data.py --samples 50 --out burgers_bc_data.npz --plot
```

Or use the helper scripts (run from `burgers_bc_data` in **Anaconda Prompt** or a shell where `conda` is in PATH):

- **Windows (cmd):** `run_with_conda.bat`
- **PowerShell:** `.\run_with_conda.ps1`

- **Generate data (manual):**  
  `python generate_burgers_bc_data.py --samples 50 --out burgers_bc_data.npz`  
  (optional: `--n_x`, `--n_t`, `--nu`, `--ell`, `--sigma`, `--seed`, `--plot`)
- **Load in Python:**  
  `data = np.load("burgers_bc_data.npz", allow_pickle=True)`  
  then `g_left`, `g_right`, `u`, `x`, `t` = `data["g_left"]`, etc.

## Solver (`generate_burgers_bc_data.py`)

- **Explicit Euler** in time; **upwind** \( u_x \); explicit diffusion \( \nu u_{xx} \); Dirichlet BCs each sub-step.
- Sub-stepping between output times for stability (default 80 sub-steps per interval).
- Domain: \( L=1 \), \( T=0.5 \). Default `nu=0.01`, `n_x=24`, `n_t=30`.
- BCs sampled from a GRF in time.

## Burgers optimal control (`generate_burgers_optimal_control_data.py`)

Distributed control \( m(x,t) \) with the same objective structure as the heat optimal-control generator:

- **State:** \( u_t + u u_x = \nu u_{xx} + m \), Dirichlet BCs \( g_{\mathrm{left}}(t), g_{\mathrm{right}}(t) \), IC \( u(x,0)=\sin(\pi x/L) \) (corners match BC at \( t=0 \)).
- **Cost:** \( J = \frac{1}{2}\sum (u-u_d)^2\,dx\,dt + \frac{\alpha}{2}\sum m^2\,dx\,dt \) with \( u_d \) from `heat_1d_numpy_scipy.u_desired`.
- **Forward:** semi-implicit diffusion \( (I-\Delta t\,\nu L)u^{k+1} = u^k + \Delta t(-F(u^k)+m^{k+1}) \) with first-order upwind \( F \).
- **Adjoint:** discrete **discretize-then-differentiate** adjoint (transpose of linearized forward w.r.t. \( u \), implicit diffusion transpose), consistent with the principles in [Fikl et al., arXiv:2209.03270](https://arxiv.org/pdf/2209.03270).
- **Gradient:** \( \nabla_m J = (p + \alpha m)\,dx\,dt \); optimization with **L-BFGS-B** (same pattern as `generate_heat_optimal_control_data.py`).

**Output** matches heat optimal-control layout: `bc_type`, `g_left`, `g_right`, `m_opts`, `x`, `t` (plus `nu`, `alpha`, `equation` metadata).

```bash
python generate_burgers_optimal_control_data.py --samples 20 --plot
```

`--plot` saves `burgers_optimal_control_verification.png` (optimal \( m \), state \( u \), desired \( u_d \), tracking error).
