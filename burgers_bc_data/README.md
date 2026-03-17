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

## Solver

- **Explicit Euler** finite difference in time (same style as the [neural operator Burgers example](https://neuraloperator.github.io/dev/auto_examples/data_gen/plot_burgers_2d_solver.html)):  
  \( u^{n+1} = u^n + \Delta t\,(-u^n u_x^n + \nu u_{xx}^n) \).
- Central differences for \( u_x \) and \( u_{xx} \); Dirichlet BCs applied after each sub-step.
- Sub-stepping between output time levels keeps the scheme stable (default 25 sub-steps per output interval).
- Domain: \( L=1 \), \( T=0.5 \). Default viscosity `nu=0.01`, grid `n_x=24`, `n_t=30`.
- Boundary functions \( g_{\mathrm{left}}(t) \), \( g_{\mathrm{right}}(t) \) are sampled from a Gaussian random field (GRF) in time.
