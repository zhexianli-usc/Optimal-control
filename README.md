# Burgers BC Data

This branch contains a minimal, paper-ready release focused on the `burgers_bc_data` module for the 1D viscous Burgers equation.

## Scope

The purpose of this branch is to provide a clean open-source artifact with:
- data generation scripts
- prepared datasets
- training scripts for boundary-condition-to-solution operator learning
- training outputs for reproducibility

All non-Burgers components from the original multi-problem repository are intentionally excluded.

## Folder Layout

```text
burgers_bc_data/
├── README.md
├── burgers_bc_data.npz
├── burgers_optimal_control_data.npz
├── generate_burgers_bc_data.py
├── generate_burgers_optimal_control_data.py
├── plot_burgers_example.py
├── train_fno_burgers_bc.py
├── train_fno_burgers_bc_real_freq.py
├── train_fno_burgers_bc_output/
├── train_fno_burgers_bc_real_freq_output/
├── train_fno_burgers_optimal_control/
└── train_fno_burgers_real_freq_output_optimal_control/
```

## Problem Setting

We model the 1D viscous Burgers equation:

$$
 u_t + u u_x = \nu u_{xx}
$$

with time-dependent Dirichlet boundary conditions:

$$
 u(0,t)=g_{\text{left}}(t), \quad u(L,t)=g_{\text{right}}(t)
$$

The learning target is the operator mapping boundary condition histories to the full space-time solution field.

## Quick Start

```bash
cd burgers_bc_data
python generate_burgers_bc_data.py --samples 50 --out burgers_bc_data.npz --plot
python train_fno_burgers_bc.py
```

## Training Models

This release includes two operator-learning training scripts for BC-to-solution mapping:

- `train_fno_burgers_bc.py`: **Extended Fourier Neural Operator (E-FNO)** implementation (complex-frequency inverse formulation).
- `train_fno_burgers_bc_real_freq.py`: **Classical Fourier Neural Operator (FNO)** implementation using real-frequency discrete Fourier transforms.

Run them from `burgers_bc_data`:

```bash
python train_fno_burgers_bc.py
python train_fno_burgers_bc_real_freq.py
```

## Data Format

The main dataset file is `burgers_bc_data/burgers_bc_data.npz`, containing:
- `g_left`: left boundary trajectories
- `g_right`: right boundary trajectories
- `u`: solution fields on the space-time grid
- `x`: spatial grid
- `t`: temporal grid

## Reproducibility Notes

- Random sampling is controlled by script-level seed options.
- Training logs and comparison metrics are included in output subfolders.
- Script defaults are tuned for consistency with experiments described in the associated paper.

## Citation

If you use this branch in research, please cite your paper and acknowledge this code release.

## License

Use the same license terms as the parent repository unless stated otherwise in the publication package.
