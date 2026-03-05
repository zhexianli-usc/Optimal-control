# Optimal Control

Optimal control for 1D heat equation with neural operator surrogates (DeepONet, FNO).

## Contents

- **Heat solver**: `heat_1d_numpy_scipy.py` — 1D time-dependent heat equation solver (NumPy/SciPy).
- **Data generation**: `generate_heat_optimal_control_data.py` — Generates optimal control data with GRF boundary conditions.
- **Training**: `train_deeponet_heat_bc.py`, `train_fno_heat_bc.py`, `train_fno_heat_bc_real_freq.py` — Train DeepONet or FNO (BC → optimal control).
- **Inference**: `heat_optimal_control_deeponet_1d.py`, `heat_optimal_control_fno_1d.py` — Use trained models for optimal control.

## Setup

- Python 3.x with NumPy, SciPy, PyTorch.
- Optional: FEniCS/dolfin-adjoint for `Optimal_control.py` (see `INSTALL_*.md`).

## Usage

1. Generate data: `python generate_heat_optimal_control_data.py`
2. Train: `python train_deeponet_heat_bc.py` or `python train_fno_heat_bc.py`
3. Run inference with a trained model as in the heat_optimal_control_* scripts.
