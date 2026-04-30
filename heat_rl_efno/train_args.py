"""
Shared argument definitions for DDPG heat-control training scripts.

Both `train_ddpg_heat_efno.py` and `train_ddpg_heat_fno_real_freq.py` call
`add_shared_args(p, env_d)` so that any change here applies to both.
"""
from __future__ import annotations

import argparse


def add_shared_args(p: argparse.ArgumentParser, env_d) -> None:
    """Add all common training arguments to *p* in-place."""
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--batch-envs", type=int, default=16, help="Parallel rollouts per episode")
    p.add_argument("--batch-train", type=int, default=64)
    p.add_argument("--buffer-size", type=int, default=50_000)
    p.add_argument("--updates-per-episode", type=int, default=80)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--lr-actor", type=float, default=1e-4)
    p.add_argument("--lr-critic", type=float, default=3e-4)
    p.add_argument("--noise-init", type=float, default=0.15)
    p.add_argument("--noise-final", type=float, default=0.02)
    p.add_argument("--noise-decay", type=float, default=0.995)
    p.add_argument("--nx", type=int, default=env_d.nx, help="Spatial grid points (incl. boundaries); default from HeatEnvConfig")
    p.add_argument("--nt", type=int, default=env_d.nt, help="Time slices incl. t=0; default from HeatEnvConfig")
    p.add_argument("--L", type=float, default=env_d.L)
    p.add_argument("--T", type=float, default=env_d.T)
    p.add_argument("--alpha", type=float, default=env_d.alpha)
    p.add_argument("--u-max", type=float, default=env_d.u_abs_max, help="Soft state threshold |u| (HeatEnvConfig.u_abs_max)")
    p.add_argument("--w-control", type=float, default=env_d.w_control)
    p.add_argument("--w-constraint", type=float, default=env_d.w_constraint)
    p.add_argument("--w-tracking", type=float, default=env_d.w_tracking)
    p.add_argument("--a-max", type=float, default=2.0, help="tanh bound on control magnitude")
    p.add_argument("--k-max", type=int, default=8)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-batch-envs", type=int, default=16, help="Fixed evaluation IC batch size (no exploration noise)")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--save-interval", type=int, default=200)
