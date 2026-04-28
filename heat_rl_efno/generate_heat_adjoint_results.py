"""
Precompute and store numerical-adjoint optimal-control results for heat equation.

This creates a standalone `.npz` containing:
  - u0 samples
  - optimal adjoint controls a_adj
  - resulting states u_adj
  - adjoint objective values / convergence flags

Then `compare_policy_adjoint.py` can load this file and only do RL-vs-adjoint plotting.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from heat_rl_efno.heat_env import HeatEnvConfig
    from heat_rl_efno.solve_heat_optimal_control_adjoint import solve_optimal_control_numerical_adjoint
    from heat_rl_efno.train_ddpg_heat_efno import random_initial_u0
else:
    from .heat_env import HeatEnvConfig
    from .solve_heat_optimal_control_adjoint import solve_optimal_control_numerical_adjoint
    from .train_ddpg_heat_efno import random_initial_u0


def load_cfg_from_checkpoint(ckpt_path: str, device: torch.device) -> HeatEnvConfig:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    return ckpt["cfg"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="Checkpoint used to get HeatEnvConfig")
    p.add_argument("--out-file", type=str, default="heat_rl_efno_compare_output/adjoint_results.npz")
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--adjoint-maxiter", type=int, default=300)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_cfg_from_checkpoint(args.checkpoint, device)
    nx, nt = cfg.nx, cfg.nt
    x = np.linspace(0.0, cfg.L, nx, dtype=np.float32)
    t = np.linspace(0.0, cfg.T, nt, dtype=np.float32)

    u0 = random_initial_u0(args.n_samples, nx, cfg.L, device).cpu().numpy().astype(np.float32)
    a_adj = np.zeros((args.n_samples, nx, nt - 1), dtype=np.float32)
    u_adj = np.zeros((args.n_samples, nx, nt), dtype=np.float32)
    cost_adj = np.zeros((args.n_samples,), dtype=np.float32)
    adj_iters = np.zeros((args.n_samples,), dtype=np.int32)
    adj_success = np.zeros((args.n_samples,), dtype=np.int8)

    for i in range(args.n_samples):
        res = solve_optimal_control_numerical_adjoint(
            u0[i],
            cfg,
            maxiter=args.adjoint_maxiter,
            init_control_full=None,
            verbose=False,
        )
        a_adj[i] = res.a_opt
        u_adj[i] = res.u_opt
        cost_adj[i] = res.objective
        adj_iters[i] = res.n_iter
        adj_success[i] = 1 if res.success else 0

    out_dir = os.path.dirname(args.out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(
        args.out_file,
        u0=u0,
        a_adj=a_adj,
        u_adj=u_adj,
        total_cost_numerical_adjoint=cost_adj,
        numerical_adjoint_iters=adj_iters,
        numerical_adjoint_success=adj_success,
        x=x,
        t=t,
        checkpoint=np.array([args.checkpoint], dtype=object),
    )
    print(f"Wrote {args.out_file}")
    print(f"Mean numerical adjoint cost: {float(cost_adj.mean()):.6f}")
    print(f"Adjoint success rate: {float(adj_success.mean()):.2%}")


if __name__ == "__main__":
    main()
