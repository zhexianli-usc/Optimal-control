"""
Load a trained EFNO policy checkpoint, export closed-loop rollouts, and compare
against an open-loop baseline optimized with **discrete adjoint** gradients
(autodiff through the same Crank–Nicolson dynamics as `HeatEnv`).

Writes:
  - `comparison_summary.csv`: per-sample costs and returns
  - `policy_vs_adjoint_export.npz`: grids x, t, and first-sample u, a for RL vs adjoint
  - optional `comparison_u.png`: temperature snapshots
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from .adjoint_open_loop import differentiable_rollout, optimize_open_loop_discrete_adjoint
from .heat_env import HeatEnv, HeatEnvConfig, cn_context_from_cfg
from .models import ActorEFNO
from .rollouts import rollout_policy, total_return_to_cost
from .train_ddpg_heat_efno import random_initial_u0


def load_actor(ckpt_path: str, device: torch.device) -> tuple[ActorEFNO, HeatEnvConfig, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: HeatEnvConfig = ckpt["cfg"]
    args: dict = ckpt["args"]
    actor = ActorEFNO(
        cfg.nx,
        cfg.nt,
        k_max=int(args["k_max"]),
        width=int(args["width"]),
        n_layers=int(args["n_layers"]),
        a_max=float(args["a_max"]),
    ).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor, cfg, args


def maybe_plot(
    out_png: str,
    x: np.ndarray,
    t: np.ndarray,
    u_rl: np.ndarray,
    u_adj: np.ndarray,
):
    import matplotlib.pyplot as plt

    # u_* shape (nx, nt): rows = x index, cols = time (matches tensors in rollout)
    extent = (float(x[0]), float(x[-1]), float(t[0]), float(t[-1]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
    last_im = None
    for ax, u, title in zip(
        axes,
        [u_rl, u_adj],
        ["RL policy (closed-loop)", "Adjoint open-loop"],
    ):
        last_im = ax.imshow(
            u,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="viridis",
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel("x")
    axes[0].set_ylabel("t")
    fig.colorbar(last_im, ax=axes, shrink=0.6, label="u")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint_ep*.pt or final .pt")
    p.add_argument("--out-dir", type=str, default="heat_rl_efno_compare_output")
    p.add_argument("--n-samples", type=int, default=8, help="Number of random ICs to evaluate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--adjoint-iters", type=int, default=500)
    p.add_argument("--adjoint-lr", type=float, default=0.05)
    p.add_argument("--warm-start-rl", action="store_true", help="Initialize adjoint optimization from RL controls")
    p.add_argument("--plot", action="store_true")
    args_ns = p.parse_args()

    device = torch.device("cpu" if args_ns.cpu or not torch.cuda.is_available() else "cuda")
    torch.manual_seed(args_ns.seed)
    np.random.seed(args_ns.seed)

    actor, cfg, train_args = load_actor(args_ns.checkpoint, device)
    env = HeatEnv(cfg, device)
    ctx = cn_context_from_cfg(cfg, device)
    nx, nt = cfg.nx, cfg.nt
    x_norm = (env.x / cfg.L).view(1, nx, 1)
    t_norm = (env.t / cfg.T).view(1, 1, nt)

    os.makedirs(args_ns.out_dir, exist_ok=True)

    u0 = random_initial_u0(args_ns.n_samples, nx, cfg.L, device)
    u_rl, a_rl, ret_rl = rollout_policy(actor, env, u0, x_norm, t_norm)
    cost_rl = total_return_to_cost(ret_rl)

    warm = a_rl if args_ns.warm_start_rl else None
    a_adj, u_adj, hist = optimize_open_loop_discrete_adjoint(
        u0,
        cfg,
        ctx,
        n_iters=args_ns.adjoint_iters,
        lr=args_ns.adjoint_lr,
        init_scale=0.0,
        a_warm_start=warm,
    )
    _, cost_adj = differentiable_rollout(u0, a_adj, ctx, cfg)

    cost_rl_np = cost_rl.cpu().numpy()
    cost_adj_np = cost_adj.cpu().numpy()
    ret_rl_np = ret_rl.cpu().numpy()

    csv_path = os.path.join(args_ns.out_dir, "comparison_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "total_return_rl", "total_cost_rl", "total_cost_adjoint_open_loop"])
        for i in range(args_ns.n_samples):
            w.writerow([i, float(ret_rl_np[i]), float(cost_rl_np[i]), float(cost_adj_np[i])])
        w.writerow(
            [
                "mean",
                float(ret_rl_np.mean()),
                float(cost_rl_np.mean()),
                float(cost_adj_np.mean()),
            ]
        )

    x_np = env.x.cpu().numpy()
    t_np = env.t.cpu().numpy()
    np.savez(
        os.path.join(args_ns.out_dir, "policy_vs_adjoint_export.npz"),
        x=x_np,
        t=t_np,
        u0=u0.cpu().numpy(),
        u_rl=u_rl.cpu().numpy(),
        a_rl=a_rl.cpu().numpy(),
        u_adj=u_adj.cpu().numpy(),
        a_adj=a_adj.cpu().numpy(),
        total_return_rl=ret_rl_np,
        total_cost_rl=cost_rl_np,
        total_cost_adjoint=cost_adj_np,
        checkpoint=np.array([args_ns.checkpoint], dtype=object),
    )

    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "heat_env_config": cfg,
            "train_args": train_args,
        },
        os.path.join(args_ns.out_dir, "exported_policy_operator.pt"),
    )

    print(f"Wrote {csv_path}")
    print(f"Mean total return (RL):     {ret_rl_np.mean():.6f}")
    print(f"Mean total cost (RL):       {cost_rl_np.mean():.6f}  (= -return)")
    print(f"Mean total cost (adjoint): {cost_adj_np.mean():.6f}")
    if hist:
        print(f"Adjoint optimization final batch-mean loss: {hist[-1]:.6f}")

    if args_ns.plot:
        maybe_plot(
            os.path.join(args_ns.out_dir, "comparison_u.png"),
            x_np,
            t_np,
            u_rl[0].cpu().numpy(),
            u_adj[0].cpu().numpy(),
        )
        print(f"Wrote {os.path.join(args_ns.out_dir, 'comparison_u.png')}")


if __name__ == "__main__":
    main()
