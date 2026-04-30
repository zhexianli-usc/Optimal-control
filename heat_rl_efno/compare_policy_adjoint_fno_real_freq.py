"""
Load a trained real-frequency FNO policy checkpoint and a precomputed numerical-adjoint
result file, then export/plot RL-vs-adjoint comparison.

Writes:
  - `comparison_summary.csv`: per-sample costs and returns
  - `policy_vs_adjoint_export.npz`: grids x, t, and trajectories/controls for RL vs numerical adjoint
  - optional `comparison_u.png`: temperature snapshots
  - optional `comparison_a.png`: control snapshots
    - optional `comparison_a_abs_error.png`: absolute control difference |a_RL - a_adj|
"""
from __future__ import annotations
from plot_style import apply_plot_style

apply_plot_style()

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

if __package__ in (None, ""):
    # Direct script execution: python heat_rl_efno/compare_policy_adjoint_fno_real_freq.py
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from heat_rl_efno.heat_env import HeatEnv, HeatEnvConfig
    from heat_rl_efno.models_fno_real import ActorFNORealFreq
    from heat_rl_efno.rollouts import rollout_policy, total_return_to_cost
else:
    from .heat_env import HeatEnv, HeatEnvConfig
    from .models_fno_real import ActorFNORealFreq
    from .rollouts import rollout_policy, total_return_to_cost


def load_actor(ckpt_path: str, device: torch.device) -> tuple[ActorFNORealFreq, HeatEnvConfig, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: HeatEnvConfig = ckpt["cfg"]
    args: dict = ckpt["args"]
    actor = ActorFNORealFreq(
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


def _xt_mesh_for_field(x: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, T) node grids for fields stored on (nx, nt) with x[i], t[j]."""
    X, T = np.meshgrid(x.astype(np.float64), t.astype(np.float64), indexing="ij")
    return X, T


def maybe_plot(
    out_png_rl: str,
    out_png_adj: str,
    x: np.ndarray,
    t: np.ndarray,
    u_rl: np.ndarray,
    u_adj: np.ndarray,
):
    import matplotlib.pyplot as plt

    X, T = _xt_mesh_for_field(x, t)
    vmin = float(min(np.min(u_rl), np.min(u_adj)))
    vmax = float(max(np.max(u_rl), np.max(u_adj)))

    for out_png, u, title in [
        (out_png_rl, u_rl, "RL policy (closed-loop, FNO real freq)"),
        (out_png_adj, u_adj, "Numerical adjoint open-loop"),
    ]:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3.5), constrained_layout=True)
        im = ax.pcolormesh(X, T, u, shading="auto", cmap="hot", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$t$")
        fig.colorbar(im, ax=ax, shrink=0.8, label="$u$")
        fig.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)


def maybe_plot_control(
    out_png_rl: str,
    out_png_adj: str,
    x: np.ndarray,
    t: np.ndarray,
    a_rl: np.ndarray,
    a_adj: np.ndarray,
    vmax: float,
):
    import matplotlib.pyplot as plt

    t_ctrl = t[:-1] if t.shape[0] == a_rl.shape[1] + 1 else t[: a_rl.shape[1]]
    X, Tc = _xt_mesh_for_field(x, t_ctrl)
    vmax = max(float(vmax), 1e-8)

    for out_png, a, title in [
        (out_png_rl, a_rl, "RL policy control (FNO real freq)"),
        (out_png_adj, a_adj, "Numerical adjoint control"),
    ]:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3.5), constrained_layout=True)
        im = ax.pcolormesh(X, Tc, a, shading="auto", cmap="hot", vmin=-vmax, vmax=vmax)
        # ax.set_title(title)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$t$")
        fig.colorbar(im, ax=ax, shrink=0.8, label="$u$")
        fig.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)


def maybe_plot_control_abs_error(
    out_png: str,
    x: np.ndarray,
    t: np.ndarray,
    a_rl: np.ndarray,
    a_adj: np.ndarray,
    vmax: float,
):
    import matplotlib.pyplot as plt

    t_ctrl = t[:-1] if t.shape[0] == a_rl.shape[1] + 1 else t[: a_rl.shape[1]]
    X, Tc = _xt_mesh_for_field(x, t_ctrl)
    abs_err = np.abs(a_rl - a_adj)
    vmax = max(float(vmax), 1e-12)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5), constrained_layout=True)
    im = ax.pcolormesh(X, Tc, abs_err, shading="auto", cmap="Oranges", vmin=0.0, vmax=1.3)
    ax.set_title("$|a_{RL} - a_{adj}|$ (FNO real freq)")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$t$")
    fig.colorbar(im, ax=ax, shrink=0.8, label="$|a_{RL} - a_{adj}|$")
    fig.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        type=str,
        default="heat_rl_fno_real_freq_output/checkpoint_final.pt",
        help="Path to checkpoint_ep*.pt or final .pt from train_ddpg_heat_fno_real_freq.py",
    )
    p.add_argument(
        "--adjoint-file",
        type=str,
        default="heat_rl_efno_compare_output/adjoint_results.npz",
        help="Path to .npz generated by generate_heat_adjoint_results.py",
    )
    p.add_argument("--out-dir", type=str, default="heat_rl_fno_real_freq_compare_output")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--plot", default=True, action="store_true")
    args_ns = p.parse_args()

    device = torch.device("cpu" if args_ns.cpu or not torch.cuda.is_available() else "cuda")

    actor, cfg, train_args = load_actor(args_ns.checkpoint, device)
    env = HeatEnv(cfg, device)
    nx, nt = cfg.nx, cfg.nt
    x_norm = (env.x / cfg.L).view(1, nx, 1)
    t_norm = (env.t / cfg.T).view(1, 1, nt)

    os.makedirs(args_ns.out_dir, exist_ok=True)

    adj_data = np.load(args_ns.adjoint_file, allow_pickle=True)
    u0_np = adj_data["u0"]
    a_adj_np = adj_data["a_adj"]
    u_adj_np = adj_data["u_adj"]
    cost_adj_np = adj_data["total_cost_numerical_adjoint"]
    adj_iters = adj_data["numerical_adjoint_iters"]
    adj_success = adj_data["numerical_adjoint_success"]
    if u0_np.shape[1] != nx or u_adj_np.shape[1] != nx or u_adj_np.shape[2] != nt:
        raise ValueError(
            "Adjoint file grid shape incompatible with checkpoint cfg: "
            f"cfg(nx={nx}, nt={nt}), got u0 {u0_np.shape}, u_adj {u_adj_np.shape}"
        )
    n_samples = int(u0_np.shape[0])
    u0 = torch.tensor(u0_np, device=device, dtype=torch.float32)
    u_rl, a_rl, ret_rl = rollout_policy(actor, env, u0, x_norm, t_norm)
    cost_rl = total_return_to_cost(ret_rl)

    cost_rl_np = cost_rl.cpu().numpy()
    ret_rl_np = ret_rl.cpu().numpy()

    csv_path = os.path.join(args_ns.out_dir, "comparison_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "total_return_rl", "total_cost_rl", "total_cost_numerical_adjoint"])
        for i in range(n_samples):
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
        x=adj_data["x"] if "x" in adj_data else x_np,
        t=adj_data["t"] if "t" in adj_data else t_np,
        u0=u0.cpu().numpy(),
        u_rl=u_rl.cpu().numpy(),
        a_rl=a_rl.cpu().numpy(),
        u_adj=u_adj_np,
        a_adj=a_adj_np,
        total_return_rl=ret_rl_np,
        total_cost_rl=cost_rl_np,
        total_cost_numerical_adjoint=cost_adj_np,
        numerical_adjoint_iters=np.asarray(adj_iters, dtype=np.int32),
        numerical_adjoint_success=np.asarray(adj_success, dtype=np.int8),
        checkpoint=np.array([args_ns.checkpoint], dtype=object),
        adjoint_file=np.array([args_ns.adjoint_file], dtype=object),
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
    print(f"Mean total return (RL, FNO real freq): {ret_rl_np.mean():.6f}")
    print(f"Mean total cost (RL):                  {cost_rl_np.mean():.6f}  (= -return)")
    print(f"Mean total cost (numerical adjoint):   {cost_adj_np.mean():.6f}")
    print(f"Adjoint success rate: {float(np.mean(adj_success)): .2%}")

    a_max_ckpt = float(train_args.get("a_max", 2.0))
    control_vmax = max(a_max_ckpt, 1e-8)
    control_abs_err_vmax = max(2.0 * a_max_ckpt, 1e-12)

    if args_ns.plot:
        maybe_plot(
            os.path.join(args_ns.out_dir, "comparison_u_rl.png"),
            os.path.join(args_ns.out_dir, "comparison_u_adj.png"),
            x_np,
            t_np,
            u_rl[0].cpu().numpy(),
            u_adj_np[0],
        )
        maybe_plot_control(
            os.path.join(args_ns.out_dir, "comparison_a_rl.png"),
            os.path.join(args_ns.out_dir, "comparison_a_adj.png"),
            x_np,
            t_np,
            a_rl[0].cpu().numpy(),
            a_adj_np[0],
            control_vmax,
        )
        maybe_plot_control_abs_error(
            os.path.join(args_ns.out_dir, "comparison_a_abs_error.png"),
            x_np,
            t_np,
            a_rl[0].cpu().numpy(),
            a_adj_np[0],
            control_abs_err_vmax,
        )
        print(f"Wrote {os.path.join(args_ns.out_dir, 'comparison_u_rl.png')}")
        print(f"Wrote {os.path.join(args_ns.out_dir, 'comparison_u_adj.png')}")
        print(f"Wrote {os.path.join(args_ns.out_dir, 'comparison_a_rl.png')}")
        print(f"Wrote {os.path.join(args_ns.out_dir, 'comparison_a_adj.png')}")
        print(f"Wrote {os.path.join(args_ns.out_dir, 'comparison_a_abs_error.png')}")


if __name__ == "__main__":
    main()
