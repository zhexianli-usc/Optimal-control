"""
Plot RL training convergence from `train_log.csv`.

Creates a 3-panel figure:
1) mean episode return (higher is better),
2) mean episode cost = -mean return (lower is better),
3) critic/actor losses.
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) < w:
        return x.copy()
    kernel = np.ones(w, dtype=np.float64) / float(w)
    y = np.convolve(x, kernel, mode="valid")
    # Left-pad so output aligns with episodes
    pad = np.full(w - 1, y[0], dtype=y.dtype)
    return np.concatenate([pad, y], axis=0)


def read_log(csv_path: str):
    ep, q, a, r = [], [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep.append(int(row["episode"]))
            q.append(float(row["q_loss"]))
            a.append(float(row["actor_loss"]))
            r.append(float(row["mean_return"]))
    return np.array(ep), np.array(q), np.array(a), np.array(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=str, default="heat_rl_efno_output/train_log.csv")
    p.add_argument("--out", type=str, default="heat_rl_efno_output/training_convergence.png")
    p.add_argument("--smooth-window", type=int, default=20)
    args = p.parse_args()

    ep, q_loss, actor_loss, mean_ret = read_log(args.log)
    if len(ep) == 0:
        raise ValueError(f"No rows found in {args.log}")

    cost = -mean_ret
    ret_s = moving_average(mean_ret, args.smooth_window)
    cost_s = moving_average(cost, args.smooth_window)
    q_s = moving_average(q_loss, args.smooth_window)
    a_s = moving_average(actor_loss, args.smooth_window)

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True, constrained_layout=True)

    axes[0].plot(ep, mean_ret, color="#4c78a8", alpha=0.35, label="raw")
    axes[0].plot(ep, ret_s, color="#1f4f8b", lw=2, label=f"MA({args.smooth_window})")
    axes[0].set_ylabel("Mean Return")
    axes[0].set_title("Return (higher is better)")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    axes[1].plot(ep, cost, color="#f58518", alpha=0.35, label="raw")
    axes[1].plot(ep, cost_s, color="#b35a00", lw=2, label=f"MA({args.smooth_window})")
    axes[1].set_ylabel("Mean Cost = -Return")
    axes[1].set_title("Objective Cost (lower is better)")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)

    axes[2].plot(ep, q_loss, color="#54a24b", alpha=0.30, label="Q loss (raw)")
    axes[2].plot(ep, q_s, color="#2f7d32", lw=2, label="Q loss (smoothed)")
    axes[2].plot(ep, actor_loss, color="#e45756", alpha=0.30, label="Actor loss (raw)")
    axes[2].plot(ep, a_s, color="#b13333", lw=2, label="Actor loss (smoothed)")
    axes[2].set_ylabel("Loss")
    axes[2].set_xlabel("Episode")
    axes[2].set_title("Optimization Diagnostics")
    axes[2].legend(frameon=False, ncol=2)
    axes[2].grid(alpha=0.25)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)

    print(f"Wrote {args.out}")
    print(f"Episodes: {len(ep)}")
    print(f"Final mean return: {mean_ret[-1]:.6f}")
    print(f"Final mean cost: {cost[-1]:.6f}")


if __name__ == "__main__":
    main()
