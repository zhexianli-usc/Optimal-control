"""
Load Burgers BC data and plot one example as (x,t) heatmap to verify correctness.
Usage: python plot_burgers_example.py [path_to_burgers_bc_data.npz] [sample_index]
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(SCRIPT_DIR, "burgers_bc_data.npz")


def plot_example_heatmap(data_path, sample_idx=0, save_path=None):
    data = np.load(data_path, allow_pickle=True)
    u = data["u"]
    g_left = data["g_left"]
    g_right = data["g_right"]
    x = data["x"]
    t = data["t"]
    sample_idx = min(sample_idx, u.shape[0] - 1)
    u_sample = u[sample_idx]
    gl = g_left[sample_idx]
    gr = g_right[sample_idx]

    fig, axes = plt.subplots(2, 1, figsize=(6, 5), height_ratios=[1.5, 0.6])
    im = axes[0].imshow(
        u_sample.T,
        extent=[x[0], x[-1], t[0], t[-1]],
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
    )
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    axes[0].set_title(f"Burgers solution u(x,t) — sample {sample_idx}")
    plt.colorbar(im, ax=axes[0], label="u")
    axes[1].plot(t, gl, label="g_left(t)", color="C0")
    axes[1].plot(t, gr, label="g_right(t)", color="C1")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("BC value")
    axes[1].set_title("Boundary conditions")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    plt.show()


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
    sample_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if not os.path.isfile(data_path):
        print(f"Data not found: {data_path}")
        print("Generate it first: python generate_burgers_bc_data.py --samples 50 --plot")
        sys.exit(1)
    save_path = os.path.join(os.path.dirname(data_path), "burgers_example_heatmap.png")
    plot_example_heatmap(data_path, sample_idx=sample_idx, save_path=save_path)
