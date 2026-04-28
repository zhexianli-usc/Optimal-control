import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_plot_style

apply_plot_style()


def load_rows(csv_path: Path):
    rows = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "example": int(float(row["example"])),
                    "x": float(row["x"]),
                    "t": float(row["t"]),
                    "pred": float(row["pred"]),
                    "target": float(row["target"]),
                    "diff": float(row["diff"]),
                }
            )
    return rows


def rows_to_grids(rows):
    examples = sorted({r["example"] for r in rows})
    x_vals = np.array(sorted({r["x"] for r in rows}), dtype=float)
    t_vals = np.array(sorted({r["t"] for r in rows}), dtype=float)

    x_to_idx = {v: i for i, v in enumerate(x_vals)}
    t_to_idx = {v: i for i, v in enumerate(t_vals)}

    data = {}
    for ex in examples:
        pred = np.zeros((len(x_vals), len(t_vals)), dtype=float)
        target = np.zeros((len(x_vals), len(t_vals)), dtype=float)
        diff = np.zeros((len(x_vals), len(t_vals)), dtype=float)
        data[ex] = {"pred": pred, "target": target, "diff": diff}

    for row in rows:
        ex = row["example"]
        ix = x_to_idx[row["x"]]
        it = t_to_idx[row["t"]]
        data[ex]["pred"][ix, it] = row["pred"]
        data[ex]["target"][ix, it] = row["target"]
        data[ex]["diff"][ix, it] = row["diff"]

    return examples, x_vals, t_vals, data


def save_example_plot(ex, x_vals, t_vals, complex_data, real_data, out_dir: Path):
    diff_complex = complex_data["diff"]

    diff_real = real_data["diff"]

    field_abs_max = max(np.abs(diff_complex).max(), np.abs(diff_real).max())
    field_abs_max = max(field_abs_max, 1e-12)

    methods = [
        ("complex_frequency", diff_complex),
        ("real_frequency", diff_real),
    ]

    for method_slug, diff_2d in methods:
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.2))
        im = ax.imshow(
            diff_2d,
            aspect="auto",
            origin="lower",
            extent=[t_vals[0], t_vals[-1], x_vals[0], x_vals[-1]],
            cmap="Oranges",
            vmin=0,
            vmax=field_abs_max,
        )
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$x$")
        plt.colorbar(im, ax=ax, label=r"Absolute Error")
        plt.tight_layout(pad=0)

        # out_path = out_dir / f"burgers_state_example_{ex}_{method_slug}.png"
        out_path = out_dir / f"burgers_control_example_{ex}_{method_slug}.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


def main():
    base_dir = Path(__file__).resolve().parent
    # csv_complex = base_dir / "train_fno_heat_bc_output" / "train_fno_heat_bc_bc_comparison_data.csv"
    # csv_real = base_dir / "train_fno_heat_bc_real_freq_output" / "train_fno_heat_bc_real_freq_bc_comparison_data.csv"

    # csv_complex = base_dir / "burgers_bc_data" / "train_fno_burgers_bc_output" / "train_fno_heat_bc_real_freq_bc_comparison_data.csv"
    # csv_real    = base_dir / "burgers_bc_data" / "train_fno_burgers_bc_real_freq_output" / "train_fno_heat_bc_real_freq_bc_comparison_data.csv"

    csv_complex = base_dir / "burgers_bc_data" / "train_fno_burgers_optimal_control" / "train_fno_heat_bc_real_freq_bc_comparison_data.csv"

    csv_real    = base_dir / "burgers_bc_data" / "train_fno_burgers_real_freq_output_optimal_control" /"train_fno_heat_bc_real_freq_bc_comparison_data.csv"

    rows_complex = load_rows(csv_complex)
    rows_real = load_rows(csv_real)

    examples_complex, x_complex, t_complex, data_complex = rows_to_grids(rows_complex)
    examples_real, x_real, t_real, data_real = rows_to_grids(rows_real)

    if examples_complex != examples_real:
        raise ValueError("Example IDs in the two CSV files do not match.")
    if not np.allclose(x_complex, x_real):
        raise ValueError("x grids in the two CSV files do not match.")
    if not np.allclose(t_complex, t_real):
        raise ValueError("t grids in the two CSV files do not match.")

    # out_dir = Path("/Users/zhexianli/Desktop/Research/Manuscript/Ehrenpreis Neural Operator /CDC 2026/fig")
    # out_dir = Path('/Users/zhexianli/Desktop/Research/Presentations/Presentation slides/Job talk 2026/fig')

    out_dir = Path('/Users/zhexianli/Dropbox/Apps/Overleaf/Presentation-slides/fig')
    out_dir.mkdir(parents=True, exist_ok=True)

    for ex in examples_complex:
        if ex == 24:
            save_example_plot(ex, x_complex, t_complex, data_complex[ex], data_real[ex], out_dir)

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
