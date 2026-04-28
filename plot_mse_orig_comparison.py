from pathlib import Path
import csv
import matplotlib.pyplot as plt

from plot_style import apply_plot_style

apply_plot_style()


def read_epoch_mse(csv_path: Path):
    epochs = []
    mses = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(float(row["epoch"]))
            mses.append(float(row["mse"]))
    return epochs, mses


def main():
    base_dir = Path(__file__).resolve().parent
    # csv_orig = base_dir / "train_fno_heat_bc_output" / "train_fno_heat_bc_mse_orig_hist.csv"
    # csv_real_freq = base_dir / "train_fno_heat_bc_real_freq_output" / "train_fno_heat_bc_mse_orig_hist.csv"

    # csv_orig = base_dir / "burgers_bc_data" / "train_fno_burgers_bc_output" / "train_fno_burgers_bc_mse_orig_hist.csv"
    # csv_real_freq = base_dir / "burgers_bc_data" / "train_fno_burgers_bc_real_freq_output" / "train_fno_burgers_bc_real_freq_mse_orig_hist.csv"

    csv_orig = base_dir / "burgers_bc_data" / "train_fno_burgers_optimal_control" / "train_fno_burgers_bc_mse_orig_hist.csv"
    csv_real_freq = base_dir / "burgers_bc_data" / "train_fno_burgers_real_freq_output_optimal_control" / "train_fno_burgers_bc_real_freq_mse_orig_hist.csv"

    epoch_orig, mse_orig = read_epoch_mse(csv_orig)
    epoch_real_freq, mse_real_freq = read_epoch_mse(csv_real_freq)

    plt.figure(figsize=(8, 5))
    # ax2.semilogy(mse_orig_epochs, mse_orig_hist, "o-", color="C1", markersize=4)
    plt.semilogy(epoch_orig, mse_orig, "o-", label=r"Extended FNO$^{\angle\theta}$", markersize=4)
    plt.semilogy(epoch_real_freq, mse_real_freq, "o-", label="Classical FNO", markersize=4)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Relative MSE")
    # plt.title("MSE Original History Comparison (Log Scale)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # out_dir = '/Users/zhexianli/Desktop/Research/Manuscript/Ehrenpreis Neural Operator /CDC 2026/fig'
    out_dir = '/Users/zhexianli/Dropbox/Apps/Overleaf/Presentation-slides/fig'

    # output_path = Path(out_dir) / "Rel_burgers_state_comparison.png"
    output_path = Path(out_dir) / "Rel_burgers_control_comparison.png"
    plt.savefig(output_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
