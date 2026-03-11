"""
Train a Fourier Neural Operator (FNO) to map boundary condition -> optimal control
for the 1D heat equation. Uses the same data and parameters as train_deeponet_heat_bc.py.

Ref: Li et al. "Fourier Neural Operator for Parametric PDEs" (ICLR 2021)
     https://arxiv.org/pdf/2512.01421 (FNO Explained: A Practical Perspective)

Input: BC encoded as a 2D field on (x, t) — for GRF, g_left(t) and g_right(t) as two channels
       broadcast in x; for constant BC, four constant channels.
Output: m(x, t) = optimal control on the grid.
Supervised loss: MSE between FNO(BC_field)(x,t) and precomputed optimal m from data.
"""
import csv
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Import shared parameters and data loading from DeepONet training script
from train_deeponet_heat_bc import (
    DATA_FILE,
    DEVICE,
    EPOCHS,
    BATCH_SIZE,
    LR,
    USE_TARGET_NORMALIZATION,
    HIDDEN,
    load_or_generate_data,
)

# FNO-specific hyperparameters (see arxiv.org/pdf/2512.01421 Sec 4)
# To improve convergence (lower MSE): increase FNO_HIDDEN (e.g. 64), FNO_N_MODES (e.g. 14),
# FNO_N_LAYERS (e.g. 6), use more training data, or train longer (EPOCHS).
FNO_N_MODES = 8   # Fourier modes per dimension (keep below Nyquist; increase for finer scales)
FNO_N_LAYERS = 4   # Number of FNO blocks (deeper = more capacity)
FNO_HIDDEN = 16    # Hidden width for FNO (larger = more capacity, try 64 if underfitting)
USE_CHANNEL_MLP = True   # 1x1 channel mixing per block (Sec 3.4.3: helps high-freq details)
SCHEDULER_PATIENCE = 1000   # Epochs before LR drop (smaller = earlier fine-tuning)
OUT_DIR = "train_fno_heat_bc_output"   # Folder for all saved data and plots


def get_low_mode_indices(n, k_max):
    """Indices for low Fourier modes: 0..k_max and n-k_max..n-1."""
    k_max = min(k_max, n // 2)
    return list(range(0, k_max + 1)) + list(range(n - k_max, n))


def get_low_mode_indices_tensor(n, k_max, device):
    return torch.tensor(get_low_mode_indices(n, k_max), dtype=torch.long, device=device)


def n_low_modes(n, k_max):
    return len(get_low_mode_indices(n, k_max))


def _idx_to_k_actual(idx, N):
    """Convert DFT index in [0, N) to actual frequency in [-N/2, N/2]. k_actual = idx for idx <= (N-1)//2, else idx - N."""
    idx = idx.to(torch.float32)
    return torch.where(idx <= (N - 1) // 2, idx, idx - N)


def _theta_from_raw(k_actual, theta_raw):
    """
    Map raw parameters to θ. Uses actual frequency sign (k_actual in [-N/2, N/2]).
    Positive modes (k_actual >= 0): θ ∈ (0, π],  θ = π * sigmoid(raw).
    Negative modes (k_actual < 0):  θ ∈ [π/2, π], θ = π/2 * (1 + sigmoid(raw)).
    """
    sig = torch.sigmoid(theta_raw)
    is_negative = (k_actual < 0).to(theta_raw.dtype)
    theta = (1.0 - is_negative) * (np.pi * sig) + is_negative * (np.pi / 2.0 * (1.0 + sig))
    return theta


def _dft2d_phase_fwd_real(k_actual_x, k_actual_t, H, W, device, norm=True):
    """
    Forward DFT phase with real frequency (mode convention [-N/2, N/2]).
    Phase = exp(-2πi * (k_actual_x*n/H + k_actual_t*m/W)) / sqrt(H*W).
    Returns (modes_x, modes_t, H, W) complex.
    """
    n_grid = torch.arange(H, device=device, dtype=torch.float32)
    m_grid = torch.arange(W, device=device, dtype=torch.float32)
    kn_H = k_actual_x.unsqueeze(1) * n_grid.unsqueeze(0) / H
    lm_W = k_actual_t.unsqueeze(1) * m_grid.unsqueeze(0) / W
    phase = kn_H.unsqueeze(1).unsqueeze(3) + lm_W.unsqueeze(0).unsqueeze(2)
    phase = torch.exp(-2.0 * np.pi * 1j * phase)
    if norm:
        phase = phase / (H * W) ** 0.5
    return phase


def _dft2d_phase_inv_complex_freq(k_actual_x, k_actual_t, theta_x, theta_t, H, W, device, norm=True):
    """
    Inverse DFT phase with complex frequency k_x = |k_x| e^{iθ} (only in inverse).
    k_complex = |k_actual| * exp(i*θ). Phase = exp(2πi * z) / sqrt(H*W), z = k_cx*n/H + k_ct*m/W.
    Returns (modes_x, modes_t, H, W) complex.
    """
    n_grid = torch.arange(H, device=device, dtype=torch.float32)
    m_grid = torch.arange(W, device=device, dtype=torch.float32)
    k_actual_x = k_actual_x.to(device)
    k_actual_t = k_actual_t.to(device)
    theta_x = theta_x.to(device)
    theta_t = theta_t.to(device)
    k_cx = torch.abs(k_actual_x) * torch.exp(1j * theta_x)
    k_ct = torch.abs(k_actual_t) * torch.exp(1j * theta_t)
    z = (
        k_cx.unsqueeze(1).unsqueeze(2).unsqueeze(3) * n_grid.view(1, 1, H, 1) / H
        + k_ct.unsqueeze(0).unsqueeze(2).unsqueeze(3) * m_grid.view(1, 1, 1, W) / W
    )
    phase = torch.exp(2.0 * np.pi * 1j * z)
    if norm:
        phase = phase / (H * W) ** 0.5
    return phase


class SpectralConv2d(nn.Module):
    """
    Forward: real frequency, mode convention [-N/2, N/2]. Inverse: complex frequency
    k_x = |k_x| e^{iθ} (θ learnable). Positive modes θ ∈ (0, π]; negative θ ∈ [π/2, π].
    """

    def __init__(self, in_ch, out_ch, modes_x, modes_t, k_max_x, k_max_t):
        super().__init__()
        self.modes_x = modes_x
        self.modes_t = modes_t
        self.k_max_x = k_max_x
        self.k_max_t = k_max_t
        scale = 1.0 / (in_ch * out_ch)
        self.weights_real = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes_x, modes_t))
        self.weights_imag = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes_x, modes_t))
        self.theta_raw_x = nn.Parameter(torch.zeros(modes_x))
        self.theta_raw_t = nn.Parameter(torch.zeros(modes_t))

    def forward(self, x):
        B, C, H, W = x.shape
        idx_x = get_low_mode_indices_tensor(H, self.k_max_x, x.device)
        idx_t = get_low_mode_indices_tensor(W, self.k_max_t, x.device)
        modes_x, modes_t = len(idx_x), len(idx_t)

        k_actual_x = _idx_to_k_actual(idx_x, H)
        k_actual_t = _idx_to_k_actual(idx_t, W)

        phase_fwd = _dft2d_phase_fwd_real(k_actual_x, k_actual_t, H, W, x.device)
        x_complex = x.to(torch.complex64)
        X = (x_complex.unsqueeze(2).unsqueeze(3) * phase_fwd.unsqueeze(0).unsqueeze(0)).sum(dim=(-2, -1))

        R = self.weights_real + 1j * self.weights_imag
        out_kept = torch.einsum("bcij,coij->boij", X, R)

        theta_x = _theta_from_raw(k_actual_x, self.theta_raw_x)
        theta_t = _theta_from_raw(k_actual_t, self.theta_raw_t)
        phase_inv = _dft2d_phase_inv_complex_freq(
            k_actual_x, k_actual_t, theta_x, theta_t, H, W, x.device
        )
        out = (out_kept.unsqueeze(-1).unsqueeze(-1) * phase_inv.unsqueeze(0).unsqueeze(0)).sum(dim=(2, 3))
        return out.real


class FNO2d(nn.Module):
    """2D FNO: lift -> [spectral + channel_mlp + skip] x n_layers -> project. GELU for smoother gradients."""

    def __init__(self, in_ch, out_ch, width, n_x, n_t, k_max, n_layers, use_channel_mlp=True):
        super().__init__()
        self.lift = nn.Conv2d(in_ch, width, 1)
        modes_x = n_low_modes(n_x, k_max)
        modes_t = n_low_modes(n_t, k_max)
        self.blocks = nn.ModuleList()
        self.use_channel_mlp = use_channel_mlp
        for _ in range(n_layers):
            block = nn.ModuleDict({
                "spectral": SpectralConv2d(width, width, modes_x, modes_t, k_max, k_max),
                "skip": nn.Conv2d(width, width, 1),
            })
            if use_channel_mlp:
                block["channel_mlp"] = nn.Sequential(
                    nn.Conv2d(width, width, 1),
                    nn.GELU(),
                    nn.Conv2d(width, width, 1),
                )
            self.blocks.append(block)
        self.project = nn.Conv2d(width, out_ch, 1)

    def forward(self, x):
        x = self.lift(x)
        for block in self.blocks:
            x_spectral = block["spectral"](x)
            if self.use_channel_mlp:
                x_spectral = block["channel_mlp"](x_spectral)
            x = torch.nn.functional.gelu(block["skip"](x) + x_spectral)
        return self.project(x)


def bc_enc_to_field(bc_enc, n_x_pts, n_t_pts):
    """
    Convert BC encoding to 2D input field for FNO.
    - GRF format: bc_enc (n_samples, 2*n_t_pts) -> 2 channels: g_left(t), g_right(t) broadcast in x.
    - Constant format: bc_enc (n_samples, 4) -> 4 constant channels over (x,t).
    Returns (n_samples, in_channels, n_x_pts, n_t_pts).
    """
    n_samples = bc_enc.shape[0]
    bc_dim = bc_enc.shape[1]
    if bc_dim == 4:
        in_channels = 4
        field = np.broadcast_to(
            bc_enc[:, :, None, None],
            (n_samples, in_channels, n_x_pts, n_t_pts),
        ).astype(np.float32).copy()
    else:
        in_channels = 2
        n_t_bc = bc_dim // 2
        g_left = bc_enc[:, :n_t_bc]
        g_right = bc_enc[:, n_t_bc:]
        field = np.zeros((n_samples, in_channels, n_x_pts, n_t_pts), dtype=np.float32)
        field[:, 0, :, :] = g_left[:, None, :]
        field[:, 1, :, :] = g_right[:, None, :]
    return field


def load_fno_checkpoint(path, device=None):
    """
    Load a saved FNO checkpoint for inference or comparison.
    Returns: (net, config, x, t)
    - net: FNO2d in eval mode, on device
    - config: dict with m_mean, m_std, n_x_pts, n_t_pts, etc.
    - x, t: 1D numpy arrays (grid)
    Example:
      net, config, x, t = load_fno_checkpoint("train_fno_heat_bc_output/fno_heat_bc_model.pt")
      in_field = bc_enc_to_field(bc_enc_new, config["n_x_pts"], config["n_t_pts"])
      with torch.no_grad():
          pred = net(torch.tensor(in_field, device=net...)) * config["m_std"] + config["m_mean"]
    """
    if device is None:
        device = DEVICE
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    net = FNO2d(
        cfg["in_channels"], 1, cfg["width"],
        cfg["n_x_pts"], cfg["n_t_pts"], cfg["k_max"], cfg["n_layers"],
        use_channel_mlp=cfg["use_channel_mlp"],
    ).to(device)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, cfg, ck["x"], ck["t"]


def main():
    bc_enc, m_opts, x, t = load_or_generate_data(n_samples=40, bc_type="dirichlet")

    n_samples = bc_enc.shape[0]
    bc_dim = bc_enc.shape[1]
    n_x_pts = len(x)
    n_t_pts = len(t)
    n_coords = n_x_pts * n_t_pts

    # Build 2D input field (n_samples, in_channels, n_x_pts, n_t_pts)
    in_field_np = bc_enc_to_field(bc_enc, n_x_pts, n_t_pts)
    in_channels = in_field_np.shape[1]

    # Targets: (n_samples, n_x_pts, n_t_pts) -> (n_samples, n_coords)
    m_flat = m_opts.reshape(n_samples, -1).astype(np.float32)
    assert m_flat.shape[1] == n_coords

    if USE_TARGET_NORMALIZATION:
        m_mean = float(m_flat.mean())
        m_std = float(m_flat.std())
        if m_std < 1e-8:
            m_std = 1.0
        m_flat = (m_flat - m_mean) / m_std
        print(f"  Target normalization: mean = {m_mean:.4f}, std = {m_std:.4f}")
    else:
        m_mean = 0.0
        m_std = 1.0

    in_field_t = torch.tensor(in_field_np, dtype=torch.float32, device=DEVICE)
    m_target_t = torch.tensor(m_flat, dtype=torch.float32, device=DEVICE)

    # FNO: input (B, in_channels, n_x_pts, n_t_pts) -> output (B, 1, n_x_pts, n_t_pts)
    k_max = min(FNO_N_MODES, n_x_pts // 2, n_t_pts // 2)
    if k_max < 1:
        k_max = 1
    net = FNO2d(
        in_channels, 1, FNO_HIDDEN,
        n_x_pts, n_t_pts,
        k_max, FNO_N_LAYERS,
        use_channel_mlp=USE_CHANNEL_MLP,
    ).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=SCHEDULER_PATIENCE, min_lr=1e-6
    )

    indices = np.arange(n_samples)
    loss_hist = []
    mse_orig_hist = []
    mse_orig_epochs = []
    for ep in range(EPOCHS):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_samples, BATCH_SIZE):
            idx = indices[start : start + BATCH_SIZE]
            inp_batch = in_field_t[idx]
            m_batch = m_target_t[idx]
            pred = net(inp_batch)
            pred_flat = pred.reshape(pred.shape[0], -1)
            loss = ((pred_flat - m_batch) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        loss_hist.append(avg_loss)
        scheduler.step(avg_loss)
        if (ep + 1) % 5 == 0 or ep == 0:
            with torch.no_grad():
                pred_all = net(in_field_t)
                pred_flat_all = pred_all.reshape(pred_all.shape[0], -1)
                pred_orig = pred_flat_all.cpu().numpy() * m_std + m_mean
                m_orig = m_target_t.cpu().numpy() * m_std + m_mean
                mse_orig = ((pred_orig - m_orig) ** 2).mean()
            mse_orig_hist.append(float(mse_orig))
            mse_orig_epochs.append(ep + 1)
            print(
                f"  Epoch {ep+1}/{EPOCHS}, train MSE (norm) = {avg_loss:.6e}, "
                f"MSE (orig) = {mse_orig:.6e}, lr = {opt.param_groups[0]['lr']:.2e}"
            )

    print("\nTraining finished.")
    if len(loss_hist) >= 2 and loss_hist[-1] < loss_hist[0]:
        print("  Loss decreased: training successful.")
    else:
        print("  (Loss may need more epochs or data.)")

    with torch.no_grad():
        pred_all = net(in_field_t)
        pred_flat_all = pred_all.reshape(pred_all.shape[0], -1)
        pred_orig = pred_flat_all.cpu().numpy() * m_std + m_mean
        m_orig_np = m_target_t.cpu().numpy() * m_std + m_mean
        test_mse_norm = ((pred_flat_all - m_target_t) ** 2).mean().item()
        test_mse_orig = ((pred_orig - m_orig_np) ** 2).mean()
    print(f"  Final test MSE (normalized): {test_mse_norm:.6e}")
    print(f"  Final test MSE (original scale): {test_mse_orig:.6e}")

    os.makedirs(OUT_DIR, exist_ok=True)

    # Save trained model and config for later comparison / inference
    checkpoint = {
        "state_dict": net.state_dict(),
        "config": {
            "in_channels": in_channels,
            "n_x_pts": n_x_pts,
            "n_t_pts": n_t_pts,
            "k_max": int(k_max),
            "width": FNO_HIDDEN,
            "n_layers": FNO_N_LAYERS,
            "use_channel_mlp": USE_CHANNEL_MLP,
            "m_mean": m_mean,
            "m_std": m_std,
        },
        "x": np.asarray(x),
        "t": np.asarray(t),
    }
    model_path = os.path.join(OUT_DIR, "fno_heat_bc_model.pt")
    torch.save(checkpoint, model_path)
    print(f"  Trained model saved to {model_path} (load for future comparisons)")

    # Save loss for every epoch to CSV
    loss_csv = os.path.join(OUT_DIR, "train_fno_heat_bc_loss_history.csv")
    with open(loss_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])
        for ep, loss_val in enumerate(loss_hist, start=1):
            writer.writerow([ep, loss_val])
    print(f"  Loss history (every epoch) saved to {OUT_DIR}/train_fno_heat_bc_loss_history.csv")

    # Save full loss curve data for plotting
    np.savetxt(
        os.path.join(OUT_DIR, "train_fno_heat_bc_loss_plot_data.csv"),
        np.column_stack([np.arange(1, len(loss_hist) + 1), loss_hist]),
        delimiter=",",
        header="epoch,loss",
        comments="",
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    epochs_arr = np.arange(1, len(loss_hist) + 1)
    ax1.semilogy(epochs_arr, loss_hist, color="C0", linewidth=0.5, alpha=0.8)
    ax1.set_ylabel("MSE (normalized)")
    ax1.set_title("FNO training MSE (normalized targets)")
    ax1.grid(True, alpha=0.3)
    if mse_orig_hist:
        ax2.semilogy(mse_orig_epochs, mse_orig_hist, "o-", color="C1", markersize=4)
    ax2.set_ylabel("MSE (original scale)")
    ax2.set_xlabel("Epoch")
    ax2.set_title("Test MSE (original scale, every 2000 epochs)")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "train_fno_mse.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Training MSE plot saved to {OUT_DIR}/train_fno_mse.png")

    # Plot pred vs training data for example BCs and save plotting data
    n_examples = min(3, n_samples)
    example_indices = np.linspace(0, n_samples - 1, n_examples, dtype=int)
    pred_all_np = pred_flat_all.cpu().numpy() * m_std + m_mean
    m_orig_np_3d = m_orig_np.reshape(n_samples, n_x_pts, n_t_pts)

    fig2, axes = plt.subplots(n_examples, 3, figsize=(12, 3 * n_examples))
    if n_examples == 1:
        axes = axes.reshape(1, -1)
    bc_comparison_rows = []
    for i, ex in enumerate(example_indices):
        pred_2d = pred_all_np[ex].reshape(n_x_pts, n_t_pts)
        target_2d = m_orig_np_3d[ex]
        diff_2d = pred_2d - target_2d
        for ix in range(n_x_pts):
            for it in range(n_t_pts):
                bc_comparison_rows.append({
                    "example": ex, "x": x[ix], "t": t[it],
                    "pred": pred_2d[ix, it], "target": target_2d[ix, it], "diff": diff_2d[ix, it],
                })
        # Plot slice at mid-x: pred(t), target(t), diff(t)
        mid_x = n_x_pts // 2
        axes[i, 0].plot(t, pred_2d[mid_x, :], "b-", label="FNO pred")
        axes[i, 0].plot(t, target_2d[mid_x, :], "r--", label="Training data")
        axes[i, 0].set_xlabel("t")
        axes[i, 0].set_ylabel("m(x_mid, t)")
        axes[i, 0].set_title(f"BC example {ex}: pred vs target")
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 1].plot(t, diff_2d[mid_x, :], "g-")
        axes[i, 1].axhline(0, color="k", linestyle=":", alpha=0.5)
        axes[i, 1].set_xlabel("t")
        axes[i, 1].set_ylabel("pred - target")
        axes[i, 1].set_title(f"BC example {ex}: error at x_mid")
        axes[i, 1].grid(True, alpha=0.3)
        im = axes[i, 2].imshow(diff_2d, aspect="auto", origin="lower",
                               extent=[t[0], t[-1], x[0], x[-1]], cmap="RdBu_r")
        axes[i, 2].set_xlabel("t")
        axes[i, 2].set_ylabel("x")
        axes[i, 2].set_title(f"BC example {ex}: error field")
        plt.colorbar(im, ax=axes[i, 2], label="pred - target")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "train_fno_heat_bc_bc_comparison.png"), dpi=150)
    plt.close()
    print(f"  BC comparison plot saved to {OUT_DIR}/train_fno_heat_bc_bc_comparison.png")
    with open(os.path.join(OUT_DIR, "train_fno_heat_bc_bc_comparison_data.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["example", "x", "t", "pred", "target", "diff"])
        writer.writeheader()
        writer.writerows(bc_comparison_rows)
    print(f"  BC comparison data saved to {OUT_DIR}/train_fno_heat_bc_bc_comparison_data.csv")

    return net, loss_hist


if __name__ == "__main__":
    print("FNO: BC field -> optimal control (supervised MSE)")
    print(f"  Device: {DEVICE}, hidden: {FNO_HIDDEN}, n_modes: {FNO_N_MODES}, n_layers: {FNO_N_LAYERS}")
    print(f"  channel_mlp: {USE_CHANNEL_MLP}, scheduler_patience: {SCHEDULER_PATIENCE}")
    print(f"  (DATA_FILE, EPOCHS, BATCH_SIZE, LR from train_deeponet_heat_bc)")
    net, loss_hist = main()
