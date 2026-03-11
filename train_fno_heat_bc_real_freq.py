"""
Train FNO for heat optimal control using discrete Fourier transform with real frequency only.
Both forward and inverse transforms use explicit DFT/IDFT as finite sums with real frequency
and mode convention [-N/2, N/2]. No complex frequency (no θ parameters).
"""
import csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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

FNO_N_MODES = 8
FNO_N_LAYERS = 4
FNO_HIDDEN = 2
USE_CHANNEL_MLP = True
SCHEDULER_PATIENCE = 1000


def get_low_mode_indices(n, k_max):
    """Indices for low Fourier modes: 0..k_max and n-k_max..n-1."""
    k_max = min(k_max, n // 2)
    return list(range(0, k_max + 1)) + list(range(n - k_max, n))


def get_low_mode_indices_tensor(n, k_max, device):
    return torch.tensor(get_low_mode_indices(n, k_max), dtype=torch.long, device=device)


def n_low_modes(n, k_max):
    return len(get_low_mode_indices(n, k_max))


def _idx_to_k_actual(idx, N):
    """Convert DFT index [0, N) to actual frequency [-N/2, N/2]."""
    idx = idx.to(torch.float32)
    return torch.where(idx <= (N - 1) // 2, idx, idx - N)


def _dft2d_phase_fwd_real(k_actual_x, k_actual_t, H, W, device, norm=True):
    """
    Forward DFT phase with real frequency (mode [-N/2, N/2]).
    X[k,l] = (1/sqrt(H*W)) sum_{n,m} x[n,m] exp(-2πi (k_actual_x*n/H + k_actual_t*m/W)).
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


def _dft2d_phase_inv_real(k_actual_x, k_actual_t, H, W, device, norm=True):
    """
    Inverse DFT phase with real frequency (mode [-N/2, N/2]).
    x[n,m] = (1/sqrt(H*W)) sum_{k,l} X[k,l] exp(2πi (k_actual_x*n/H + k_actual_t*m/W)).
    """
    n_grid = torch.arange(H, device=device, dtype=torch.float32)
    m_grid = torch.arange(W, device=device, dtype=torch.float32)
    kn_H = k_actual_x.unsqueeze(1) * n_grid.unsqueeze(0) / H
    lm_W = k_actual_t.unsqueeze(1) * m_grid.unsqueeze(0) / W
    phase = kn_H.unsqueeze(1).unsqueeze(3) + lm_W.unsqueeze(0).unsqueeze(2)
    phase = torch.exp(2.0 * np.pi * 1j * phase)
    if norm:
        phase = phase / (H * W) ** 0.5
    return phase


class SpectralConv2dRealFreq(nn.Module):
    """
    2D spectral convolution using explicit DFT/IDFT with real frequency only.
    Mode convention [-N/2, N/2]. No learnable θ; both forward and inverse use real k_actual.
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

    def forward(self, x):
        B, C, H, W = x.shape
        idx_x = get_low_mode_indices_tensor(H, self.k_max_x, x.device)
        idx_t = get_low_mode_indices_tensor(W, self.k_max_t, x.device)
        k_actual_x = _idx_to_k_actual(idx_x, H)
        k_actual_t = _idx_to_k_actual(idx_t, W)

        phase_fwd = _dft2d_phase_fwd_real(k_actual_x, k_actual_t, H, W, x.device)
        x_complex = x.to(torch.complex64)
        X = (x_complex.unsqueeze(2).unsqueeze(3) * phase_fwd.unsqueeze(0).unsqueeze(0)).sum(dim=(-2, -1))

        R = self.weights_real + 1j * self.weights_imag
        out_kept = torch.einsum("bcij,coij->boij", X, R)

        phase_inv = _dft2d_phase_inv_real(k_actual_x, k_actual_t, H, W, x.device)
        out = (out_kept.unsqueeze(-1).unsqueeze(-1) * phase_inv.unsqueeze(0).unsqueeze(0)).sum(dim=(2, 3))
        return out.real


class FNO2d(nn.Module):
    """2D FNO with real-frequency spectral conv: lift -> [spectral + channel_mlp + skip] x n_layers -> project."""

    def __init__(self, in_ch, out_ch, width, n_x, n_t, k_max, n_layers, use_channel_mlp=True):
        super().__init__()
        self.lift = nn.Conv2d(in_ch, width, 1)
        modes_x = n_low_modes(n_x, k_max)
        modes_t = n_low_modes(n_t, k_max)
        self.blocks = nn.ModuleList()
        self.use_channel_mlp = use_channel_mlp
        for _ in range(n_layers):
            block = nn.ModuleDict({
                "spectral": SpectralConv2dRealFreq(width, width, modes_x, modes_t, k_max, k_max),
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
    """BC encoding -> 2D field (n_samples, in_channels, n_x_pts, n_t_pts)."""
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


def main():
    bc_enc, m_opts, x, t = load_or_generate_data(n_samples=40, bc_type="dirichlet")
    n_samples = bc_enc.shape[0]
    n_x_pts = len(x)
    n_t_pts = len(t)
    n_coords = n_x_pts * n_t_pts
    in_field_np = bc_enc_to_field(bc_enc, n_x_pts, n_t_pts)
    in_channels = in_field_np.shape[1]
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
        m_mean, m_std = 0.0, 1.0

    in_field_t = torch.tensor(in_field_np, dtype=torch.float32, device=DEVICE)
    m_target_t = torch.tensor(m_flat, dtype=torch.float32, device=DEVICE)

    k_max = min(FNO_N_MODES, n_x_pts // 2, n_t_pts // 2)
    if k_max < 1:
        k_max = 1
    net = FNO2d(
        in_channels, 1, FNO_HIDDEN,
        n_x_pts, n_t_pts, k_max, FNO_N_LAYERS,
        use_channel_mlp=USE_CHANNEL_MLP,
    ).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=SCHEDULER_PATIENCE, min_lr=1e-6
    )

    indices = np.arange(n_samples)
    loss_hist = []
    loss_every_500 = []
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
        if (ep + 1) % 500 == 0:
            loss_every_500.append((ep + 1, avg_loss))
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
    with torch.no_grad():
        pred_all = net(in_field_t)
        pred_flat_all = pred_all.reshape(pred_all.shape[0], -1)
        pred_orig = pred_flat_all.cpu().numpy() * m_std + m_mean
        m_orig_np = m_target_t.cpu().numpy() * m_std + m_mean
        test_mse_orig = ((pred_orig - m_orig_np) ** 2).mean()
    print(f"  Final test MSE (original scale): {test_mse_orig:.6e}")

    # Save loss every 500 epochs to CSV
    loss_500_csv = "train_fno_heat_bc_real_freq_loss_every_500.csv"
    with open(loss_500_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])
        for ep, loss_val in loss_every_500:
            writer.writerow([ep, loss_val])
    print(f"  Loss every 500 epochs saved to {loss_500_csv}")

    np.savetxt(
        "train_fno_heat_bc_real_freq_loss_plot_data.csv",
        np.column_stack([np.arange(1, len(loss_hist) + 1), loss_hist]),
        delimiter=",",
        header="epoch,loss",
        comments="",
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.semilogy(np.arange(1, len(loss_hist) + 1), loss_hist, color="C0", linewidth=0.5, alpha=0.8)
    ax1.set_ylabel("MSE (normalized)")
    ax1.set_title("FNO (real frequency) training MSE")
    ax1.grid(True, alpha=0.3)
    if mse_orig_hist:
        ax2.semilogy(mse_orig_epochs, mse_orig_hist, "o-", color="C1", markersize=4)
    ax2.set_ylabel("MSE (original scale)")
    ax2.set_xlabel("Epoch")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("train_fno_real_freq_mse.png", dpi=150)
    plt.close()
    print("  Plot saved to train_fno_real_freq_mse.png")

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
    plt.savefig("train_fno_heat_bc_real_freq_bc_comparison.png", dpi=150)
    plt.close()
    print("  BC comparison plot saved to train_fno_heat_bc_real_freq_bc_comparison.png")
    with open("train_fno_heat_bc_real_freq_bc_comparison_data.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["example", "x", "t", "pred", "target", "diff"])
        writer.writeheader()
        writer.writerows(bc_comparison_rows)
    print("  BC comparison data saved to train_fno_heat_bc_real_freq_bc_comparison_data.csv")
    return net, loss_hist


if __name__ == "__main__":
    print("FNO (discrete Fourier transform, real frequency only)")
    print(f"  Device: {DEVICE}, hidden: {FNO_HIDDEN}, n_modes: {FNO_N_MODES}, n_layers: {FNO_N_LAYERS}")
    net, loss_hist = main()
