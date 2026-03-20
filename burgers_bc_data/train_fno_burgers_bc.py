"""
Train FNO to map boundary condition -> solution u(x,t) for the 1D Burgers equation.
Follows train_fno_heat_bc.py: same FNO (complex-frequency inverse), BC as 2D field.
Input: g_left(t), g_right(t) as 2 channels on (x,t). Output: u(x,t).
Data: burgers_bc_data/burgers_bc_data.npz (g_left, g_right, u, x, t).
"""
import csv
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "burgers_bc_data.npz")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50000
BATCH_SIZE = 16
LR = 1e-3
USE_TARGET_NORMALIZATION = True
FNO_N_MODES = 8
FNO_N_LAYERS = 4
FNO_HIDDEN = 2
USE_CHANNEL_MLP = True
SCHEDULER_PATIENCE = 1000
OUT_DIR = os.path.join(SCRIPT_DIR, "train_fno_burgers_bc_output")
OUTPUT_FREQUENCY = 500


def load_burgers_data(path=DATA_FILE):
    """Load Burgers BC data. Returns bc_enc (n, 2*n_t), u (n, n_x+1, n_t+1), x, t."""
    data = np.load(path, allow_pickle=True)
    g_left = data["g_left"]
    g_right = data["g_right"]
    u = data["u"]
    x = data["x"]
    t = data["t"]
    bc_enc = np.concatenate([g_left, g_right], axis=1).astype(np.float32)
    return bc_enc, u, x, t


def get_low_mode_indices(n, k_max):
    k_max = min(k_max, n // 2)
    return list(range(0, k_max + 1)) + list(range(n - k_max, n))


def get_low_mode_indices_tensor(n, k_max, device):
    return torch.tensor(get_low_mode_indices(n, k_max), dtype=torch.long, device=device)


def n_low_modes(n, k_max):
    return len(get_low_mode_indices(n, k_max))


def _idx_to_k_actual(idx, N):
    idx = idx.to(torch.float32)
    return torch.where(idx <= (N - 1) // 2, idx, idx - N)


def _theta_from_raw(k_actual, theta_raw):
    sig = torch.sigmoid(theta_raw)
    is_negative = (k_actual < 0).to(theta_raw.dtype)
    theta = (1.0 - is_negative) * (np.pi * sig) + is_negative * (np.pi / 2.0 * (1.0 + sig))
    return theta


def _dft2d_phase_fwd_real(k_actual_x, k_actual_t, H, W, device, norm=True):
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
    n_samples = bc_enc.shape[0]
    bc_dim = bc_enc.shape[1]
    in_channels = 2
    n_t_bc = bc_dim // 2
    g_left = bc_enc[:, :n_t_bc]
    g_right = bc_enc[:, n_t_bc:]
    field = np.zeros((n_samples, in_channels, n_x_pts, n_t_pts), dtype=np.float32)
    field[:, 0, :, :] = g_left[:, None, :]
    field[:, 1, :, :] = g_right[:, None, :]
    return field


def main():
    bc_enc, u, x, t = load_burgers_data()
    n_samples = bc_enc.shape[0]
    n_x_pts = len(x)
    n_t_pts = len(t)
    n_coords = n_x_pts * n_t_pts

    in_field_np = bc_enc_to_field(bc_enc, n_x_pts, n_t_pts)
    in_channels = in_field_np.shape[1]
    u_flat = u.reshape(n_samples, -1).astype(np.float32)
    assert u_flat.shape[1] == n_coords

    if USE_TARGET_NORMALIZATION:
        u_mean = float(u_flat.mean())
        u_std = float(u_flat.std())
        if u_std < 1e-8:
            u_std = 1.0
        u_flat = (u_flat - u_mean) / u_std
        print(f"  Target normalization: mean = {u_mean:.4f}, std = {u_std:.4f}")
    else:
        u_mean, u_std = 0.0, 1.0

    in_field_t = torch.tensor(in_field_np, dtype=torch.float32, device=DEVICE)
    u_target_t = torch.tensor(u_flat, dtype=torch.float32, device=DEVICE)

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
    mse_orig_hist = []
    mse_orig_epochs = []
    for ep in range(EPOCHS):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_samples, BATCH_SIZE):
            idx = indices[start : start + BATCH_SIZE]
            inp_batch = in_field_t[idx]
            u_batch = u_target_t[idx]
            pred = net(inp_batch)
            pred_flat = pred.reshape(pred.shape[0], -1)
            loss = ((pred_flat - u_batch) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        loss_hist.append(avg_loss)
        scheduler.step(avg_loss)
        if (ep + 1) % OUTPUT_FREQUENCY == 0 or ep == 0:
            with torch.no_grad():
                pred_all = net(in_field_t)
                pred_flat_all = pred_all.reshape(pred_all.shape[0], -1)
                pred_orig = pred_flat_all.cpu().numpy() * u_std + u_mean
                u_orig = u_target_t.cpu().numpy() * u_std + u_mean
                mse_orig = ((pred_orig - u_orig) ** 2).mean() / ((u_orig ** 2).mean())
            mse_orig_hist.append(float(mse_orig))
            mse_orig_epochs.append(ep + 1)
            print(
                f"  Epoch {ep+1}/{EPOCHS}, train MSE (norm) = {avg_loss:.6e}, "
                f"REL (orig) = {mse_orig:.6e}, lr = {opt.param_groups[0]['lr']:.2e}"
            )

    print("\nTraining finished.")
    with torch.no_grad():
        pred_all = net(in_field_t)
        pred_flat_all = pred_all.reshape(pred_all.shape[0], -1)
        pred_orig = pred_flat_all.cpu().numpy() * u_std + u_mean
        u_orig_np = u_target_t.cpu().numpy() * u_std + u_mean
        test_mse_orig = ((pred_orig - u_orig_np) ** 2).mean() / ((u_orig ** 2).mean())
    print(f"  Final test REL (original scale): {test_mse_orig:.6e}")

    os.makedirs(OUT_DIR, exist_ok=True)
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
            "u_mean": u_mean,
            "u_std": u_std,
        },
        "x": np.asarray(x),
        "t": np.asarray(t),
    }
    model_path = os.path.join(OUT_DIR, "fno_burgers_bc_model.pt")
    torch.save(checkpoint, model_path)
    print(f"  Model saved to {model_path}")

    loss_csv = os.path.join(OUT_DIR, "train_fno_burgers_bc_loss_history.csv")
    with open(loss_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])
        for ep, loss_val in enumerate(loss_hist, start=1):
            writer.writerow([ep, loss_val])

    np.savetxt(
        os.path.join(OUT_DIR, "train_fno_burgers_bc_mse_orig_hist.csv"),
        np.column_stack([mse_orig_epochs, mse_orig_hist]),
        delimiter=",",
        header="epoch,mse",
        comments="",
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.semilogy(np.arange(1, len(loss_hist) + 1), loss_hist, color="C0", linewidth=0.5, alpha=0.8)
    ax1.set_ylabel("MSE (normalized)")
    ax1.set_title("FNO Burgers BC training MSE")
    ax1.grid(True, alpha=0.3)
    if mse_orig_hist:
        ax2.semilogy(mse_orig_epochs, mse_orig_hist, "o-", color="C1", markersize=4)
    ax2.set_ylabel("MSE (original scale)")
    ax2.set_xlabel("Epoch")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "train_fno_burgers_bc_mse.png"), dpi=150)
    plt.close()

    n_examples = min(3, n_samples)
    example_indices = np.linspace(0, n_samples - 1, n_examples, dtype=int)
    pred_all_np = pred_flat_all.cpu().numpy() * u_std + u_mean
    u_orig_3d = u_orig_np.reshape(n_samples, n_x_pts, n_t_pts)
    fig2, axes = plt.subplots(n_examples, 3, figsize=(12, 3 * n_examples))
    if n_examples == 1:
        axes = axes.reshape(1, -1)
    for i, ex in enumerate(example_indices):
        pred_2d = pred_all_np[ex].reshape(n_x_pts, n_t_pts)
        target_2d = u_orig_3d[ex]
        diff_2d = pred_2d - target_2d
        mid_x = n_x_pts // 2
        axes[i, 0].plot(t, pred_2d[mid_x, :], "b-", label="FNO pred")
        axes[i, 0].plot(t, target_2d[mid_x, :], "r--", label="Target u")
        axes[i, 0].set_xlabel("t")
        axes[i, 0].set_ylabel("u(x_mid, t)")
        axes[i, 0].set_title(f"Sample {ex}: pred vs target")
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 1].plot(t, diff_2d[mid_x, :], "g-")
        axes[i, 1].axhline(0, color="k", linestyle=":", alpha=0.5)
        axes[i, 1].set_xlabel("t")
        axes[i, 1].set_ylabel("pred - target")
        axes[i, 1].grid(True, alpha=0.3)
        im = axes[i, 2].imshow(diff_2d, aspect="auto", origin="lower",
                               extent=[t[0], t[-1], x[0], x[-1]], cmap="RdBu_r")
        axes[i, 2].set_xlabel("t")
        axes[i, 2].set_ylabel("x")
        plt.colorbar(im, ax=axes[i, 2], label="pred - target")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "train_fno_burgers_bc_comparison.png"), dpi=150)
    plt.close()
    print(f"  Plots saved to {OUT_DIR}")
    return net, loss_hist


if __name__ == "__main__":
    print("FNO Burgers: BC -> solution u(x,t)")
    print(f"  Data: {DATA_FILE}, Device: {DEVICE}")
    print(f"  hidden: {FNO_HIDDEN}, n_modes: {FNO_N_MODES}, n_layers: {FNO_N_LAYERS}")
    net, loss_hist = main()
