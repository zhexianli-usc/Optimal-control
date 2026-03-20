"""
Train a DeepONet to map boundary condition -> optimal control for the 1D heat equation.
Input: BC encoding (4,) = [left_type, left_val, right_type, right_val].
Output: m(x, t) = optimal control on the grid.
Supervised loss: MSE between DeepONet(BC)(x,t) and the precomputed optimal m from data.
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Try to load data; if not found, generate a small set
DATA_FILE = "heat_optimal_control_data.npz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN = 16
EPOCHS = 50000
BATCH_SIZE = 16
LR = 1e-3

# Target normalization: train on (m - m_mean) / m_std so loss scale is ~1 when fitting well
USE_TARGET_NORMALIZATION = True


class DeepONetBC2Control(nn.Module):
    """Branch: BC encoding (4,) -> hidden. Trunk: (x, t) -> hidden. Output: branch·trunk = m(x,t)."""

    def __init__(self, bc_dim=4, hidden=16):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(bc_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
        )
        self.trunk = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, bc, coords):
        """
        bc: (B, 4), coords: (N, 2) with (x, t) per row.
        Returns (B, N): m at each (x,t) for each batch item.
        """
        branch_out = self.branch(bc)   # (B, hidden)
        trunk_out = self.trunk(coords) # (N, hidden)
        return torch.mm(branch_out, trunk_out.t())  # (B, N)


def load_or_generate_data(n_samples=25, bc_type="dirichlet"):
    """Load from DATA_FILE or generate and save. Supports (1) old format: bc_encodings (n,4); (2) GRF format: g_left, g_right (n, n_t+1)."""
    try:
        data = np.load(DATA_FILE, allow_pickle=True)
        m_opts = data["m_opts"]
        x = data["x"]
        t = data["t"]
        if "g_left" in data and "g_right" in data:
            g_left = data["g_left"]
            g_right = data["g_right"]
            bc_enc = np.concatenate([g_left, g_right], axis=1).astype(np.float32)
            print(f"Loaded {len(bc_enc)} samples from {DATA_FILE} (GRF BC, bc_type={data.get('bc_type', '?')})")
        else:
            bc_enc = data["bc_encodings"]
            print(f"Loaded {len(bc_enc)} samples from {DATA_FILE} (constant BC encoding)")
    except FileNotFoundError:
        print(f"{DATA_FILE} not found. Generating {n_samples} samples ({bc_type} GRF BCs)...")
        from generate_heat_optimal_control_data import generate_dataset
        g_left, g_right, m_opts, x, t = generate_dataset(n_samples, bc_type=bc_type, n_x=24, n_t=30, verbose=False)
        bc_enc = np.concatenate([g_left, g_right], axis=1).astype(np.float32)
        np.savez(DATA_FILE, bc_type=bc_type, g_left=g_left, g_right=g_right, m_opts=m_opts, x=x, t=t)
        print(f"Saved to {DATA_FILE}")
    return bc_enc, m_opts, x, t


def main():
    bc_enc, m_opts, x, t = load_or_generate_data(n_samples=40, bc_type="dirichlet")

    n_samples = bc_enc.shape[0]
    bc_dim = bc_enc.shape[1]
    n_x_pts = len(x)
    n_t_pts = len(t)
    X_grid, T_grid = np.meshgrid(x, t, indexing="ij")
    coords = np.stack([X_grid.ravel(), T_grid.ravel()], axis=1).astype(np.float32)
    n_coords = coords.shape[0]

    # Targets: (n_samples, n_coords)
    m_flat = m_opts.reshape(n_samples, -1).astype(np.float32)
    assert m_flat.shape[1] == n_coords

    # Normalize targets so MSE is on a ~O(1) scale; makes optimization easier
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

    bc_t = torch.tensor(bc_enc, dtype=torch.float32, device=DEVICE)
    coords_t = torch.tensor(coords, dtype=torch.float32, device=DEVICE)
    m_target_t = torch.tensor(m_flat, dtype=torch.float32, device=DEVICE)

    net = DeepONetBC2Control(bc_dim=bc_dim, hidden=HIDDEN).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2000, min_lr=1e-6)

    indices = np.arange(n_samples)
    loss_hist = []
    mse_orig_hist = []  # original-scale MSE at logging steps
    mse_orig_epochs = []
    for ep in range(EPOCHS):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_samples, BATCH_SIZE):
            idx = indices[start : start + BATCH_SIZE]
            bc_batch = bc_t[idx]
            m_batch = m_target_t[idx]
            pred = net(bc_batch, coords_t)
            loss = ((pred - m_batch) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        loss_hist.append(avg_loss)
        scheduler.step(avg_loss)
        if (ep + 1) % 2000 == 0 or ep == 0:
            # Report MSE in original scale for interpretability
            with torch.no_grad():
                pred_all = net(bc_t, coords_t)
                pred_orig = pred_all.cpu().numpy() * m_std + m_mean
                m_orig = m_target_t.cpu().numpy() * m_std + m_mean
                mse_orig = ((pred_orig - m_orig) ** 2).mean()
            mse_orig_hist.append(float(mse_orig))
            mse_orig_epochs.append(ep + 1)
            print(f"  Epoch {ep+1}/{EPOCHS}, train MSE (norm) = {avg_loss:.6e}, MSE (orig) = {mse_orig:.6e}, lr = {opt.param_groups[0]['lr']:.2e}")

    print("\nTraining finished.")
    if len(loss_hist) >= 2 and loss_hist[-1] < loss_hist[0]:
        print("  Loss decreased: training successful.")
    else:
        print("  (Loss may need more epochs or data.)")

    # Final test: MSE in original scale
    with torch.no_grad():
        pred_all = net(bc_t, coords_t)
        pred_orig = pred_all.cpu().numpy() * m_std + m_mean
        m_orig_np = m_target_t.cpu().numpy() * m_std + m_mean
        test_mse_norm = ((pred_all - m_target_t) ** 2).mean().item()
        test_mse_orig = ((pred_orig - m_orig_np) ** 2).mean()
    print(f"  Final test MSE (normalized): {test_mse_norm:.6e}")
    print(f"  Final test MSE (original scale): {test_mse_orig:.6e}")

    # Plot MSE during training
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    epochs_arr = np.arange(1, len(loss_hist) + 1)
    ax1.semilogy(epochs_arr, loss_hist, color="C0", linewidth=0.5, alpha=0.8)
    ax1.set_ylabel("MSE (normalized)")
    ax1.set_title("Training MSE (normalized targets)")
    ax1.grid(True, alpha=0.3)

    if mse_orig_hist:
        ax2.semilogy(mse_orig_epochs, mse_orig_hist, "o-", color="C1", markersize=4)
    ax2.set_ylabel("MSE (original scale)")
    ax2.set_xlabel("Epoch")
    ax2.set_title("Test MSE (original scale, every 2000 epochs)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = "train_deeponet_mse.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Training MSE plot saved to {plot_path}")

    return net, loss_hist


if __name__ == "__main__":
    print("DeepONet: BC -> optimal control (supervised MSE)")
    print(f"  Device: {DEVICE}, hidden: {HIDDEN}, epochs: {EPOCHS}")
    net, loss_hist = main()
