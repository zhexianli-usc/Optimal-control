"""
Short test for train_fno_heat_bc: run a few epochs to verify the FNO pipeline works.
Run with Anaconda Python: python test_fno_heat_bc.py
"""
import sys
sys.path.insert(0, ".")

from train_fno_heat_bc import (
    load_or_generate_data,
    bc_enc_to_field,
    FNO2d,
    FNO_N_MODES,
    FNO_N_LAYERS,
    FNO_HIDDEN,
    USE_CHANNEL_MLP,
    USE_TARGET_NORMALIZATION,
    DEVICE,
    BATCH_SIZE,
    LR,
)

import torch
import numpy as np

def main():
    print("FNO heat BC test (Anaconda Python)")
    print("Loading data...")
    bc_enc, m_opts, x, t = load_or_generate_data(n_samples=40, bc_type="dirichlet")
    n_samples = bc_enc.shape[0]
    n_x_pts, n_t_pts = len(x), len(t)
    in_field_np = bc_enc_to_field(bc_enc, n_x_pts, n_t_pts)
    in_channels = in_field_np.shape[1]
    m_flat = m_opts.reshape(n_samples, -1).astype(np.float32)

    if USE_TARGET_NORMALIZATION:
        m_mean = float(m_flat.mean())
        m_std = float(m_flat.std())
        if m_std < 1e-8:
            m_std = 1.0
        m_flat = (m_flat - m_mean) / m_std
        print(f"  Target norm: mean={m_mean:.4f}, std={m_std:.4f}")

    in_field_t = torch.tensor(in_field_np, dtype=torch.float32, device=DEVICE)
    m_target_t = torch.tensor(m_flat, dtype=torch.float32, device=DEVICE)
    k_max = min(FNO_N_MODES, n_x_pts // 2, n_t_pts // 2) or 1
    net = FNO2d(in_channels, 1, FNO_HIDDEN, n_x_pts, n_t_pts, k_max, FNO_N_LAYERS, use_channel_mlp=USE_CHANNEL_MLP).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    n_test_epochs = 50
    print(f"Training {n_test_epochs} epochs...")
    for ep in range(n_test_epochs):
        idx = np.random.permutation(n_samples)
        for start in range(0, n_samples, BATCH_SIZE):
            i = idx[start : start + BATCH_SIZE]
            pred = net(in_field_t[i]).reshape(len(i), -1)
            loss = ((pred - m_target_t[i]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (ep + 1) % 50 == 0 or ep == 0:
            with torch.no_grad():
                mse = ((net(in_field_t).reshape(n_samples, -1) - m_target_t) ** 2).mean().item()
            print(f"  Epoch {ep+1:4d}, MSE = {mse:.6e}")

    print("Test passed: FNO training runs correctly.")


if __name__ == "__main__":
    main()
