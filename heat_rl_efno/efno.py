"""
Extended Fourier Neural Operator (2D, complex-frequency inverse) — same construction
as burgers_bc_data/train_fno_burgers_bc.py for reuse in RL policy/value networks.
"""
import numpy as np
import torch
import torch.nn as nn


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


class EFNO2d(nn.Module):
    """Extended FNO: 2D (x, t) field in, field out."""

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
