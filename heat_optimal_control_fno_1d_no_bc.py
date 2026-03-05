"""
Optimal control of 1D heat equation via FNO (no boundary condition as input).
Setup follows Wang et al. "Fast PDE-constrained optimization via self-supervised operator learning"
https://arxiv.org/pdf/2110.13297 but with FNO replacing DeepONet.
Domain padding per Duruisseaux et al. "Fourier Neural Operators Explained: A Practical Perspective"
https://arxiv.org/pdf/2512.01421 (Sec 3.6): pad input so FNO sees extended domain; loss on original domain.

PDE: s_t - nu*s_xx = u(t)  in (0,1) x (0,T)
     s=0 at t=0,  s=0 on x=0 and x=1  (fixed BC/IC).
- Step 1: Train FNO to learn solution operator G: u -> s (input u(t) only).
- Step 2: Parametrize control u(t) by an MLP, minimize J using trained FNO.
Cost: J(u) = (1/2) int (s-d)^2 dx dt + (alpha/2) int u^2 dt,  d = 4*x*(1-x)*sin(pi*t).
"""
import torch
import torch.nn as nn
import numpy as np

T, nu = 2.0, 0.01
m = 50
n_x, n_t = 32, 32
# Domain padding (paper 2512.01421 Sec 3.6): pad by fraction of domain; loss computed on original domain only
pad_ratio = 0.1
pad_x = max(1, int(round(n_x * pad_ratio)))
pad_t = max(1, int(round(n_t * pad_ratio)))
n_x_pad = n_x + 2 * pad_x
n_t_pad = n_t + 2 * pad_t
batch = 32
epochs1, epochs2 = 30000, 10000
width = 32
k_max = 12
n_fno_layers = 4
alpha = 0.0
lambda_pde, lambda_bc, lambda_ic = 1.0, 5.0, 5.0
device = "cuda" if torch.cuda.is_available() else "cpu"

x_grid = torch.linspace(0.0, 1.0, n_x, device=device)
t_grid = torch.linspace(0.0, T, n_t, device=device)
dx = 1.0 / (n_x - 1)
dt = T / (n_t - 1)


# ---------- FNO building blocks ----------
def get_low_mode_indices(n, k_max):
    k_max = min(k_max, n // 2)
    return list(range(0, k_max + 1)) + list(range(n - k_max, n))


def get_low_mode_indices_tensor(n, k_max, device):
    return torch.tensor(get_low_mode_indices(n, k_max), dtype=torch.long, device=device)


def n_low_modes(n, k_max):
    return len(get_low_mode_indices(n, k_max))


class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes_x, modes_t, k_max_x, k_max_t):
        super().__init__()
        self.k_max_x = k_max_x
        self.k_max_t = k_max_t
        scale = 1.0 / (in_ch * out_ch)
        self.weights_real = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes_x, modes_t))
        self.weights_imag = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes_x, modes_t))

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.fft2(x, dim=(-2, -1))
        idx_x = get_low_mode_indices_tensor(H, self.k_max_x, x.device)
        idx_t = get_low_mode_indices_tensor(W, self.k_max_t, x.device)
        modes_x, modes_t = len(idx_x), len(idx_t)
        x_kept = x_ft[:, :, idx_x, :][:, :, :, idx_t]
        R = self.weights_real + 1j * self.weights_imag
        out_kept = torch.einsum("bcij,coij->boij", x_kept, R)
        out_ft = torch.zeros(B, C, H, W, dtype=torch.complex64, device=x.device)
        ix_exp = idx_x.reshape(1, 1, -1, 1).expand(B, C, modes_x, modes_t)
        iw_exp = idx_t.reshape(1, 1, 1, -1).expand(B, C, modes_x, modes_t)
        b_exp = torch.arange(B, device=x.device).view(B, 1, 1, 1).expand(B, C, modes_x, modes_t)
        c_exp = torch.arange(C, device=x.device).view(1, C, 1, 1).expand(B, C, modes_x, modes_t)
        out_ft.index_put_((b_exp, c_exp, ix_exp, iw_exp), out_kept)
        return torch.fft.ifft2(out_ft, dim=(-2, -1)).real


class FNO2d(nn.Module):
    def __init__(self, in_ch, out_ch, width, n_x, n_t, k_max, n_layers):
        super().__init__()
        self.lift = nn.Conv2d(in_ch, width, 1)
        modes_x = n_low_modes(n_x, k_max)
        modes_t = n_low_modes(n_t, k_max)
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                "spectral": SpectralConv2d(width, width, modes_x, modes_t, k_max, k_max),
                "skip": nn.Conv2d(width, width, 1),
            }))
        self.project = nn.Conv2d(width, out_ch, 1)

    def forward(self, x):
        x = self.lift(x)
        for block in self.blocks:
            x = torch.relu(block["skip"](x) + block["spectral"](x))
        return self.project(x)


# ---------- Domain padding (paper 2512.01421 Sec 3.6) ----------
def pad_2d(x, pad_x, pad_t, mode="zero"):
    """Pad 2D grid (B, C, H, W) by (pad_x, pad_t) on each side. mode: 'zero' or 'replicate' (mirror-like)."""
    if pad_x <= 0 and pad_t <= 0:
        return x
    pad_t_left, pad_t_right = pad_t, pad_t
    pad_x_top, pad_x_bottom = pad_x, pad_x
    if mode == "replicate":
        return torch.nn.functional.pad(x, (pad_t_left, pad_t_right, pad_x_top, pad_x_bottom), mode="replicate")
    return torch.nn.functional.pad(x, (pad_t_left, pad_t_right, pad_x_top, pad_x_bottom), mode="constant", value=0.0)


def unpad_2d(x, pad_x, pad_t):
    """Restrict to original domain: x (B, C, H, W) -> x[:, :, pad_x:pad_x+n_x, pad_t:pad_t+n_t]."""
    if pad_x <= 0 and pad_t <= 0:
        return x
    return x[:, :, pad_x : pad_x + n_x, pad_t : pad_t + n_t]


# ---------- Input: u(t) only (no boundary condition) ----------
def make_grid_inputs_u_only(u_np, n_x, n_t, T_val, device):
    """Single channel: u(t) interpolated to grid and broadcast in x. (B, 1, n_x, n_t)."""
    B = u_np.shape[0]
    ts = np.linspace(0, T_val, m)
    t_axis = np.linspace(0, T_val, n_t)
    u_on_t = np.array([np.interp(t_axis, ts, u_np[i]) for i in range(B)])
    u_grid = np.broadcast_to(u_on_t[:, None, :], (B, n_x, n_t)).astype(np.float32)
    return torch.tensor(u_grid[:, None, :, :], device=device)


def sample_u(n, l=0.3):
    ts = np.linspace(0, T, m)
    K = np.exp(-0.5 * ((ts[:, None] - ts[None, :]) / l) ** 2)
    L = np.linalg.cholesky(K + 1e-5 * np.eye(m))
    u = (L @ np.random.randn(m, n)).T
    u = u / (np.std(u, axis=1, keepdims=True) + 1e-6)
    return 0.5 * u


# ---------- Spectral differentiation ----------
def compute_residual(s, u_grid, dx, dt, nu):
    s_ft = torch.fft.fft2(s, dim=(-2, -1))
    k_x = torch.fft.fftfreq(n_x, d=dx, device=s.device).reshape(1, 1, n_x, 1)
    k_t = torch.fft.fftfreq(n_t, d=dt, device=s.device).reshape(1, 1, 1, n_t)
    s_t = torch.fft.ifft2(2 * np.pi * 1j * k_t * s_ft, dim=(-2, -1)).real
    s_xx = torch.fft.ifft2(-(2 * np.pi * k_x) ** 2 * s_ft, dim=(-2, -1)).real
    return s_t - nu * s_xx - u_grid


# ---------- Step 1: Train FNO G: u -> s ----------
def train_pde_fno():
    pde_net = FNO2d(1, 1, width, n_x_pad, n_t_pad, k_max, n_fno_layers).to(device)
    opt = torch.optim.Adam(pde_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.995)

    for ep in range(epochs1):
        u_np = sample_u(batch)
        inp = make_grid_inputs_u_only(u_np, n_x, n_t, T, device)
        inp_padded = pad_2d(inp, pad_x, pad_t, mode="zero")
        s_full = pde_net(inp_padded)
        s = unpad_2d(s_full, pad_x, pad_t)
        u_on_grid = inp
        r = compute_residual(s, u_on_grid, dx, dt, nu)
        loss_pde = r.pow(2).mean()

        s_bc_x0 = s[:, :, 0, :]
        s_bc_x1 = s[:, :, -1, :]
        loss_bc = s_bc_x0.pow(2).mean() + s_bc_x1.pow(2).mean()

        s_ic = s[:, :, :, 0]
        loss_ic = s_ic.pow(2).mean()

        loss = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_ic * loss_ic
        if not torch.isfinite(loss):
            print(f"  PDE FNO {ep+1}, non-finite loss. Stopping.")
            break
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pde_net.parameters(), 1.0)
        opt.step()
        if (ep + 1) % 200 == 0:
            scheduler.step()
        if (ep + 1) % 500 == 0:
            print(f"  PDE FNO {ep+1}/{epochs1}, loss={loss.item():.2e}, pde={loss_pde.item():.2e}, bc={loss_bc.item():.2e}, ic={loss_ic.item():.2e}")

    return pde_net


# ---------- Step 2: Control MLP u_alpha(t), minimize J ----------
class ControlMLP(nn.Module):
    """Parametrize u(t) by MLP, as in Wang et al. (paper Sec 2.3)."""
    def __init__(self, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, t):
        return self.mlp(t).squeeze(-1)


def optimize_control(pde_net):
    ctrl = ControlMLP(hidden=64).to(device)
    opt = torch.optim.Adam(ctrl.parameters(), lr=1e-2)
    t_sens = torch.linspace(0, T, m, device=device)
    xx = torch.linspace(0.0, 1.0, n_x, device=device)
    tt = torch.linspace(0.0, T, n_t, device=device)
    xx, tt = torch.meshgrid(xx, tt, indexing="ij")
    d = 4 * xx * (1 - xx) * torch.sin(np.pi * tt)
    d = d.unsqueeze(0).unsqueeze(0)

    for ep in range(epochs2):
        u = ctrl(t_sens.unsqueeze(-1))
        u_np = u.detach().unsqueeze(0).expand(1, -1).cpu().numpy()
        inp = make_grid_inputs_u_only(u_np, n_x, n_t, T, device)
        inp_padded = pad_2d(inp, pad_x, pad_t, mode="zero")
        s_full = pde_net(inp_padded)
        s = unpad_2d(s_full, pad_x, pad_t)
        loss_tracking = ((s - d).pow(2)).mean()
        loss_control = (u.pow(2)).mean()
        loss = loss_tracking + alpha * 0.5 * loss_control
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (ep + 1) % 500 == 0:
            print(f"  Control {ep+1}/{epochs2}, J={loss.item():.2e} (tracking={loss_tracking.item():.2e}, u^2={loss_control.item():.2e})")

    return ctrl


if __name__ == "__main__":
    print("1D heat: FNO solution operator (u -> s), fixed BC/IC. Ref: Wang et al. 2110.13297")
    print("Step 1: Train FNO G: u -> s")
    pde_net = train_pde_fno()

    print("Step 2: Optimize control u(t) via MLP")
    ctrl = optimize_control(pde_net)

    t_plot = np.linspace(0, T, 100)
    with torch.no_grad():
        u_out = ctrl(torch.tensor(t_plot, dtype=torch.float32, device=device).unsqueeze(-1)).cpu().numpy()
    np.savetxt("u_optimal_fno_1d_no_bc.csv", np.column_stack([t_plot, u_out]), delimiter=",", header="t,u", comments="")
    print("Done. u(t) saved to u_optimal_fno_1d_no_bc.csv")
