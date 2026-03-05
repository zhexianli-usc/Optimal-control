"""
Optimal control of 1D heat equation via Fourier Neural Operator (FNO).
Ref: Li et al. "Fourier Neural Operator for Parametric Partial Differential Equations"
     https://arxiv.org/pdf/2010.08895 (ICLR 2021)

PDE: s_t - nu*s_xx = u(t)  in (0,1) x (0,T)
     s=0 at t=0, s=g on boundary (g from GRF).
- PDE FNO: (u, g) -> s(x,t) on a fixed (n_x, n_t) grid.
- Control FNO: g -> u(t). Train with E_g[J].
"""
import torch
import torch.nn as nn
import numpy as np

T, nu = 2.0, 0.01
m = 50
n_b = 20
n_x, n_t = 20, 20
batch = 32
epochs1, epochs2 = 30000, 30000  # use 30000, 30000 for full training
width = 32
k_max = 12
n_fno_layers = 4
lambda_pde, lambda_bc, lambda_ic = 1.0, 5.0, 5.0
device = "cuda" if torch.cuda.is_available() else "cpu"

# Grid for (x, t)
x_grid = torch.linspace(0.0, 1.0, n_x, device=device)
t_grid = torch.linspace(0.0, T, n_t, device=device)
dx = 1.0 / (n_x - 1)
dt = T / (n_t - 1)


def get_boundary_sensor_locations():
    locs = []
    n_per_edge = n_b // 2
    t_ = np.linspace(0.05, T - 0.05, n_per_edge)
    for ti in t_:
        locs.append([0.0, ti])
    for ti in t_:
        locs.append([1.0, ti])
    return np.array(locs, dtype=np.float32)


_boundary_sensor_locs = get_boundary_sensor_locations()


# ---------- Fourier layer (paper Sec 4: K(v) = F^{-1}(R · F(v))) ----------
def get_low_mode_indices(n, k_max):
    """Indices for low Fourier modes: 0..k_max and n-k_max..n-1."""
    k_max = min(k_max, n // 2)
    return list(range(0, k_max + 1)) + list(range(n - k_max, n))


def get_low_mode_indices_tensor(n, k_max, device):
    """Same as get_low_mode_indices but returns a 1D LongTensor."""
    idx = get_low_mode_indices(n, k_max)
    return torch.tensor(idx, dtype=torch.long, device=device)


def n_low_modes(n, k_max):
    return len(get_low_mode_indices(n, k_max))


class SpectralConv2d(nn.Module):
    """Fourier integral operator for 2D: FFT2 -> R·(truncated) -> IFFT2. Paper Eq (4)."""

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


class SpectralConv1d(nn.Module):
    """Fourier integral operator for 1D."""

    def __init__(self, in_ch, out_ch, modes, k_max):
        super().__init__()
        self.modes = modes
        self.k_max = k_max
        scale = 1.0 / (in_ch * out_ch)
        self.weights_real = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes))
        self.weights_imag = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes))

    def forward(self, x):
        B, in_ch, L = x.shape
        x_ft = torch.fft.fft(x, dim=-1)
        idx = get_low_mode_indices_tensor(L, self.k_max, x.device)
        modes = len(idx)
        x_kept = x_ft[:, :, idx]
        R = self.weights_real + 1j * self.weights_imag
        out_kept = torch.einsum("bci,coi->boi", x_kept, R)
        out_ch = out_kept.shape[1]
        out_ft = torch.zeros(B, out_ch, L, dtype=torch.complex64, device=x.device)
        k_exp = idx.reshape(1, 1, -1).expand(B, out_ch, modes)
        b_exp = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, out_ch, modes)
        c_exp = torch.arange(out_ch, device=x.device).view(1, -1, 1).expand(B, out_ch, modes)
        out_ft.index_put_((b_exp, c_exp, k_exp), out_kept)
        return torch.fft.ifft(out_ft, dim=-1).real


# ---------- PDE FNO: input (u, g) on 2D grid -> s(x,t) ----------
def make_grid_inputs(u_np, g_np, n_x, n_t, T_val, device):
    """Build 2-channel input: channel0 = u(t) broadcast in x; channel1 = g(x,t) from boundary interp."""
    B = u_np.shape[0]
    ts = np.linspace(0, T_val, m)
    t_axis = np.linspace(0, T_val, n_t)
    u_on_t = np.array([np.interp(t_axis, ts, u_np[i]) for i in range(B)])
    u_grid = np.broadcast_to(u_on_t[:, None, :], (B, n_x, n_t))
    t0 = _boundary_sensor_locs[: n_b // 2, 1]
    t1 = _boundary_sensor_locs[n_b // 2 :, 1]
    g0 = np.array([np.interp(t_axis, t0, g_np[i, : n_b // 2]) for i in range(B)])
    g1 = np.array([np.interp(t_axis, t1, g_np[i, n_b // 2 :]) for i in range(B)])
    x_axis = np.linspace(0, 1, n_x)
    g_grid = g0[:, None, :] * (1 - x_axis)[None, :, None] + g1[:, None, :] * x_axis[None, :, None]
    inp = np.stack([u_grid, g_grid], axis=1).astype(np.float32)
    return torch.tensor(inp, device=device)


class FNO2d(nn.Module):
    """2D FNO: lift -> [Fourier layer + skip] x L -> project. Paper Fig 2."""

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


# ---------- Control FNO: g (1D) -> u(t) ----------
def g_to_grid(g_np, m_out, device):
    """Interpolate g from n_b sensors to m_out time points (for 1D FNO input)."""
    B = g_np.shape[0]
    t_sens = _boundary_sensor_locs[:, 1]
    t_out = np.linspace(0, T, m_out)
    g_interp = np.array([np.interp(t_out, t_sens, g_np[i]) for i in range(B)])
    return torch.tensor(g_interp[:, None, :], dtype=torch.float32, device=device)


class FNO1d(nn.Module):
    """1D FNO: lift -> [Fourier layer + skip] x L -> project. For control g -> u."""

    def __init__(self, in_ch, out_ch, width, length, k_max, n_layers):
        super().__init__()
        self.lift = nn.Conv1d(in_ch, width, 1)
        modes = n_low_modes(length, k_max)
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                "spectral": SpectralConv1d(width, width, modes, k_max),
                "skip": nn.Conv1d(width, width, 1),
            }))
        self.project = nn.Conv1d(width, out_ch, 1)
        self.length = length

    def forward(self, x):
        x = self.lift(x)
        for block in self.blocks:
            x = torch.relu(block["skip"](x) + block["spectral"](x))
        return self.project(x)


# ---------- Sampling ----------
def sample_u(n, l=0.3):
    ts = np.linspace(0, T, m)
    K = np.exp(-0.5 * ((ts[:, None] - ts[None, :]) / l) ** 2)
    L = np.linalg.cholesky(K + 1e-5 * np.eye(m))
    u = (L @ np.random.randn(m, n)).T
    u = u / (np.std(u, axis=1, keepdims=True) + 1e-6)
    return 0.5 * u


def sample_g(n, l=0.4):
    pts = _boundary_sensor_locs
    d = (pts[:, None, :] - pts[None, :, :]) ** 2
    K = np.exp(-0.5 * (d.sum(axis=2) / (l ** 2)))
    L = np.linalg.cholesky(K + 1e-5 * np.eye(n_b))
    g = (L @ np.random.randn(n_b, n)).T
    g = g / (np.std(g, axis=1, keepdims=True) + 1e-6)
    return 0.5 * g


# ---------- Derivatives on grid (spectral / FFT) ----------
def compute_residual(s, u_grid, dx, dt, nu):
    """s, u_grid: (B, 1, n_x, n_t). Residual s_t - nu*s_xx - u via FFT (spectral differentiation)."""
    s_ft = torch.fft.fft2(s, dim=(-2, -1))
    k_x = torch.fft.fftfreq(n_x, d=dx, device=s.device).reshape(1, 1, n_x, 1)
    k_t = torch.fft.fftfreq(n_t, d=dt, device=s.device).reshape(1, 1, 1, n_t)
    s_t = torch.fft.ifft2(2 * np.pi * 1j * k_t * s_ft, dim=(-2, -1)).real
    s_xx = torch.fft.ifft2(-(2 * np.pi * k_x) ** 2 * s_ft, dim=(-2, -1)).real
    return s_t - nu * s_xx - u_grid


def train_pde_fno():
    pde_net = FNO2d(2, 1, width, n_x, n_t, k_max, n_fno_layers).to(device)
    opt = torch.optim.Adam(pde_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.995)

    for ep in range(epochs1):
        u_np = sample_u(batch)
        g_np = sample_g(batch)
        inp = make_grid_inputs(u_np, g_np, n_x, n_t, T, device)
        s = pde_net(inp)

        u_on_grid = make_grid_inputs(u_np, g_np, n_x, n_t, T, device)[:, 0:1]
        r = compute_residual(s, u_on_grid, dx, dt, nu)
        loss_pde = r.pow(2).mean()

        s_bc_x0 = s[:, :, 0, :]
        s_bc_x1 = s[:, :, -1, :]
        g0 = np.array([np.interp(t_grid.cpu().numpy(), _boundary_sensor_locs[: n_b // 2, 1], g_np[i, : n_b // 2]) for i in range(batch)])
        g1 = np.array([np.interp(t_grid.cpu().numpy(), _boundary_sensor_locs[n_b // 2 :, 1], g_np[i, n_b // 2 :]) for i in range(batch)])
        g0_t = torch.tensor(g0, dtype=torch.float32, device=device).unsqueeze(1)
        g1_t = torch.tensor(g1, dtype=torch.float32, device=device).unsqueeze(1)
        loss_bc = (s_bc_x0 - g0_t).pow(2).mean() + (s_bc_x1 - g1_t).pow(2).mean()

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
        if (ep + 1) % 2 == 0:
            print(f"  PDE FNO {ep+1}/{epochs1}, loss={loss.item():.2e}, pde={loss_pde.item():.2e}, bc={loss_bc.item():.2e}, ic={loss_ic.item():.2e}")

    return pde_net


def optimize_control(pde_net):
    ctrl_net = FNO1d(1, 1, width, m, k_max, n_fno_layers).to(device)
    opt = torch.optim.Adam(ctrl_net.parameters(), lr=1e-2)
    # Cost grid (interior points)
    xi = torch.linspace(0.05, 0.95, n_x, device=device)
    ti = torch.linspace(0.05, T - 0.05, n_t, device=device)
    xx, tt = torch.meshgrid(xi, ti, indexing="ij")
    n_pts = xx.numel()

    for ep in range(epochs2):
        g_np = sample_g(batch)
        g_in = g_to_grid(g_np, m, device)
        u = ctrl_net(g_in).squeeze(1)
        u_np = u.detach().cpu().numpy()
        inp = make_grid_inputs(u_np, g_np, n_x, n_t, T, device)
        s = pde_net(inp)
        loss = s.pow(2).mean() + u.pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (ep + 1) % 100 == 0:
            print(f"  Control FNO {ep+1}/{epochs2}, J={loss.item():.2e}")

    return ctrl_net


if __name__ == "__main__":
    print("1D heat equation: optimal control via Fourier Neural Operator")
    print("Step 1: Train PDE FNO (u,g -> s(x,t))")
    pde_net = train_pde_fno()

    print("Step 2: Optimize control FNO g -> u(t) with E_g[J]")
    ctrl_net = optimize_control(pde_net)

    t_plot = np.linspace(0, T, 100)
    g_ex = sample_g(1)
    g_in = g_to_grid(g_ex, m, device)
    with torch.no_grad():
        u_out = ctrl_net(g_in).squeeze(1).squeeze(0).cpu().numpy()
    u_interp = np.interp(t_plot, np.linspace(0, T, m), u_out)
    np.savetxt("u_optimal_fno_1d.csv", np.column_stack([t_plot, u_interp]), delimiter=",", header="t,u", comments="")
    print("Done. u(g,t) for one sample g saved to u_optimal_fno_1d.csv")
