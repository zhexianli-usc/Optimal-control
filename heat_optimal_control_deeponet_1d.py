"""
Optimal control of 1D heat equation via physics-informed DeepONet.
Same structure as the 2D case: PDE DeepONet (u,g -> s), Control DeepONet (g -> u).

PDE: s_t - nu*s_xx = u(t)  in (0,1) x (0,T)
     s=0 at t=0, s=g on boundary (x=0 and x=1; g from Gaussian random field).
- PDE DeepONet: (u, g) -> s(x,t). Trained with u,g sampled from GRFs.
- Control DeepONet: g -> u(.). Train with E_g[J].
"""
import torch
import torch.nn as nn
import numpy as np

T, nu = 2.0, 0.01
m, hidden = 50, 64
n_b = 20  # boundary sensors: 10 on x=0, 10 on x=1 (g(x,t) at fixed (x,t) on boundary)
batch, n_colloc = 32, 100
epochs1, epochs2 = 300000, 300000
clip_grad = 1.0
lambda_pde, lambda_bc, lambda_ic = 1.0, 5.0, 5.0
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_boundary_sensor_locations():
    """(n_b, 2): (x, t) on boundary x=0 or x=1."""
    locs = []
    n_per_edge = n_b // 2
    t_ = np.linspace(0.05, T - 0.05, n_per_edge)
    for i in range(n_per_edge):
        locs.append([0.0, t_[i]])
    for i in range(n_per_edge):
        locs.append([1.0, t_[i]])
    return np.array(locs, dtype=np.float32)


_boundary_sensor_locs = get_boundary_sensor_locations()


class DeepONet(nn.Module):
    """G: (u(t), g(boundary)) -> s(x,t). Branch: (u at m sensors, g at n_b). Trunk: (x, t)."""
    def __init__(self):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(m + n_b, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),
        )
        self.trunk = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, u, g, y):
        """u: (B,m), g: (B,n_b), y: (B,2) with y=(x,t) -> (B,1)"""
        ug = torch.cat([u, g], dim=-1)
        return (self.branch(ug) * self.trunk(y)).sum(dim=-1, keepdim=True)


def sample_u(n, l=0.3):
    ts = np.linspace(0, T, m)
    K = np.exp(-0.5 * ((ts[:, None] - ts[None, :]) / l) ** 2)
    L = np.linalg.cholesky(K + 1e-5 * np.eye(m))
    u = (L @ np.random.randn(m, n)).T
    u = u / (np.std(u, axis=1, keepdims=True) + 1e-6)
    return 0.5 * u


def sample_g(n, l=0.4):
    """Sample boundary condition g at _boundary_sensor_locs (x,t) from a GRF."""
    pts = _boundary_sensor_locs
    d = (pts[:, None, :] - pts[None, :, :]) ** 2
    K = np.exp(-0.5 * (d.sum(axis=2) / (l ** 2)))
    L = np.linalg.cholesky(K + 1e-5 * np.eye(n_b))
    g = (L @ np.random.randn(n_b, n)).T
    g = g / (np.std(g, axis=1, keepdims=True) + 1e-6)
    return 0.5 * g


def bc_target_from_g(g_np, y_bc_base):
    """y_bc_base (n_bc, 2) = (x, t). Nearest neighbor from g at sensors -> (B, n_bc)."""
    dist = np.sum((y_bc_base[:, None, :] - _boundary_sensor_locs[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(dist, axis=1)
    return g_np[:, nearest]


def train_deeponet():
    net = DeepONet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.995)

    for ep in range(epochs1):
        u_np = sample_u(batch)
        g_np = sample_g(batch)
        u = torch.tensor(u_np, dtype=torch.float32, device=device)
        g = torch.tensor(g_np, dtype=torch.float32, device=device)

        x = np.random.rand(n_colloc)
        t = np.random.rand(n_colloc) * T
        pts = np.stack([x, t], axis=1)

        u_exp = u.repeat_interleave(n_colloc, dim=0)
        g_exp = g.repeat_interleave(n_colloc, dim=0)

        y_exp = torch.tensor(pts, dtype=torch.float32, device=device).repeat(batch, 1)
        y_exp.requires_grad_(True)

        s = net(u_exp, g_exp, y_exp)
        ds = torch.autograd.grad(s.sum(), y_exp, create_graph=True)[0]
        s_t = ds[:, 1:2]
        s_x = ds[:, 0:1]
        s_xx = torch.autograd.grad(s_x.sum(), y_exp, retain_graph=True)[0][:, 0:1]

        u_interp = np.array([np.interp(t, np.linspace(0, T, m), u_np[i]) for i in range(batch)])
        u_pt = torch.tensor(u_interp, dtype=torch.float32, device=device).reshape(-1, 1)
        r = s_t - nu * s_xx - u_pt

        n_bc = n_colloc
        xb = np.random.choice([0.0, 1.0], n_bc)
        tb = np.random.rand(n_bc) * T
        y_bc_base_np = np.stack([xb, tb], 1)
        y_bc_base = torch.tensor(y_bc_base_np, dtype=torch.float32, device=device)
        y_bc = y_bc_base.repeat(batch, 1)
        u_bc = u.repeat_interleave(n_bc, dim=0)
        g_bc = g.repeat_interleave(n_bc, dim=0)
        s_bc = net(u_bc, g_bc, y_bc)
        bc_target_np = bc_target_from_g(g_np, y_bc_base_np)
        bc_target = torch.tensor(bc_target_np, dtype=torch.float32, device=device).reshape(-1, 1)

        n_ic = n_colloc
        xi = np.random.rand(n_ic)
        y_ic_base = torch.tensor(np.stack([xi, np.zeros(n_ic)], 1), dtype=torch.float32, device=device)
        y_ic = y_ic_base.repeat(batch, 1)
        u_ic = u.repeat_interleave(n_ic, dim=0)
        g_ic = g.repeat_interleave(n_ic, dim=0)
        s_ic = net(u_ic, g_ic, y_ic)

        loss_pde = r.pow(2).mean()
        loss_bc = (s_bc - bc_target).pow(2).mean()
        loss_ic = s_ic.pow(2).mean()
        loss = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_ic * loss_ic
        if not torch.isfinite(loss):
            print(f"  DeepONet {ep+1}/{epochs1}, non-finite loss. Stopping.")
            break
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
        opt.step()
        if (ep + 1) % 200 == 0:
            scheduler.step()
        if (ep + 1) % 200 == 0:
            print(
                f"  DeepONet {ep+1}/{epochs1}, "
                f"loss={loss.item():.2e}, pde={loss_pde.item():.2e}, "
                f"bc={loss_bc.item():.2e}, ic={loss_ic.item():.2e}, "
                f"lr={opt.param_groups[0]['lr']:.2e}"
            )

    return net


class ControlDeepONet(nn.Module):
    """Optimal control u(t) as function of boundary condition g. Branch: g. Trunk: t."""
    def __init__(self):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(n_b, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),
        )
        self.trunk = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, g, t):
        """g: (B, n_b), t: (m,) or (B, m) -> u: (B, m)."""
        if t.dim() == 1:
            t = t.unsqueeze(0).expand(g.shape[0], -1)
        trunk_out = self.trunk(t.unsqueeze(-1))
        branch_out = self.branch(g)
        return (branch_out.unsqueeze(1) * trunk_out).sum(dim=-1)


def optimize_control(net):
    t_sens = torch.linspace(0, T, m, device=device)
    ctrl = ControlDeepONet().to(device)
    opt = torch.optim.Adam(ctrl.parameters(), lr=1e-2)
    n_x, n_t = 24, 12
    x = torch.linspace(0.05, 0.95, n_x, device=device)
    t = torch.linspace(0.05, T - 0.05, n_t, device=device)
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    coords = torch.stack([xx.flatten(), tt.flatten()], dim=1)
    n_pts = coords.shape[0]

    for ep in range(epochs2):
        g_np = sample_g(batch)
        g_batch = torch.tensor(g_np, dtype=torch.float32, device=device)
        u = ctrl(g_batch, t_sens)
        u_exp = u.repeat_interleave(n_pts, dim=0)
        g_exp = g_batch.repeat_interleave(n_pts, dim=0)
        y_exp = coords.unsqueeze(0).expand(batch, -1, -1).reshape(-1, 2)
        s = net(u_exp, g_exp, y_exp)
        loss = s.pow(2).mean() + 0.50 * u.pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (ep + 1) % 100 == 0:
            print(f"  Control {ep+1}/{epochs2}, J={loss.item():.2e}")

    return ctrl


if __name__ == "__main__":
    print("1D heat equation: optimal control via DeepONet")
    print("Step 1: Train PDE DeepONet (u,g -> s(x,t))")
    net = train_deeponet()

    print("Step 2: Optimize control DeepONet u(g,t) with E_g[J]")
    ctrl = optimize_control(net)

    t_plot = np.linspace(0, T, 100)
    g_example = sample_g(1)
    with torch.no_grad():
        g_t = torch.tensor(g_example, dtype=torch.float32, device=device)
        t_t = torch.tensor(t_plot, dtype=torch.float32, device=device)
        u_opt = ctrl(g_t, t_t).cpu().numpy().flatten()
    np.savetxt(
        "u_optimal_1d.csv",
        np.column_stack([t_plot, u_opt]),
        delimiter=",",
        header="t,u",
        comments="",
    )
    print("Done. u(g,t) for one sample g saved to u_optimal_1d.csv")
