"""
Actor–critic RL for additive heat control using extended FNO (complex-frequency EFNO).

Uses a continuous-action DDPG-style objective (policy gradient through Q),
which is the standard actor–critic extension of Q-learning to high-dimensional
actions. Policy and Q are neural operators on (x, t).

PDE: u_t = alpha * u_xx + a(x,t), Dirichlet boundaries, reward penalizes control
energy and soft violations of |u| <= u_abs_max at every interior x and time.
"""
from __future__ import annotations

import argparse
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim

from .heat_env import HeatEnv, HeatEnvConfig, heat_env_defaults
from .models import ActorEFNO, CriticQEFNO, gather_action_time, state_tensor
from .replay_buffer import ReplayBuffer


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def random_initial_u0(B: int, nx: int, L: float, device: torch.device, n_modes: int = 6) -> torch.Tensor:
    """Smooth Dirichlet-compatible ICs: sum_k A_k sin(k pi x / L)."""
    x = torch.linspace(0.0, L, nx, device=device, dtype=torch.float32)
    u0 = torch.zeros(B, nx, device=device, dtype=torch.float32)
    for k in range(1, n_modes + 1):
        amp = (torch.randn(B, 1, device=device) * (0.4 / k)).clamp(-0.6 / k, 0.6 / k)
        u0 = u0 + amp * torch.sin(k * torch.pi * x / L)
    u0[:, 0] = 0.0
    u0[:, -1] = 0.0
    return u0


def collect_transitions(
    env: HeatEnv,
    actor: ActorEFNO,
    u0: torch.Tensor,
    x_norm: torch.Tensor,
    t_norm: torch.Tensor,
    noise_scale: float,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """One parallel rollout per batch row; returns list of (s, a, r, s2, done, n)."""
    B, nx = u0.shape
    nt = env.cfg.nt
    device = u0.device
    u = torch.zeros(B, nx, nt, device=device, dtype=u0.dtype)
    u[:, :, 0] = u0
    u[:, 0, :] = 0.0
    u[:, -1, :] = 0.0
    out = []
    for n in range(nt - 1):
        s = state_tensor(u, n, x_norm, t_norm, nt)
        with torch.no_grad():
            a_full = actor(s)
            a = gather_action_time(a_full, torch.full((B,), n, device=device, dtype=torch.long))
            if noise_scale > 0:
                a = a.clone()
                a[:, 1:-1] = a[:, 1:-1] + noise_scale * torch.randn(B, nx - 2, device=device, dtype=a.dtype)
        a[:, 0] = 0.0
        a[:, -1] = 0.0
        u_next = env.step(u[:, :, n], a)
        u[:, :, n + 1] = u_next
        r = env.step_reward(a, u_next)
        done = n == nt - 2
        if done:
            s2 = torch.zeros_like(s)
        else:
            s2 = state_tensor(u, n + 1, x_norm, t_norm, nt)
        n_t = torch.full((B,), n, device=device, dtype=torch.long)
        d = torch.full((B,), 1.0 if done else 0.0, device=device, dtype=torch.float32)
        out.append((s, a, r, s2, d, n_t))
    return out


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)
    if device.type == "cuda":
        di = 0 if device.index is None else int(device.index)
        print(f"Device: {device} ({torch.cuda.get_device_name(di)})")
    else:
        print(f"Device: {device}")

    cfg = HeatEnvConfig.from_rl_args(args)
    env = HeatEnv(cfg, device)

    nx, nt = cfg.nx, cfg.nt
    x_norm = (env.x / cfg.L).view(1, nx, 1)
    t_norm = (env.t / cfg.T).view(1, 1, nt)

    actor = ActorEFNO(nx, nt, k_max=args.k_max, width=args.width, n_layers=args.n_layers, a_max=args.a_max).to(device)
    actor_tgt = ActorEFNO(nx, nt, k_max=args.k_max, width=args.width, n_layers=args.n_layers, a_max=args.a_max).to(device)
    actor_tgt.load_state_dict(actor.state_dict())

    critic = CriticQEFNO(nx, nt, k_max=args.k_max, width=args.width, n_layers=args.n_layers).to(device)
    critic_tgt = CriticQEFNO(nx, nt, k_max=args.k_max, width=args.width, n_layers=args.n_layers).to(device)
    critic_tgt.load_state_dict(critic.state_dict())

    opt_a = optim.Adam(actor.parameters(), lr=args.lr_actor)
    opt_c = optim.Adam(critic.parameters(), lr=args.lr_critic)

    buf = ReplayBuffer(args.buffer_size, device)

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("episode,q_loss,actor_loss,mean_return\n")

    for ep in range(1, args.episodes + 1):
        noise = max(args.noise_final, args.noise_init * (args.noise_decay**ep))
        u0 = random_initial_u0(args.batch_envs, nx, cfg.L, device)
        trajs = collect_transitions(env, actor, u0, x_norm, t_norm, noise_scale=noise)
        ep_ret = torch.zeros(args.batch_envs, device=device)
        for (s, a, r, s2, d, n_step) in trajs:
            ep_ret = ep_ret + r
            for b in range(args.batch_envs):
                buf.push(
                    s[b],
                    a[b],
                    float(r[b].item()),
                    s2[b],
                    bool(d[b].item() > 0.5),
                    int(n_step[b].item()),
                )

        mean_ret = float(ep_ret.mean().item())
        q_loss_acc = 0.0
        a_loss_acc = 0.0
        n_updates = 0

        if len(buf) >= args.batch_train:
            for _ in range(args.updates_per_episode):
                s_b, a_b, r_b, s2_b, d_b, n_b = buf.sample_with_n(args.batch_train)
                n_next = (n_b + 1).clamp(max=nt - 1)

                with torch.no_grad():
                    a2_full = actor_tgt(s2_b)
                    a2 = gather_action_time(a2_full, n_next)
                    a2[:, 0] = 0.0
                    a2[:, -1] = 0.0
                    q2 = critic_tgt(s2_b, a2).squeeze(-1)
                    y = r_b + (1.0 - d_b) * args.gamma * q2

                q_pred = critic(s_b, a_b).squeeze(-1)
                loss_q = nn.functional.mse_loss(q_pred, y)
                opt_c.zero_grad()
                loss_q.backward()
                opt_c.step()

                a_full_pi = actor(s_b)
                a_pi = gather_action_time(a_full_pi, n_b)
                a_pi[:, 0] = 0.0
                a_pi[:, -1] = 0.0
                q_pi = critic(s_b, a_pi).squeeze(-1)
                loss_a = -q_pi.mean()
                opt_a.zero_grad()
                loss_a.backward()
                opt_a.step()

                soft_update(actor_tgt, actor, args.tau)
                soft_update(critic_tgt, critic, args.tau)

                q_loss_acc += float(loss_q.item())
                a_loss_acc += float(loss_a.item())
                n_updates += 1

        if n_updates > 0:
            q_loss_acc /= n_updates
            a_loss_acc /= n_updates
        with open(log_path, "a") as f:
            f.write(f"{ep},{q_loss_acc:.6f},{a_loss_acc:.6f},{mean_ret:.6f}\n")

        if ep % args.log_interval == 0 or ep == 1:
            print(
                f"episode {ep}/{args.episodes}  mean_return={mean_ret:.4f}  "
                f"q_loss={q_loss_acc:.4f}  actor_loss={a_loss_acc:.4f}  noise={noise:.4f}  buf={len(buf)}"
            )

        if ep % args.save_interval == 0:
            ckpt = {
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "cfg": cfg,
                "args": vars(args),
            }
            torch.save(ckpt, os.path.join(args.out_dir, f"checkpoint_ep{ep}.pt"))

    final_ckpt = {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "cfg": cfg,
        "args": vars(args),
    }
    torch.save(final_ckpt, os.path.join(args.out_dir, "checkpoint_final.pt"))


def main():
    env_d = heat_env_defaults()
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--batch-envs", type=int, default=16, help="Parallel rollouts per episode")
    p.add_argument("--batch-train", type=int, default=64)
    p.add_argument("--buffer-size", type=int, default=50_000)
    p.add_argument("--updates-per-episode", type=int, default=80)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--lr-actor", type=float, default=1e-4)
    p.add_argument("--lr-critic", type=float, default=3e-4)
    p.add_argument("--noise-init", type=float, default=0.15)
    p.add_argument("--noise-final", type=float, default=0.02)
    p.add_argument("--noise-decay", type=float, default=0.995)
    p.add_argument("--nx", type=int, default=env_d.nx, help="Spatial grid points (incl. boundaries); default from HeatEnvConfig")
    p.add_argument("--nt", type=int, default=env_d.nt, help="Time slices incl. t=0; default from HeatEnvConfig")
    p.add_argument("--L", type=float, default=env_d.L)
    p.add_argument("--T", type=float, default=env_d.T)
    p.add_argument("--alpha", type=float, default=env_d.alpha)
    p.add_argument("--u-max", type=float, default=env_d.u_abs_max, help="Soft state threshold |u| (HeatEnvConfig.u_abs_max)")
    p.add_argument("--w-control", type=float, default=env_d.w_control)
    p.add_argument("--w-constraint", type=float, default=env_d.w_constraint)
    p.add_argument("--a-max", type=float, default=2.0, help="tanh bound on control magnitude")
    p.add_argument("--k-max", type=int, default=8)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out-dir", type=str, default="heat_rl_efno_output")
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--save-interval", type=int, default=20)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    # Run from repo root: python -m heat_rl_efno.train_ddpg_heat_efno
    main()
