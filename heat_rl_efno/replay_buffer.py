"""Replay buffer for (s, a, r, s_next, done, n_step) transitions."""
from __future__ import annotations

import random

import torch


class ReplayBuffer:
    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.buf: list[tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, bool, int]] = []
        self.idx = 0

    def push(self, s, a, r, s_next, done, n_step: int):
        t = (s.cpu(), a.cpu(), float(r), s_next.cpu(), bool(done), int(n_step))
        if len(self.buf) < self.capacity:
            self.buf.append(t)
        else:
            self.buf[self.idx] = t
            self.idx = (self.idx + 1) % self.capacity

    def sample_with_n(self, batch_size: int):
        batch = random.sample(self.buf, min(batch_size, len(self.buf)))
        s = torch.stack([b[0] for b in batch]).to(self.device)
        a = torch.stack([b[1] for b in batch]).to(self.device)
        r = torch.tensor([b[2] for b in batch], device=self.device, dtype=torch.float32)
        s2 = torch.stack([b[3] for b in batch]).to(self.device)
        d = torch.tensor([b[4] for b in batch], device=self.device, dtype=torch.float32)
        n = torch.tensor([b[5] for b in batch], device=self.device, dtype=torch.long)
        return s, a, r, s2, d, n

    def __len__(self):
        return len(self.buf)
