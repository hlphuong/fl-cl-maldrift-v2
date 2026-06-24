"""
modules/privacy.py — Manual DP-SGD
Clip gradient + Gaussian noise. Không cần Opacus.
"""
import math
import torch
import torch.nn as nn


class DPSGD:
    """
    Bước 1: clip L2 norm gradient về ≤ C
    Bước 2: thêm N(0, σ²C²I) vào gradient

    Gọi sau loss.backward(), trước optimizer.step().
    """
    def __init__(self, noise_multiplier: float = 1.2,
                 max_grad_norm: float = 1.0,
                 enabled: bool = True):
        self.sigma   = noise_multiplier
        self.C       = max_grad_norm
        self.enabled = enabled

    def clip_and_noise(self, model: nn.Module):
        if not self.enabled:
            return
        # clip
        norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                norm += p.grad.detach().norm(2).item() ** 2
        norm = math.sqrt(norm)
        coef = self.C / max(norm, self.C)
        if coef < 1.0:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.detach().mul_(coef)
        # noise
        for p in model.parameters():
            if p.grad is not None:
                noise = torch.randn_like(p.grad) * self.sigma * self.C
                p.grad.detach().add_(noise)

    def epsilon(self, T: int, delta: float = 1e-5) -> float:
        """Tính ε tích lũy qua T rounds (advanced composition)."""
        if not self.enabled:
            return float("inf")
        t1 = math.sqrt(2 * T * math.log(1 / delta)) / self.sigma
        t2 = T * (math.exp(1 / self.sigma) - 1) / self.sigma
        return t1 + t2
