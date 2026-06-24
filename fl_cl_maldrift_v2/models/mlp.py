"""
models/mlp.py — MLP nhẹ cho malware detection.
Dùng GroupNorm thay BatchNorm khi bật DP (Opacus requirement).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class SafeBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d that falls back to running stats for singleton train batches."""

    def forward(self, x):
        if self.training and x.size(0) == 1:
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                self.momentum,
                self.eps,
            )
        return super().forward(x)


class MalwareMLP(nn.Module):
    def __init__(self, input_dim: int,
                 hidden_dims: List[int] = None,
                 num_classes: int = 2,
                 dropout: float = 0.3,
                 use_dp: bool = False):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128, 64]
        layers = []
        in_d = input_dim
        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(in_d, h))
            is_last_hidden = i == len(hidden_dims) - 1
            if not is_last_hidden:
                if use_dp:
                    layers.append(nn.GroupNorm(min(32, h), h))
                else:
                    layers.append(SafeBatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if not is_last_hidden:
                layers.append(nn.Dropout(dropout))
            in_d = h
        layers.append(nn.Linear(in_d, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def get_params(self) -> List[torch.Tensor]:
        return [p.data.clone() for p in self.parameters()]

    def set_params(self, params: List[torch.Tensor]):
        for p, np_ in zip(self.parameters(), params):
            p.data.copy_(np_)
