"""
modules/continual_learning.py
Replay Buffer (Reservoir Sampling) + EWC (diagonal Fisher, multi-anchor).

EWC ở đây là Continual-Learning EWC đúng nghĩa: Fisher + θ* được
"consolidate" (đóng băng) ở RANH GIỚI mỗi task — bảo vệ trọng số quan
trọng của TẤT CẢ task đã học, chứ không phải task hiện tại.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List, Tuple


# ── Replay Buffer ────────────────────────────────────────
class ReplayBuffer:
    """
    Reservoir Sampling: mọi mẫu có xác suất bằng nhau ở trong buffer.
    """
    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self._X: List[np.ndarray] = []
        self._y: List[int]        = []
        self._n = 0                   # tổng mẫu đã thấy

    def update(self, X: np.ndarray, y: np.ndarray):
        for xi, yi in zip(X, y):
            self._n += 1
            if len(self._X) < self.max_size:
                self._X.append(xi.copy())
                self._y.append(int(yi))
            else:
                j = np.random.randint(0, self._n)
                if j < self.max_size:
                    self._X[j] = xi.copy()
                    self._y[j] = int(yi)

    def sample(self, batch: int, device: torch.device
               ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not self._X:
            return None
        n   = min(batch, len(self._X))
        idx = np.random.choice(len(self._X), n, replace=False)
        X   = torch.tensor(np.stack([self._X[i] for i in idx]),
                           dtype=torch.float32, device=device)
        y   = torch.tensor([self._y[i] for i in idx],
                           dtype=torch.long, device=device)
        return X, y

    def __len__(self):
        return len(self._X)


# ── EWC ─────────────────────────────────────────────────
class EWC:
    """
    Elastic Weight Consolidation (multi-anchor / per-task).

        L_EWC = λ · Σ_{task k} Σᵢ Fᵢ^(k) · (θᵢ − θ*ᵢ^(k))²

    Mỗi khi học xong một task ta gọi `consolidate(...)` để chốt một anchor
    (Fisher diagonal + θ*) trên dữ liệu task vừa xong. Penalty cộng dồn qua
    mọi anchor → chống catastrophic forgetting cho TẤT CẢ task cũ.
    """
    def __init__(self, lam: float = 0.4, n_samples: int = 100):
        self.lam       = lam
        self.n_samples = n_samples
        # mỗi anchor = (fisher: List[Tensor], theta_star: List[Tensor])
        self._anchors: List[Tuple[List[torch.Tensor], List[torch.Tensor]]] = []

    def consolidate(self, model: nn.Module,
                    loader: torch.utils.data.DataLoader,
                    device: torch.device):
        """Tính Fisher diagonal trên dữ liệu task vừa học xong và chốt anchor."""
        model.eval()
        criterion = nn.CrossEntropyLoss()
        fisher    = [torch.zeros_like(p) for p in model.parameters()]
        n = 0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            model.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            for f, p in zip(fisher, model.parameters()):
                if p.grad is not None:
                    f += p.grad.detach() ** 2
            n += 1
            if n >= self.n_samples:
                break
        fisher     = [f.detach() / max(n, 1) for f in fisher]
        theta_star = [p.detach().clone()     for p in model.parameters()]
        self._anchors.append((fisher, theta_star))
        model.zero_grad()
        model.train()

    def penalty(self, model: nn.Module) -> torch.Tensor:
        if not self._anchors:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        params = list(model.parameters())
        loss = torch.tensor(0.0, device=params[0].device)
        for fisher, theta_star in self._anchors:
            for f, opt, p in zip(fisher, theta_star, params):
                loss = loss + (f.to(p.device)
                               * (p - opt.to(p.device)) ** 2).sum()
        return self.lam * loss

    @property
    def ready(self) -> bool:
        return len(self._anchors) > 0

    @property
    def num_anchors(self) -> int:
        return len(self._anchors)

    # Backward-compat: gọi compute() = consolidate() một anchor.
    def compute(self, model, loader, device):
        self.consolidate(model, loader, device)
