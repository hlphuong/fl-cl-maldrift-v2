"""
modules/drc.py — Drift Resolution Controller (DRC)

Triển khai trung thực Algorithm 3 (slide 7): 3 giai đoạn leo thang
    Stage 1  Replay     : consecutive_drift ≤ K1
    Stage 2  EWC        : K1 < consecutive_drift ≤ K2
    Stage 3  Escalation : consecutive_drift > K2  → withheld + Recovery Monitor

Recovery Monitor: theo dõi EMA của drift score; tái gia nhập khi
    ema(s) < τ_re = τ_t + δ  liên tiếp R vòng.

Mỗi round client gọi drc.step(score, server_tau) một lần.
Bộ đếm consecutive_drift tồn tại xuyên suốt nhiều round trong một task.
"""
from enum import IntEnum
import numpy as np
import torch
import torch.nn as nn
from modules.continual_learning import ReplayBuffer, EWC


class Stage(IntEnum):
    STABLE     = 0
    REPLAY     = 1
    EWC        = 2
    ESCALATION = 3
    RECOVERY   = 4


class DRC:
    def __init__(self, cfg: dict, cl_cfg: dict):
        self.K1    = cfg["K1"]            # 3
        self.K2    = cfg["K2"]            # 8
        self.R     = cfg["R"]             # 3
        self.delta = cfg["delta"]         # 0.05

        self.buf = ReplayBuffer(cl_cfg["replay_buffer_size"])
        self.ewc = EWC(cl_cfg["ewc_lambda"], cl_cfg["ewc_fisher_samples"])

        self.count    = 0                 # consecutive_drift
        self.stage    = Stage.STABLE
        self.withheld = False

        self._stable_cnt = 0
        self._ema        = 0.0
        self._ema_alpha  = cfg.get("ema_alpha", 0.3)

        # ── thống kê để tính Escalation Rate / Recovery Rate ──
        self.total_rounds     = 0   # tổng round đã xử lý
        self.drift_rounds     = 0   # số round drift được phát hiện (score > τ)
        self.escalation_events = 0  # số lần leo thang sang Stage 3
        self.recovery_events  = 0   # số lần phục hồi & tái gia nhập thành công
        self.rounds_withheld  = 0   # tổng số round bị tạm dừng

        # log chi tiết theo round
        self.log: list = []

    # ── Giao diện chính ─────────────────────────────────
    def step(self, score: float, server_tau: float,
             local_tau: float = None) -> Stage:
        """Gọi mỗi round sau khi tính drift score. Trả về Stage hiện tại."""
        self.total_rounds += 1
        self._ema = self._ema_alpha * score + (1 - self._ema_alpha) * self._ema

        tau = server_tau if server_tau is not None else local_tau
        if tau is None:
            tau = 0.5
        is_drift = score > tau
        if is_drift:
            self.drift_rounds += 1

        # đang ở chế độ withheld → chỉ theo dõi recovery
        if self.withheld:
            self.rounds_withheld += 1
            stage = self._check_recovery(server_tau)
        elif not is_drift:
            # không drift → reset
            self.count = 0
            self.stage = Stage.STABLE
            stage = self.stage
        else:
            # drift → tăng bộ đếm và chọn giai đoạn
            self.count += 1
            if self.count <= self.K1:
                self.stage = Stage.REPLAY
            elif self.count <= self.K2:
                self.stage = Stage.EWC
            else:
                self.stage             = Stage.ESCALATION
                self.withheld          = True
                self._stable_cnt       = 0
                self.escalation_events += 1
                print(f"[DRC] Escalation → client withheld "
                      f"(consecutive_drift={self.count}).")
            stage = self.stage

        self.log.append({"round":    self.total_rounds,
                         "count":    self.count,
                         "score":    round(score, 4),
                         "tau":      round(float(tau), 4),
                         "ema":      round(self._ema, 4),
                         "stage":    self.stage.name,
                         "withheld": self.withheld})
        return stage

    def _check_recovery(self, server_tau: float) -> Stage:
        """Recovery Monitor: ema(s) < τ_re trong R vòng → tái gia nhập."""
        tau_re = server_tau + self.delta
        if self._ema < tau_re:
            self._stable_cnt += 1
        else:
            self._stable_cnt = 0

        if self._stable_cnt >= self.R:
            self.count          = 0
            self.withheld       = False
            self._stable_cnt    = 0
            self.stage          = Stage.STABLE
            self.recovery_events += 1
            print("[DRC] Client recovered → rejoining federation.")
        else:
            self.stage = Stage.RECOVERY
        return self.stage

    # ── Reset ở ranh giới task ──────────────────────────
    def reset_episode(self):
        """Bắt đầu task mới: reset escalation state, GIỮ buffer + EWC anchors."""
        self.count       = 0
        self.stage       = Stage.STABLE
        self.withheld    = False
        self._stable_cnt = 0

    # ── Helpers cho training ─────────────────────────────
    def replay_loss(self, model: nn.Module, criterion: nn.Module,
                    device: torch.device, batch: int) -> torch.Tensor:
        sample = self.buf.sample(batch, device)
        if sample is None:
            return torch.tensor(0.0, device=device)
        X, y = sample
        return criterion(model(X), y)

    def ewc_loss(self, model: nn.Module) -> torch.Tensor:
        return self.ewc.penalty(model)

    def update_buffer(self, X: np.ndarray, y: np.ndarray):
        self.buf.update(X, y)

    def consolidate_ewc(self, model, loader, device):
        """Chốt một EWC anchor cho task vừa học xong."""
        self.ewc.consolidate(model, loader, device)

    @property
    def should_send(self) -> bool:
        return not self.withheld

    @property
    def flag(self) -> str:
        return self.stage.name

    # ── Thống kê ─────────────────────────────────────────
    def stats(self) -> dict:
        esc_rate = self.escalation_events / max(self.drift_rounds, 1)
        rec_rate = (self.recovery_events / self.escalation_events
                    if self.escalation_events > 0 else None)
        return {
            "total_rounds":      self.total_rounds,
            "drift_rounds":      self.drift_rounds,
            "escalation_events": self.escalation_events,
            "recovery_events":   self.recovery_events,
            "rounds_withheld":   self.rounds_withheld,
            "escalation_rate":   esc_rate,
            "recovery_rate":     rec_rate,   # None nếu chưa từng escalation
        }
