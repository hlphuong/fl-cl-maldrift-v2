"""
fl/server.py — FL-CL-MalDrift Server

Hành vi theo method:
  - FedAvg       (drift_aware=False): pure weighted avg w=n_c, không filter, không EWMA
  - FL-MalDrift  (drift_aware=True) : filter theo drift_score, w=n_c×(1−s_c), EWMA τ
  - FL-CL-MalDrift (drift_aware=True): giống FL-MalDrift + nhận withheld từ DRC client
"""
import numpy as np
from typing import List, Dict, Tuple


class FLCLServer:
    def __init__(self, server_cfg: dict,
                 initial_params: List[np.ndarray],
                 aggregation: str = "fedavg",
                 drift_aware: bool = True):
        self.cfg         = server_cfg
        self.agg         = aggregation.lower()
        self.params      = initial_params
        self.drift_aware = drift_aware   # False → pure FedAvg (no drift filtering/EWMA)

        self._tau        = float(self.cfg.get("tau_init", 0.8))
        self._round      = 0
        self._recent     : List[float] = []
        self._round_logs : List[dict]  = []

    # ── Main interface ───────────────────────────────────
    def aggregate(self, results: List[Dict]) -> Tuple[List[np.ndarray], Dict]:
        self._round += 1
        warmup = self._round <= self.cfg["warmup_rounds"]

        accepted, scores = self._filter(results, warmup)

        if not accepted:
            print(f"[Server] Round {self._round}: no stable clients!")
            return self.params, {"active": 0, "tau": round(self._tau, 4)}

        if self.agg == "fedavg":
            new_params = self._fedavg(accepted)
        else:
            new_params = self._fedsgd(accepted)

        self.params = new_params

        # EWMA τ chỉ cập nhật khi drift_aware (FL-MalDrift / FL-CL-MalDrift)
        if self.drift_aware and not warmup and scores:
            self._update_tau(scores, len(accepted), len(results))

        metrics = {
            "round":         self._round,
            "active":        len(accepted),
            "total":         len(results),
            "tau":           round(self._tau, 4),
            "avg_score":     round(float(np.mean(scores)) if scores else 0, 4),
            "avg_train_acc": round(float(np.mean(
                [r.get("train_acc", 0) for r in accepted])), 4),
        }
        self._round_logs.append(metrics)
        print(f"[Server] Round {self._round}: "
              f"active={len(accepted)}/{len(results)} "
              f"| τ={self._tau:.3f} "
              f"| avg_score={metrics['avg_score']:.3f} "
              f"| avg_acc={metrics['avg_train_acc']:.3f}")
        return new_params, metrics

    # ── Filtering ────────────────────────────────────────
    def _filter(self, results: List[Dict], warmup: bool
                ) -> Tuple[List[Dict], List[float]]:
        # client hợp lệ (không bị withheld bởi DRC, có dữ liệu)
        eligible = [r for r in results
                    if not r.get("withheld", False) and r.get("n_samples", 0) > 0]
        scores = [float(r.get("drift_score", 0.0)) for r in eligible]

        # FedAvg thuần hoặc warmup: nhận tất cả
        if not self.drift_aware or warmup:
            return list(eligible), scores

        # Drift-aware: accept (stable) / accept+down-weight (warning) / reject
        accepted = [r for r in eligible
                    if float(r.get("drift_score", 0.0)) <= self._tau * 1.5]

        # Participation guard: không để số client tổng hợp rớt quá sàn —
        # nếu reject quá tay, giữ lại các client drift thấp nhất.
        min_keep = max(2, int(np.ceil(
            self.cfg.get("min_participation", 0.5) * len(eligible))))
        if eligible and len(accepted) < min_keep:
            ranked   = sorted(eligible,
                              key=lambda r: float(r.get("drift_score", 0.0)))
            accepted = ranked[:min_keep]

        return accepted, scores

    # ── Aggregation ──────────────────────────────────────
    def _fedavg(self, accepted: List[Dict]) -> List[np.ndarray]:
        weights, all_p = [], []
        for r in accepted:
            p = r["params"]
            n = r.get("n_samples", 1)
            if self.drift_aware:
                # FL-MalDrift / FL-CL-MalDrift: w_c = n_c × (1 − s_c)
                s = r.get("drift_score", 0.0)
                w = n * max(0.1, 1.0 - s)
            else:
                # FedAvg thuần: w_c = n_c
                w = float(n)
            weights.append(w)
            all_p.append(p)
        total_w = sum(weights)
        return [
            sum(w * p[i] for w, p in zip(weights, all_p)) / max(total_w, 1e-8)
            for i in range(len(all_p[0]))
        ]

    def _fedsgd(self, accepted: List[Dict]) -> List[np.ndarray]:
        """Uniform average (1 epoch per client → ≈ FedSGD)."""
        all_p = [r["params"] for r in accepted]
        n = len(all_p)
        return [sum(p[i] for p in all_p) / n for i in range(len(all_p[0]))]

    # ── EWMA Threshold ────────────────────────────────────
    def _update_tau(self, scores: List[float],
                    n_active: int, n_total: int):
        """τ_t = clip(α·τ_{t-1} + (1-α)·(μ_W + k·σ_W) + η·(p*-p_t))"""
        cfg = self.cfg
        self._recent.extend(scores)
        if len(self._recent) > 10:
            self._recent = self._recent[-10:]

        mu      = float(np.mean(self._recent))
        sigma   = float(np.std(self._recent)) + 1e-8
        tau_hat = mu + cfg["k_sigma"] * sigma

        p_t   = n_active / max(n_total, 1)
        new_tau = (cfg["ewma_alpha"] * self._tau
                   + (1 - cfg["ewma_alpha"]) * tau_hat
                   + cfg["eta"] * (cfg["target_participation"] - p_t))
        self._tau = float(np.clip(new_tau, cfg["tau_min"], cfg["tau_max"]))

    @property
    def tau(self) -> float:
        return self._tau

    @property
    def round_logs(self) -> List[dict]:
        return self._round_logs
