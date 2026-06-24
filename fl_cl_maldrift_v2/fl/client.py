"""
fl/client.py — FL-CL-MalDrift Client

Pipeline mỗi round:
  1. Nhận θ_t từ server → set vào local model
  2. Train 1 epoch (+ Replay loss hoặc + EWC penalty tùy stage)
  3. HDDM-W tính drift score
  4. DRC.step() → quyết định stage
  5. DP-SGD clip+noise
  6. Trả về (params, drift_score, flag, consecutive_count)
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List, Optional

from models.mlp import MalwareMLP
from modules.drift_detector import get_detector
from modules.drc import DRC, Stage
from modules.privacy import DPSGD
from utils.metrics import binary_metrics


class FLCLClient:
    def __init__(self, client_id: int,
                 train_loader: DataLoader,
                 val_loader:   DataLoader,
                 test_loader:  DataLoader,
                 input_dim:    int,
                 model_cfg:    dict,
                 drift_cfg:    dict,
                 drc_cfg:      dict,
                 cl_cfg:       dict,
                 privacy_cfg:  dict,
                 local_epochs: int = 1,
                 device:       Optional[torch.device] = None):

        self.cid    = client_id
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.local_epochs = max(1, int(local_epochs))

        # model
        self.model = MalwareMLP(
            input_dim   = input_dim,
            hidden_dims = model_cfg["hidden_dims"],
            dropout     = model_cfg["dropout"],
            use_dp      = privacy_cfg.get("enabled", False),
        ).to(self.device)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=0.01, momentum=0.9, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()

        # modules
        self.detector = get_detector(drift_cfg["detector"], drift_cfg)
        self.local_tau = drift_cfg.get("local_threshold", 0.5)
        self.drc  = DRC(drc_cfg, cl_cfg)
        self.dp   = DPSGD(privacy_cfg.get("noise_multiplier", 1.2),
                          privacy_cfg.get("max_grad_norm",    1.0),
                          privacy_cfg.get("enabled",          True))
        self.rb_batch = cl_cfg.get("replay_batch_size", 32)

        self._round        = 0
        self._server_tau   = 0.5
        self._total_rounds = 0   # cho epsilon computation

    # ── Flower-compatible interface ──────────────────────
    def set_parameters(self, params: List[np.ndarray]):
        tensors = [torch.tensor(p, device=self.device) for p in params]
        self.model.set_params(tensors)

    def get_parameters(self) -> List[np.ndarray]:
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def end_task(self, global_params: List[np.ndarray]):
        """
        Gọi ở RANH GIỚI task (sau khi học xong task hiện tại, trước khi đổi
        loaders). Chốt một EWC anchor trên dữ liệu task vừa xong, dùng trọng số
        GLOBAL sau task làm θ* → bảo vệ kiến thức task này về sau.
        """
        self.set_parameters(global_params)
        print(f"  [DRC] Client {self.cid:02d} consolidating EWC anchor "
              f"(task done) ...")
        self.drc.consolidate_ewc(self.model, self.train_loader, self.device)

    def update_loaders(self, train_loader: DataLoader,
                       val_loader: DataLoader,
                       test_loader: DataLoader):
        """
        Cập nhật data loaders khi chuyển task mới.
        - GIỮ: replay buffer, drift detector state, EWC anchors
          (buffer chứa data task cũ để replay; anchors bảo vệ task cũ)
        - RESET: DRC escalation state (count/stage/withheld)
          (mỗi task bắt đầu fresh, tránh withheld tích lũy từ task trước)
        """
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader

        # Reset DRC escalation state — task mới bắt đầu từ STABLE
        self.drc.reset_episode()

        # Reset optimizer để tránh momentum cũ từ task trước nhiễu training
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=0.01, momentum=0.9, weight_decay=1e-4)

    def drc_stats(self) -> Dict:
        """Thống kê escalation/recovery của client (cho main aggregate)."""
        return self.drc.stats()

    def fit(self, params: List[np.ndarray],
            server_tau: float = 0.5, round_id: int = 0
            ) -> Dict:
        """
        Nhận params từ server, train 1 round, trả về kết quả.
        """
        self._round      = round_id
        self._server_tau = server_tau
        self._total_rounds += 1

        # 1. cập nhật model
        self.set_parameters(params)

        # 2. train + drift detect
        loss, acc, score = self._train_round()

        # 3. DRC quyết định
        stage = self.drc.step(score, server_tau)

        result = {
            "client_id":   self.cid,
            "drift_score": round(float(score), 4),
            "stage":       stage.name,
            "count":       self.drc.count,
            "withheld":    self.drc.withheld,
            "train_loss":  round(float(loss), 4),
            "train_acc":   round(float(acc),  4),
            "n_samples":   0 if self.drc.withheld else
                           len(self.train_loader.dataset),
        }

        if not self.drc.withheld:
            result["params"] = self.get_parameters()

        self._log_round(stage, score, loss, acc)
        return result

    def evaluate(self, params: List[np.ndarray]) -> Dict:
        self.set_parameters(params)
        loss, acc, f1, recall, precision = self._eval(self.test_loader)
        return {"loss": loss, "accuracy": acc, "f1": f1,
                "recall": recall, "precision": precision,
                "n_samples": len(self.test_loader.dataset),
                "client_id": self.cid}

    # ── Training logic ───────────────────────────────────
    def _train_round(self) -> Tuple[float, float, float]:
        self.model.train()
        stage = self.drc.stage

        total_loss = 0.0; correct = 0; total = 0
        errors = []

        for _ in range(self.local_epochs):
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                logits = self.model(X)
                loss   = self.criterion(logits, y)

                # CL kích hoạt khi đang xử lý drift (Stage ≥ Replay):
                #   - Replay buffer (mẫu task cũ) từ Stage 1 trở đi
                #   - EWC penalty (Fisher anchors) ở Stage 2 (EWC)
                if stage >= Stage.REPLAY and len(self.drc.buf) > 0:
                    loss = loss + self.drc.replay_loss(
                        self.model, self.criterion, self.device, self.rb_batch)
                if stage == Stage.EWC and self.drc.ewc.ready:
                    loss = loss + self.drc.ewc_loss(self.model)

                loss.backward()
                self.dp.clip_and_noise(self.model)   # DP-SGD
                self.optimizer.step()

                # update replay buffer
                self.drc.update_buffer(
                    X.detach().cpu().numpy(),
                    y.detach().cpu().numpy())

                # error stream cho detector
                with torch.no_grad():
                    preds = logits.argmax(1)
                    errs  = (preds != y).float().cpu().numpy()
                    errors.extend(errs.tolist())

                total_loss += loss.item() * len(y)
                correct    += (preds == y).sum().item()
                total      += len(y)

        # cập nhật detector
        for e in errors:
            self.detector.update(e)
        score = self.detector.score()

        return (total_loss / max(total, 1),
                correct    / max(total, 1),
                score)

    def _eval_with(self, params: List[np.ndarray],
                   loader: DataLoader) -> Tuple[float, float, float, float, float]:
        """Set params rồi đánh giá trên loader bất kỳ (dùng cho FWT pre-eval)."""
        self.set_parameters(params)
        return self._eval(loader)

    def _eval(self, loader: DataLoader) -> Tuple[float, float, float, float, float]:
        self.model.eval()
        losses, preds, labels = [], [], []
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                out  = self.model(X)
                losses.append(self.criterion(out, y).item())
                preds.extend(out.argmax(1).cpu().numpy())
                labels.extend(y.cpu().numpy())
        m = binary_metrics(np.array(labels), np.array(preds))
        return float(np.mean(losses)), m["accuracy"], m["f1"], m["recall"], m["precision"]

    def _log_round(self, stage, score, loss, acc):
        withheld = self.drc.withheld
        flag = "🔴 WITHHELD" if withheld else "✓"
        print(f"  Client {self.cid:02d} | round={self._round:3d} "
              f"| stage={stage.name:10s} | score={score:.3f} "
              f"| loss={loss:.4f} | acc={acc:.3f} {flag}")
