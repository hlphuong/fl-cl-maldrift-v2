"""
utils/metrics.py — classification metrics + CL metrics tracker
"""
import numpy as np
from typing import Dict, List
from sklearn.metrics import (accuracy_score, f1_score,
                              precision_score, recall_score)


def binary_metrics(y_true: np.ndarray,
                   y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred,
                                    average="binary", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred,
                                           average="binary", zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred,
                                        average="binary", zero_division=0)),
    }


class CLTracker:
    """
    Luu accuracy/f1/recall matrix a[i][j]:
    metric tren task j sau khi hoc xong task i.

    Tinh ACC, Forgetting, BWT, FWT theo tung metric.
    """
    def __init__(self, num_tasks: int):
        self.T      = num_tasks
        self.mat    : Dict[int, Dict[int, float]] = {}   # accuracy
        self.f1_mat : Dict[int, Dict[int, float]] = {}   # f1
        self.rec_mat: Dict[int, Dict[int, float]] = {}   # recall
        self.pre_mat: Dict[int, Dict[int, float]] = {}   # precision
        # accuracy trên task t bằng model đã học tới t-1 (cho FWT)
        self.pretask: Dict[int, float] = {}

    def record(self, trained_on: int, evaluated_on: int,
               acc: float, f1: float = 0.0, recall: float = 0.0,
               precision: float = 0.0):
        self.mat.setdefault(trained_on, {})[evaluated_on]     = acc
        self.f1_mat.setdefault(trained_on, {})[evaluated_on]  = f1
        self.rec_mat.setdefault(trained_on, {})[evaluated_on] = recall
        self.pre_mat.setdefault(trained_on, {})[evaluated_on] = precision

    def record_pretask(self, task_id: int, acc: float):
        """Accuracy của global model (đã học tới task_id-1) trên test task_id."""
        self.pretask[task_id] = acc

    # ── Helper: tinh CL metrics tu 1 matrix ─────────────
    def _compute_from(self, mat: Dict) -> Dict[str, float]:
        T    = self.T
        last = max(mat.keys()) if mat else 0

        acc_list = [mat.get(last, {}).get(j, 0.0)
                    for j in range(T) if j in mat.get(last, {})]
        acc = float(np.mean(acc_list)) if acc_list else 0.0

        fgt_list = []
        for j in range(T - 1):
            a_jj = mat.get(j,    {}).get(j, 0.0)
            a_Tj = mat.get(last, {}).get(j, 0.0)
            fgt_list.append(max(0.0, a_jj - a_Tj))
        forgetting = float(np.mean(fgt_list)) if fgt_list else 0.0

        bwt_list = [
            mat.get(last, {}).get(j, 0.0) - mat.get(j, {}).get(j, 0.0)
            for j in range(T - 1)
            if j in mat.get(last, {}) and j in mat.get(j, {})
        ]
        bwt = float(np.mean(bwt_list)) if bwt_list else 0.0

        return {"ACC": acc, "Forgetting": forgetting, "BWT": bwt}

    def _fwt(self) -> float:
        """
        Forward Transfer (Lopez-Paz & Ranzato 2017):
            FWT = mean_{t≥1}( a_{t-1,t} − b_t )
        với a_{t-1,t} = accuracy trên task t bằng model đã học tới t-1
        (đo TRƯỚC khi train task t), b_t = baseline ngẫu nhiên = 0.5 (nhị phân).
        """
        vals = [self.pretask[t] - 0.5
                for t in range(1, self.T) if t in self.pretask]
        return float(np.mean(vals)) if vals else 0.0

    def compute(self, extra: Dict[str, float] = None) -> Dict[str, float]:
        acc_m = self._compute_from(self.mat)
        f1_m  = self._compute_from(self.f1_mat)
        rec_m = self._compute_from(self.rec_mat)
        pre_m = self._compute_from(self.pre_mat)
        out = {
            "ACC":        round(acc_m["ACC"],        4),
            "Forgetting": round(acc_m["Forgetting"], 4),
            "BWT":        round(acc_m["BWT"],        4),
            "FWT":        round(self._fwt(),         4),
            "Precision":  round(pre_m["ACC"],        4),
            "Precision_BWT": round(pre_m["BWT"],     4),
            "F1":         round(f1_m["ACC"],         4),
            "F1_BWT":     round(f1_m["BWT"],         4),
            "Recall":     round(rec_m["ACC"],        4),
            "Recall_BWT": round(rec_m["BWT"],        4),
        }
        if extra:
            out.update(extra)
        return out

    def print_matrix(self):
        T = self.T
        print("\n[CL] Accuracy Matrix (row=trained_on, col=eval_on)")
        header = "         " + "".join(f"  T{j}" for j in range(T))
        print(header)
        for i in sorted(self.mat.keys()):
            row = f"After T{i}: "
            for j in range(T):
                v = self.mat[i].get(j)
                row += f" {v:.3f}" if v is not None else "   - "
            print(row)

        print("\n[CL] F1 Matrix")
        print(header)
        for i in sorted(self.f1_mat.keys()):
            row = f"After T{i}: "
            for j in range(T):
                v = self.f1_mat[i].get(j)
                row += f" {v:.3f}" if v is not None else "   - "
            print(row)

        print("\n[CL] Recall Matrix")
        print(header)
        for i in sorted(self.rec_mat.keys()):
            row = f"After T{i}: "
            for j in range(T):
                v = self.rec_mat[i].get(j)
                row += f" {v:.3f}" if v is not None else "   - "
            print(row)

        print("\n[CL] Precision Matrix")
        print(header)
        for i in sorted(self.pre_mat.keys()):
            row = f"After T{i}: "
            for j in range(T):
                v = self.pre_mat[i].get(j)
                row += f" {v:.3f}" if v is not None else "   - "
            print(row)
        print()
