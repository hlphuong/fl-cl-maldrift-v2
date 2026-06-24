"""
utils/logger.py — JSON logger + matplotlib plots
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict


class Logger:
    def __init__(self, out_dir: str, name: str):
        self.out  = out_dir
        self.name = name
        os.makedirs(out_dir, exist_ok=True)
        self._rounds : List[dict] = []
        self._tasks  : List[dict] = []

    def log_round(self, d: dict):   self._rounds.append(d)
    def log_task(self, d: dict):    self._tasks.append(d)

    def save(self):
        path = os.path.join(self.out, f"{self.name}.json")
        with open(path, "w") as f:
            json.dump({"rounds": self._rounds,
                       "tasks":  self._tasks}, f, indent=2)
        print(f"[Log] Saved → {path}")

    def plot_accuracy(self):
        if not self._rounds: return
        rs  = [r["round"] for r in self._rounds]
        acc = [r.get("avg_train_acc", 0) for r in self._rounds]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(rs, acc, color="#2563EB", linewidth=2)
        ax.set(xlabel="Round", ylabel="Accuracy",
               title=f"{self.name} — Accuracy over rounds")
        ax.grid(True, alpha=.3)
        plt.tight_layout()
        p = os.path.join(self.out, f"{self.name}_acc.png")
        plt.savefig(p, dpi=150); plt.close()
        print(f"[Log] Plot → {p}")

    def plot_drift(self):
        if not self._rounds: return
        rs  = [r["round"] for r in self._rounds]
        sc  = [r.get("avg_score", 0) for r in self._rounds]
        tau = [r.get("tau", 0.5)     for r in self._rounds]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(rs, sc,  color="#D97706", linewidth=2, label="avg drift score")
        ax.plot(rs, tau, color="#DC2626", linewidth=1.5,
                linestyle="--", label="τ threshold")
        ax.fill_between(rs, sc, tau,
                        where=[s > t for s, t in zip(sc, tau)],
                        alpha=.2, color="#DC2626")
        ax.set(xlabel="Round", ylabel="Drift score",
               title=f"{self.name} — Drift & Threshold")
        ax.legend(); ax.grid(True, alpha=.3)
        plt.tight_layout()
        p = os.path.join(self.out, f"{self.name}_drift.png")
        plt.savefig(p, dpi=150); plt.close()
        print(f"[Log] Plot → {p}")
