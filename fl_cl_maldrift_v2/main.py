"""
main.py — FL-CL-MalDrift Entry Point

Chạy:
    python main.py                          # default: synthetic, fl_cl_maldrift
    python main.py --method fedavg          # baseline
    python main.py --method fl_maldrift     # không CL
    python main.py --dataset drebin         # dùng Drebin
    python main.py --no_dp                  # tắt Differential Privacy
    python main.py --tasks 3 --rounds 10    # quick test

So sánh 3 phương pháp:
    python main.py --compare
"""
import argparse
import os
import sys
import json
import copy
import numpy as np
import torch

# Console Windows (cp1252) không in được ký tự Unicode (─, →, 🔴...) → ép UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from config import FL, DATA, MODEL, DRIFT, DRC, CL, PRIVACY, SERVER, DATASET_PRESETS
from data.dataset import DataManager
from models.mlp import MalwareMLP
from fl.client import FLCLClient
from fl.server import FLCLServer
from utils.metrics import CLTracker, binary_metrics
from utils.logger import Logger


# ── CLI ──────────────────────────────────────────────────
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--method",  default="fl_cl_maldrift",
                   choices=["fedavg","fl_maldrift","fl_cl_maldrift"])
    p.add_argument("--dataset", default="synthetic",
                   choices=["synthetic","drebin","cicmaldroid"])
    p.add_argument("--tasks",   type=int, default=None)
    p.add_argument("--rounds",  type=int, default=None)
    p.add_argument("--clients", type=int, default=None)
    p.add_argument("--local_epochs", type=int, default=None,
                   help="Số epoch local mỗi communication round")
    p.add_argument("--alpha", type=float, default=None,
                   help="Dirichlet alpha cho Non-IID client partition; nhỏ hơn = Non-IID mạnh hơn")
    p.add_argument("--task_strategy", default=None,
                   choices=["temporal", "category", "category_strict", "category_revisit"],
                   help="Cách tạo continual tasks")
    p.add_argument("--partition_strategy", default=None,
                   choices=["dirichlet", "category"],
                   help="Cách chia mỗi task cho clients")
    p.add_argument("--ewc_lambda", type=float, default=None,
                   help="Override λ_EWC cho FL-CL-MalDrift")
    p.add_argument("--replay_buffer", type=int, default=None,
                   help="Override kích thước replay buffer cho FL-CL-MalDrift")
    p.add_argument("--tau_init", type=float, default=None,
                   help="Ngưỡng drift ban đầu của server")
    p.add_argument("--tau_min", type=float, default=None,
                   help="Sàn ngưỡng drift của server")
    p.add_argument("--warmup_rounds", type=int, default=None,
                   help="Số round đầu nhận mọi client trước khi lọc drift")
    p.add_argument("--no_dp",   action="store_true",
                   help="Tắt DP cho FL-CL-MalDrift (mặc định OFF khi --compare)")
    p.add_argument("--dp",      action="store_true",
                   help="Bật DP cho FL-CL-MalDrift (dùng riêng khi ablation DP)")
    p.add_argument("--compare", action="store_true",
                   help="Chạy cả 3 phương pháp rồi so sánh")
    p.add_argument("--drc_stress", action="store_true",
                   help="E3 DRC stress test: hạ K2 để buộc Escalation/Recovery xảy ra")
    p.add_argument("--runs",    type=int, default=1,
                   help="Số lần chạy độc lập (seed khác nhau), báo cáo mean±std")
    p.add_argument("--kfold",   type=int, default=0,
                   help="Chạy Stratified K-Fold theo class_id; ví dụ --kfold 5")
    p.add_argument("--seed",    type=int, default=None,
                   help="Override random seed (mặc định dùng config)")
    p.add_argument("--out",     default="results/")
    return p.parse_args()


# ── Build configs ─────────────────────────────────────────
def make_configs(args):
    preset = DATASET_PRESETS.get(args.dataset, {})
    dcfg = {**DATA, **preset, "dataset": args.dataset}
    flcfg = copy.deepcopy(FL)
    if args.tasks:   dcfg["num_tasks"]      = args.tasks
    if args.rounds:  flcfg["num_rounds"]    = args.rounds
    if args.clients: flcfg["num_clients"]   = args.clients
    if args.local_epochs: flcfg["local_epochs"] = args.local_epochs
    if args.alpha is not None:
        dcfg["non_iid_alpha"] = args.alpha
    if args.task_strategy is not None:
        dcfg["task_strategy"] = args.task_strategy
    if args.partition_strategy is not None:
        dcfg["partition_strategy"] = args.partition_strategy
    if getattr(args, "seed", None) is not None:
        dcfg["random_seed"] = args.seed
    dcfg["num_clients"] = flcfg["num_clients"]

    # DP chỉ bật khi tường minh --dp (tránh làm lệch kết quả so sánh 3 method)
    dp_on   = getattr(args, "dp", False) and not getattr(args, "no_dp", False)
    priv    = {**PRIVACY, "enabled": dp_on}
    drc_cfg = copy.deepcopy(DRC)
    cl_cfg  = copy.deepcopy(CL)
    srv_cfg = copy.deepcopy(SERVER)
    if args.ewc_lambda is not None:
        cl_cfg["ewc_lambda"] = args.ewc_lambda
    if args.replay_buffer is not None:
        cl_cfg["replay_buffer_size"] = args.replay_buffer
    if args.tau_init is not None:
        srv_cfg["tau_init"] = args.tau_init
    if args.tau_min is not None:
        srv_cfg["tau_min"] = args.tau_min
    if args.warmup_rounds is not None:
        srv_cfg["warmup_rounds"] = args.warmup_rounds

    # ── Ablation 3 method (trung thực với slide 3 & 4) ──────
    if args.method == "fedavg":
        # Baseline: FL thuần, không drift-handling, không CL, không DP.
        drc_cfg = {**DRC, "K1": 9999, "K2": 9999}
        cl_cfg  = {**CL,  "replay_buffer_size": 0, "ewc_lambda": 0.0}
        priv    = {**PRIVACY, "enabled": False}

    elif args.method == "fl_maldrift":
        # Bài báo gốc: drift detection + drift-aware server filtering,
        # KHÔNG Continual Learning (đây chính là gap → catastrophic forgetting),
        # KHÔNG escalation (K2 = ∞).
        drc_cfg = {**DRC, "K2": 9999}
        cl_cfg  = {**CL, "replay_buffer_size": 0, "ewc_lambda": 0.0}
        priv    = {**PRIVACY, "enabled": False}

    else:
        # fl_cl_maldrift: DRC 3 giai đoạn + CL (Replay+EWC) mặc định.
        # --drc_stress (E3): hạ ngưỡng leo thang để Escalation/Recovery chắc
        # chắn kích hoạt — Stage 3 ngay khi có 2 round drift liên tiếp.
        if getattr(args, "drc_stress", False):
            drc_cfg = {**drc_cfg, "K1": 1, "K2": 1}

    return dcfg, flcfg, drc_cfg, cl_cfg, priv, srv_cfg
    


# ── Build clients ─────────────────────────────────────────
def build_clients(loaders, input_dim, method, flcfg,
                  drc_cfg, cl_cfg, priv):
    clients = []
    for c, ldr in enumerate(loaders):
        clients.append(FLCLClient(
            client_id    = c,
            train_loader = ldr["train"],
            val_loader   = ldr["val"],
            test_loader  = ldr["test"],
            input_dim    = input_dim,
            model_cfg    = MODEL,
            drift_cfg    = DRIFT,
            drc_cfg      = drc_cfg,
            cl_cfg       = cl_cfg,
            privacy_cfg  = priv,
            local_epochs = flcfg.get("local_epochs", 1),
        ))
    return clients


# ── Run one task ──────────────────────────────────────────
def run_task(task_id: int, clients, server: FLCLServer,
             num_rounds: int, logger: Logger):
    """
    Chay num_rounds communication rounds cho task task_id.
    Tra ve (mean_acc, mean_f1, mean_recall).
    """
    print(f"\n{'─'*55}")
    print(f" TASK {task_id}  ({num_rounds} rounds)")
    print(f"{'─'*55}")

    for rnd in range(1, num_rounds + 1):
        current_params = server.params
        tau            = server.tau
        results        = []

        for client in clients:
            res = client.fit(current_params,
                             server_tau=tau, round_id=rnd)
            results.append(res)

        new_params, srv_metrics = server.aggregate(results)
        srv_metrics["task"] = task_id
        logger.log_round(srv_metrics)

    # evaluate tren test set cua task nay
    accs, f1s, recs, pres = [], [], [], []
    for client in clients:
        ev = client.evaluate(server.params)
        accs.append(ev["accuracy"])
        f1s.append(ev["f1"])
        recs.append(ev["recall"])
        pres.append(ev["precision"])

    mean_acc = float(np.mean(accs))
    mean_f1  = float(np.mean(f1s))
    mean_rec = float(np.mean(recs))
    mean_pre = float(np.mean(pres))
    print(f" Task {task_id} | Acc={mean_acc:.4f}  Precision={mean_pre:.4f}"
          f"  F1={mean_f1:.4f}  Recall={mean_rec:.4f}")
    return mean_acc, mean_f1, mean_rec, mean_pre


# ── Main experiment ───────────────────────────────────────
def run_experiment(method: str, args, dcfg, flcfg,
                   drc_cfg, cl_cfg, priv, srv_cfg, out_dir) -> dict:
    print(f"\n{'='*55}")
    print(f" Method: {method.upper()}  |  Dataset: {dcfg['dataset'].upper()}")
    print(f" Tasks: {dcfg['num_tasks']}  |  Rounds/task: {flcfg['num_rounds']}  "
          f"|  Clients: {flcfg['num_clients']}")
    print(f" DP: {'ON' if priv['enabled'] else 'OFF'}")
    print(f"{'='*55}")

    torch.manual_seed(dcfg["random_seed"])
    np.random.seed(dcfg["random_seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Device: {device}")

    # 1. data
    dm = DataManager(dcfg, flcfg)
    all_tasks = dm.prepare()
    input_dim = dcfg["num_features"]

    # 2. init model
    init_model = MalwareMLP(
        input_dim   = input_dim,
        hidden_dims = MODEL["hidden_dims"],
        dropout     = MODEL["dropout"],
        use_dp      = priv["enabled"],
    )
    init_params = [p.detach().cpu().numpy() for p in init_model.parameters()]

    # 3. server — FedAvg dùng pure aggregation (drift_aware=False)
    server = FLCLServer(
        server_cfg     = srv_cfg,
        initial_params = init_params,
        aggregation    = flcfg["aggregation"],
        drift_aware    = (method != "fedavg"),
    )

    # 4. CL tracker + logger
    cl = CLTracker(dcfg["num_tasks"])
    logger = Logger(out_dir, f"{method}_{dcfg['dataset']}")

    # 5. run tasks
    # Build clients MOT LAN truoc vong lap task.
    # Giua cac task chi update data loaders, giu nguyen:
    #   - Replay buffer (task t co data cua task 0..t-1 de replay)
    #   - Drift detector (thay duoc transition distribution giua 2 task)
    #   - DRC count/stage va EWC Fisher (bao ve weights cu)
    clients = build_clients(all_tasks[0], input_dim, method,
                            flcfg, drc_cfg, cl_cfg, priv)

    for t, task_loaders in enumerate(all_tasks):
        if t > 0:
            # (a) FWT: đo accuracy trên task t BẰNG model đã học tới t-1
            #     (trước khi train task t) → forward transfer.
            pre_accs = []
            for c, ldr in enumerate(task_loaders):
                _, a_p, _, _, _ = clients[c]._eval_with(server.params,
                                                        ldr["test"])
                pre_accs.append(a_p)
            cl.record_pretask(t, float(np.mean(pre_accs)))

            # (b) Chốt EWC anchor cho task t-1 (dùng global model sau task)
            #     và (c) đổi sang data loaders của task t.
            for c, ldr in enumerate(task_loaders):
                clients[c].end_task(server.params)
                clients[c].update_loaders(
                    ldr["train"], ldr["val"], ldr["test"])

        acc_t, f1_t, rec_t, pre_t = run_task(t, clients, server,
                                              flcfg["num_rounds"], logger)
        cl.record(t, t, acc_t, f1_t, rec_t, pre_t)

        # evaluate tren tat ca tasks cu (BWT/Forgetting)
        for prev_t in range(t):
            accs_p, f1s_p, recs_p, pres_p = [], [], [], []
            for c, client in enumerate(clients):
                test_ldr = all_tasks[prev_t][c]["test"]
                _, acc_p, f1_p, rec_p, pre_p = client._eval_with(
                    server.params, test_ldr)
                accs_p.append(acc_p)
                f1s_p.append(f1_p)
                recs_p.append(rec_p)
                pres_p.append(pre_p)
            cl.record(t, prev_t, float(np.mean(accs_p)),
                      float(np.mean(f1s_p)), float(np.mean(recs_p)),
                      float(np.mean(pres_p)))

        logger.log_task({"task": t, "acc": acc_t, "precision": pre_t,
                         "f1": f1_t, "recall": rec_t, "cl": cl.compute()})

    # 6. final metrics — gồm DRC Escalation Rate & Recovery Rate (slide 8)
    stats = [c.drc_stats() for c in clients]
    tot_drift = sum(s["drift_rounds"]      for s in stats)
    tot_esc   = sum(s["escalation_events"] for s in stats)
    tot_rec   = sum(s["recovery_events"]   for s in stats)
    esc_rate  = tot_esc / max(tot_drift, 1)
    rec_rate  = (tot_rec / tot_esc) if tot_esc > 0 else 0.0
    extra = {"EscalationRate": round(esc_rate, 4),
             "RecoveryRate":   round(rec_rate, 4)}

    cl_metrics = cl.compute(extra=extra)
    print(f"\n[DRC] drift_rounds={tot_drift}  escalations={tot_esc}  "
          f"recoveries={tot_rec}  "
          f"→ EscalationRate={esc_rate:.3f}  RecoveryRate={rec_rate:.3f}"
          + ("" if tot_esc > 0 else "  (chưa có escalation)"))
    cl.print_matrix()
    print("\n[Results]")
    print(f"  {'Metric':<14} {'Value':>8}")
    print(f"  {'-'*24}")
    for k, v in cl_metrics.items():
        print(f"  {k:<14} {v:>8.4f}")

    # DP epsilon
    if priv["enabled"]:
        from modules.privacy import DPSGD
        dp = DPSGD(priv["noise_multiplier"], priv["max_grad_norm"])
        eps = dp.epsilon(flcfg["num_rounds"] * dcfg["num_tasks"])
        print(f"  DP epsilon  : {eps:.2f}")
        cl_metrics["dp_epsilon"] = round(eps, 3)

    # Thêm Precision (trung bình qua các task) vào cl_metrics
    cl_metrics.setdefault("Precision", 0.0)

    logger.save()
    logger.plot_accuracy()
    logger.plot_drift()

    return cl_metrics, logger._rounds, logger._tasks


# ── Comparison CSV + Chart ────────────────────────────────
def _save_comparison_csv(all_rounds, all_tasks, out_dir):
    import csv
    # rounds CSV
    rpath = os.path.join(out_dir, "comparison_rounds.csv")
    with open(rpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "method","task","round","train_acc","drift_score","tau","active","total"])
        w.writeheader()
        for method, rounds in all_rounds.items():
            for r in rounds:
                w.writerow({"method": method,
                            "task":        r.get("task", ""),
                            "round":       r.get("round", ""),
                            "train_acc":   round(r.get("avg_train_acc", 0), 4),
                            "drift_score": round(r.get("avg_score", 0), 4),
                            "tau":         round(r.get("tau", 0), 4),
                            "active":      r.get("active", ""),
                            "total":       r.get("total", "")})
    print(f"[CSV] rounds → {rpath}")

    # tasks CSV
    tpath = os.path.join(out_dir, "comparison_tasks.csv")
    with open(tpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "method","task","acc","precision","f1","recall"])
        w.writeheader()
        for method, tasks in all_tasks.items():
            for t in tasks:
                w.writerow({"method":    method,
                            "task":      t.get("task", ""),
                            "acc":       round(t.get("acc", 0), 4),
                            "precision": round(t.get("precision", 0), 4),
                            "f1":        round(t.get("f1", 0), 4),
                            "recall":    round(t.get("recall", 0), 4)})
    print(f"[CSV] tasks  → {tpath}")


def _save_comparison_chart(all_rounds, all_tasks, out_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    COLORS  = {"fedavg": "#4472C4", "fl_maldrift": "#ED7D31", "fl_cl_maldrift": "#70AD47"}
    LABELS  = {"fedavg": "FedAvg",  "fl_maldrift": "FL-MalDrift",
               "fl_cl_maldrift": "FL-CL-MalDrift"}
    MARKERS = {"fedavg": "o", "fl_maldrift": "s", "fl_cl_maldrift": "^"}
    LSTYLES = {"fedavg": "-", "fl_maldrift": "--", "fl_cl_maldrift": "-."}

    fig = plt.figure(figsize=(16, 13))
    fig.suptitle("FL-CL-MalDrift vs FL-MalDrift vs FedAvg — CICMalDroid 2020",
                 fontsize=14, fontweight="bold", y=0.99)
    gs = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.32,
                          top=0.94, bottom=0.06, left=0.08, right=0.97)

    # ── Row 0: Training Accuracy per round (span 2 cols) ─────
    ax_tr = fig.add_subplot(gs[0, :])
    boundary_xs, cum_total = [], 0
    for method, rounds in all_rounds.items():
        if not rounds: continue
        xs, ys, prev_task, cum = [], [], -1, 0
        for r in rounds:
            cum += 1
            task = r.get("task", 0)
            if task != prev_task and prev_task >= 0:
                boundary_xs.append(cum - 0.5)
            prev_task = task
            xs.append(cum)
            ys.append(r.get("avg_train_acc", 0))
        cum_total = cum
        ax_tr.plot(xs, ys, color=COLORS[method], linewidth=2.2,
                   linestyle=LSTYLES[method], label=LABELS[method], alpha=0.9)

    boundary_xs = sorted(set(boundary_xs))
    for bx in boundary_xs:
        ax_tr.axvline(x=bx, color="#888888", linestyle=":", linewidth=1.2, alpha=0.7)
    # task labels ở giữa mỗi đoạn
    all_bounds = [0] + [int(b + 0.5) for b in boundary_xs] + [cum_total]
    for i in range(len(all_bounds) - 1):
        mid = (all_bounds[i] + all_bounds[i + 1]) / 2
        ax_tr.text(mid, 0.01, f"Task {i}", ha="center", va="bottom",
                   fontsize=9, color="#555555", fontweight="bold",
                   transform=ax_tr.get_xaxis_transform())
    ax_tr.set_xlabel("Cumulative Round", fontsize=10)
    ax_tr.set_ylabel("Train Accuracy", fontsize=10)
    ax_tr.set_title("Training Accuracy per Round (phân theo Task)", fontsize=11, pad=8)
    ax_tr.legend(loc="lower right", fontsize=9, framealpha=0.8)
    ax_tr.grid(True, alpha=0.3)
    ax_tr.set_ylim(0.4, 1.02)
    ax_tr.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    # ── Rows 1-2: per-task test metrics 2×2 ──────────────────
    metrics  = [("acc", "Accuracy"), ("precision", "Precision"),
                ("f1",  "F1-Score"), ("recall",    "Recall")]
    positions = [(1, 0), (1, 1), (2, 0), (2, 1)]

    for (mkey, mlabel), (row, col) in zip(metrics, positions):
        ax = fig.add_subplot(gs[row, col])
        all_vals = []
        task_ids = []
        for method, tasks in all_tasks.items():
            if not tasks: continue
            task_ids = [t.get("task", i) for i, t in enumerate(tasks)]
            vals     = [t.get(mkey, 0) for t in tasks]
            all_vals.extend(vals)
            ax.plot(task_ids, vals,
                    marker=MARKERS[method], markersize=7, linewidth=2.2,
                    linestyle=LSTYLES[method],
                    color=COLORS[method], label=LABELS[method])
            # giá trị tại mỗi điểm
            for x, v in zip(task_ids, vals):
                ax.annotate(f"{v:.3f}", (x, v),
                            textcoords="offset points", xytext=(0, 7),
                            ha="center", fontsize=7, color=COLORS[method])
        # tự động scale trục Y quanh vùng dữ liệu
        if all_vals:
            lo = max(0.0, min(all_vals) - 0.05)
            hi = min(1.0, max(all_vals) + 0.08)
            ax.set_ylim(lo, hi)
        ax.set_xlabel("Task", fontsize=9)
        ax.set_ylabel(mlabel, fontsize=9)
        ax.set_title(mlabel + " per Task", fontsize=10, pad=6)
        if task_ids:
            ax.set_xticks(task_ids)
        ax.legend(fontsize=7, framealpha=0.8)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    chart_path = os.path.join(out_dir, "comparison_chart.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Chart] comparison → {chart_path}")


def _save_dataset_chart(data_dir, out_dir):
    """Pie chart phân phối class trong dataset CICMalDroid."""
    import pandas as pd
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = os.path.join(data_dir, "cicmaldroid", "features.csv")
    if not os.path.exists(path):
        print("[Chart] dataset_chart: features.csv not found, skipping.")
        return

    df = pd.read_csv(path)
    CLASS_NAMES = {1: "Adware", 2: "Banking", 3: "SMS",
                   4: "Riskware", 5: "Benign"}
    PIE_COLORS  = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]

    if "class_id" in df.columns:
        counts = df["class_id"].value_counts().sort_index()
        labels = [CLASS_NAMES.get(int(k), str(k)) for k in counts.index]
        sizes  = counts.values
    else:
        counts = df["label"].value_counts().sort_index()
        labels = ["Benign (0)", "Malware (1)"]
        sizes  = counts.values

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("CICMalDroid 2020 — Dataset Composition", fontsize=13,
                 fontweight="bold")

    # ── Pie chart ─────────────────────────────────────────────
    ax = axes[0]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels,
        autopct=lambda p: f"{p:.1f}%\n({int(round(p * sum(sizes) / 100)):,})",
        colors=PIE_COLORS[:len(sizes)],
        startangle=140, pctdistance=0.78,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    for at in autotexts:
        at.set_fontsize(8)
    ax.set_title("Class Distribution", fontsize=11, pad=12)

    # ── Bar chart ─────────────────────────────────────────────
    ax2 = axes[1]
    bars = ax2.bar(labels, sizes, color=PIE_COLORS[:len(sizes)],
                   edgecolor="white", linewidth=1)
    for bar, val in zip(bars, sizes):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(sizes) * 0.01,
                 f"{val:,}", ha="center", va="bottom", fontsize=9)
    ax2.set_xlabel("Class", fontsize=10)
    ax2.set_ylabel("Number of Samples", fontsize=10)
    ax2.set_title("Sample Count per Class", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, max(sizes) * 1.15)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "dataset_chart.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Chart] dataset   → {out_path}")


# ── Entry point ───────────────────────────────────────────
def main():
    args = parse()
    dcfg, flcfg, drc_cfg, cl_cfg, priv, srv_cfg = make_configs(args)
    os.makedirs(args.out, exist_ok=True)

    if getattr(args, "kfold", 0) and args.kfold > 1:
        _run_kfold(args)
        return

    if args.runs > 1:
        _run_multi(args)
        return

    if args.compare:
        all_results, all_rounds, all_tasks_data = {}, {}, {}
        for method in ["fedavg", "fl_maldrift", "fl_cl_maldrift"]:
            a = copy.copy(args)
            a.method = method
            d2, f2, dr2, cl2, pr2, sv2 = make_configs(a)
            res, rounds, tasks = run_experiment(
                method, a, d2, f2, dr2, cl2, pr2, sv2, args.out)
            all_results[method]     = res
            all_rounds[method]      = rounds
            all_tasks_data[method]  = tasks

        print("\n" + "="*80)
        print(" COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Method':<16} {'ACC':>7} {'Prec':>7} {'F1':>7} {'Recall':>7} "
              f"{'Forget':>8} {'BWT':>7} {'FWT':>7} {'EscRate':>8} {'RecRate':>8}")
        print("-"*92)
        DISPLAY = {"fedavg":         "FedAvg",
                   "fl_maldrift":    "FL-MalDrift",
                   "fl_cl_maldrift": "FL-CL-MalDrift"}
        for m, r in all_results.items():
            name = DISPLAY.get(m, m)
            print(f"{name:<16} {r.get('ACC',0):>7.4f} {r.get('Precision',0):>7.4f} "
                  f"{r.get('F1',0):>7.4f} {r.get('Recall',0):>7.4f} "
                  f"{r.get('Forgetting',0):>8.4f} {r.get('BWT',0):>7.4f} "
                  f"{r.get('FWT',0):>7.4f} {r.get('EscalationRate',0):>8.4f} "
                  f"{r.get('RecoveryRate',0):>8.4f}")

        with open(os.path.join(args.out, "comparison.json"), "w") as f:
            json.dump(all_results, f, indent=2)

        _save_comparison_csv(all_rounds, all_tasks_data, args.out)
        _save_comparison_chart(all_rounds, all_tasks_data, args.out)
        _save_dataset_chart(dcfg["data_dir"], args.out)
        print(f"\nResults saved to {args.out}")
    else:
        run_experiment(args.method, args, dcfg, flcfg,
                       drc_cfg, cl_cfg, priv, srv_cfg, args.out)


def _run_multi(args):
    """Chạy --runs lần với seed khác nhau.
    Lưu 3 cấp độ: per-run, per-task, và tổng kết mean±std.
    """
    import csv
    BASE_SEED = args.seed if args.seed is not None else DATA["random_seed"]
    TASK_M   = ["acc", "precision", "f1", "recall"]   # metrics từ logger._tasks
    METRICS  = ["ACC", "Precision", "F1", "Recall"]
    METRICS2 = ["Forgetting", "BWT"]
    ALL_M    = METRICS + METRICS2
    METHODS  = ["fedavg", "fl_maldrift", "fl_cl_maldrift"]
    DISPLAY  = {"fedavg":         "FedAvg",
                "fl_maldrift":    "FL-MalDrift",
                "fl_cl_maldrift": "FL-CL-MalDrift"}

    # per_run[method] = [{run, seed, ACC, ...}, ...]
    per_run  = {m: [] for m in METHODS}
    # per_task[method] = [{run, seed, task, acc, prec, f1, recall}, ...]
    per_task = {m: [] for m in METHODS}

    for run_i in range(args.runs):
        seed    = BASE_SEED + run_i * 7
        run_dir = os.path.join(args.out, f"run_{run_i:02d}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"\n{'#'*55}")
        print(f"  RUN {run_i+1}/{args.runs}  |  seed={seed}")
        print(f"{'#'*55}")

        methods = METHODS if args.compare else [args.method]
        for method in methods:
            a        = copy.copy(args)
            a.method = method
            a.seed   = seed
            a.runs   = 1
            d2, f2, dr2, cl2, pr2, sv2 = make_configs(a)
            res, _, tasks = run_experiment(method, a, d2, f2, dr2, cl2, pr2, sv2, run_dir)

            # lưu per-run (trung bình qua tasks — đã tính trong cl_metrics)
            per_run[method].append({
                "run": run_i + 1, "seed": seed,
                **{k: round(res.get(k, 0.0), 4) for k in ALL_M}
            })
            # lưu per-task
            for t in tasks:
                per_task[method].append({
                    "run":  run_i + 1,
                    "fold": run_i + 1,      # alias cho K-fold
                    "seed": seed,
                    "task": t.get("task", ""),
                    **{k: round(t.get(k, 0.0), 4) for k in TASK_M}
                })

    # ─── Console 1: Per-run (mỗi fold) ─────────────────────────
    print(f"\n{'='*75}")
    print(f"  KẾT QUẢ TỪNG LẦN CHẠY / FOLD  ({args.runs} folds)")
    print(f"{'='*75}")
    for m in METHODS:
        rows = per_run[m]
        if not rows: continue
        print(f"\n[{DISPLAY[m]}]")
        hdr = f"  {'Fold':>4}  {'Seed':>6}" + "".join(f"  {k:>9}" for k in ALL_M)
        print(hdr)
        print("  " + "-"*(len(hdr)-2))
        for r in rows:
            vals = "".join(f"  {r[k]:>9.4f}" for k in ALL_M)
            print(f"  {r['run']:>4}  {r['seed']:>6}{vals}")
        # mean ± std của các runs
        print("  " + "-"*(len(hdr)-2))
        mu_row = "".join(
            f"  {np.mean([r[k] for r in rows]):>5.3f}±{np.std([r[k] for r in rows]):.3f}"
            for k in ALL_M)
        print(f"  {'Mean':>4}  {'':>6}{mu_row}")

    # ─── Console 2: Per-task (trung bình qua folds) ────────────
    num_tasks = max((r["task"] for m in METHODS for r in per_task[m]),
                    default=0) + 1
    print(f"\n{'='*75}")
    print(f"  KẾT QUẢ TỪNG TASK (mean ± std qua {args.runs} folds)")
    print(f"{'='*75}")
    for m in METHODS:
        rows = per_task[m]
        if not rows: continue
        print(f"\n[{DISPLAY[m]}]")
        print(f"  {'Task':>4}" + "".join(f"  {k.capitalize():>14}" for k in TASK_M))
        print("  " + "-"*65)
        for t_id in range(num_tasks):
            t_rows = [r for r in rows if r["task"] == t_id]
            if not t_rows: continue
            cells = "".join(
                f"  {np.mean([r[k] for r in t_rows]):>5.3f}±{np.std([r[k] for r in t_rows]):.3f}"
                for k in TASK_M)
            print(f"  {t_id:>4}{cells}")

    # ─── CSV 1: per-run (mỗi lần chạy, trung bình tasks) ───────
    p1 = os.path.join(args.out, "multirun_per_fold.csv")
    with open(p1, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method","run","fold","seed"] + ALL_M)
        w.writeheader()
        for m in METHODS:
            for r in per_run[m]:
                w.writerow({"method": DISPLAY[m], "fold": r["run"], **r})
    print(f"\n[CSV] per-fold   → {p1}")

    # ─── CSV 2: per-task (mỗi task, mỗi fold) ───────────────────
    p2 = os.path.join(args.out, "multirun_per_task.csv")
    with open(p2, "w", newline="") as f:
        fields = ["method","fold","seed","task"] + TASK_M
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in METHODS:
            for r in per_task[m]:
                w.writerow({"method": DISPLAY[m],
                            "fold": r["run"], "seed": r["seed"],
                            "task": r["task"],
                            **{k: r[k] for k in TASK_M}})
    print(f"[CSV] per-task   → {p2}")

    # ─── CSV 3: tổng kết mean±std (theo fold) ───────────────────
    p3 = os.path.join(args.out, "multirun_summary.csv")
    with open(p3, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + [f"{k}_mean" for k in ALL_M]
                               + [f"{k}_std"  for k in ALL_M])
        for m in METHODS:
            rows = per_run[m]
            if not rows: continue
            w.writerow([DISPLAY[m]]
                + [round(float(np.mean([r[k] for r in rows])), 4) for k in ALL_M]
                + [round(float(np.std( [r[k] for r in rows])), 4) for k in ALL_M])
    print(f"[CSV] summary    → {p3}")

    # ─── CSV 4: tổng kết mean±std (theo task) ───────────────────
    p4 = os.path.join(args.out, "multirun_task_summary.csv")
    with open(p4, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method","task"]
                   + [f"{k}_mean" for k in TASK_M]
                   + [f"{k}_std"  for k in TASK_M])
        for m in METHODS:
            for t_id in range(num_tasks):
                t_rows = [r for r in per_task[m] if r["task"] == t_id]
                if not t_rows: continue
                w.writerow([DISPLAY[m], t_id]
                    + [round(float(np.mean([r[k] for r in t_rows])), 4) for k in TASK_M]
                    + [round(float(np.std( [r[k] for r in t_rows])), 4) for k in TASK_M])
    print(f"[CSV] task_summary → {p4}")

def _load_kfold_strata(args, dcfg):
    """Return sample indices and labels used for Stratified K-Fold."""
    import pandas as pd

    ds = args.dataset.lower()
    if ds == "cicmaldroid":
        path = os.path.join(dcfg["data_dir"], "cicmaldroid", "features.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Khong thay CICMalDroid features.csv tai {path}")
        df = pd.read_csv(path)
        strat_col = "class_id" if "class_id" in df.columns else "label"
        strata = df[strat_col].values.astype(int)
        y = df["label"].values.astype(int)
        return np.arange(len(df)), strata, y

    if ds == "drebin":
        path = os.path.join(dcfg["data_dir"], "drebin", "features.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Khong thay Drebin features.csv tai {path}")
        df = pd.read_csv(path)
        y = df.iloc[:, 0].values.astype(int)
        return np.arange(len(df)), y, y

    _, y, _ = DataManager(dcfg, FL)._load()
    return np.arange(len(y)), y.astype(int), y.astype(int)


def _run_kfold(args):
    """Run true Stratified K-Fold, then run continual tasks inside each fold."""
    import csv
    from collections import Counter
    from sklearn.model_selection import StratifiedKFold

    base_seed = args.seed if args.seed is not None else DATA["random_seed"]
    task_metrics = ["acc", "precision", "f1", "recall"]
    all_metrics = ["ACC", "Precision", "F1", "Recall",
                   "Forgetting", "BWT", "FWT",
                   "EscalationRate", "RecoveryRate"]
    default_methods = ["fedavg", "fl_maldrift", "fl_cl_maldrift"]
    display = {"fedavg": "FedAvg",
               "fl_maldrift": "FL-MalDrift",
               "fl_cl_maldrift": "FL-CL-MalDrift"}

    dcfg0, _, _, _, _, _ = make_configs(args)
    indices, strata, binary_y = _load_kfold_strata(args, dcfg0)
    skf = StratifiedKFold(
        n_splits=args.kfold, shuffle=True, random_state=base_seed)

    methods = default_methods if args.compare else [args.method]
    per_fold = {m: [] for m in methods}
    per_task = {m: [] for m in methods}

    for fold_i, (tr_pos, te_pos) in enumerate(skf.split(indices, strata), 1):
        train_idx = indices[tr_pos]
        test_idx = indices[te_pos]
        train_y = binary_y[train_idx]
        test_y = binary_y[test_idx]
        train_dist = dict(sorted(Counter(strata[train_idx].tolist()).items()))
        test_dist = dict(sorted(Counter(strata[test_idx].tolist()).items()))

        fold_dir = os.path.join(args.out, f"fold_{fold_i:02d}")
        os.makedirs(fold_dir, exist_ok=True)
        print(f"\n{'#'*72}")
        print(f"  FOLD {fold_i}/{args.kfold} | clients={args.clients} | "
              f"train={len(train_idx)} test={len(test_idx)}")
        print(f"  Train Ben/Mal = {int((train_y == 0).sum())}/"
              f"{int((train_y == 1).sum())}")
        print(f"  Test  Ben/Mal = {int((test_y == 0).sum())}/"
              f"{int((test_y == 1).sum())}")
        print(f"  Train class dist: {train_dist}")
        print(f"  Test  class dist: {test_dist}")
        print(f"{'#'*72}")

        for method in methods:
            a = copy.copy(args)
            a.method = method
            a.kfold = 0
            a.runs = 1
            a.seed = base_seed

            d2, f2, dr2, cl2, pr2, sv2 = make_configs(a)
            d2["sample_indices"] = train_idx
            d2["outer_test_indices"] = test_idx

            res, _, tasks = run_experiment(
                method, a, d2, f2, dr2, cl2, pr2, sv2, fold_dir)

            row = {"fold": fold_i, "seed": base_seed,
                   "train_n": len(train_idx), "test_n": len(test_idx)}
            row.update({k: round(res.get(k, 0.0), 4) for k in all_metrics})
            per_fold[method].append(row)

            for t in tasks:
                per_task[method].append({
                    "fold": fold_i,
                    "seed": base_seed,
                    "task": t.get("task", ""),
                    **{k: round(t.get(k, 0.0), 4)
                       for k in task_metrics},
                })

    print(f"\n{'='*92}")
    print(f" K-FOLD SUMMARY ({args.kfold} folds)")
    print(f"{'='*92}")
    print(f"{'Method':<16} {'ACC':>12} {'Prec':>12} {'F1':>12} "
          f"{'Recall':>12} {'Forget':>12} {'BWT':>12} {'FWT':>12}")
    print("-"*92)
    for m in methods:
        rows = per_fold[m]
        if not rows:
            continue

        def mean_std(key):
            vals = [r[key] for r in rows]
            return f"{np.mean(vals):.4f}+/-{np.std(vals):.4f}"

        print(f"{display.get(m, m):<16} {mean_std('ACC'):>12} "
              f"{mean_std('Precision'):>12} {mean_std('F1'):>12} "
              f"{mean_std('Recall'):>12} {mean_std('Forgetting'):>12} "
              f"{mean_std('BWT'):>12} {mean_std('FWT'):>12}")

    p1 = os.path.join(args.out, "kfold_per_fold.csv")
    with open(p1, "w", newline="") as f:
        fields = ["method", "fold", "seed", "train_n", "test_n"] + all_metrics
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in methods:
            for r in per_fold[m]:
                w.writerow({"method": display.get(m, m), **r})
    print(f"\n[CSV] per-fold   -> {p1}")

    p2 = os.path.join(args.out, "kfold_per_task.csv")
    with open(p2, "w", newline="") as f:
        fields = ["method", "fold", "seed", "task"] + task_metrics
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in methods:
            for r in per_task[m]:
                w.writerow({"method": display.get(m, m), **r})
    print(f"[CSV] per-task   -> {p2}")

    p3 = os.path.join(args.out, "kfold_summary.csv")
    with open(p3, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + [f"{k}_mean" for k in all_metrics]
                   + [f"{k}_std" for k in all_metrics])
        for m in methods:
            rows = per_fold[m]
            if not rows:
                continue
            w.writerow([display.get(m, m)]
                       + [round(float(np.mean([r[k] for r in rows])), 4)
                          for k in all_metrics]
                       + [round(float(np.std([r[k] for r in rows])), 4)
                          for k in all_metrics])
    print(f"[CSV] summary    -> {p3}")


if __name__ == "__main__":
    main()
