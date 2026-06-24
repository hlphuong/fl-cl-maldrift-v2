"""
config.py — Tất cả hyperparameter tập trung một chỗ.
Thay đổi ở đây, không sửa từng file.
"""

# ── Federated Learning ─────────────────────────────────
FL = dict(
    num_clients        = 10,
    num_rounds         = 25,       # rounds mỗi task
    fraction_fit       = 1.0,
    local_epochs       = 1,
    batch_size         = 64,
    learning_rate      = 0.01,
    momentum           = 0.9,
    aggregation        = "fedavg", # "fedavg" | "fedsgd"
)

# ── Dataset ─────────────────────────────────────────────
DATA = dict(
    dataset            = "cicmaldroid",  # "drebin" | "cicmaldroid" | "synthetic"
    data_dir           = "data/",
    num_tasks          = 5,
    non_iid_alpha      = 0.5,
    test_split         = 0.2,
    val_split          = 0.1,
    num_features       = 200,       # drebin=1000, cicmaldroid=200, synthetic=200
    random_seed        = 42,
    task_strategy      = "category", # "category" | "temporal"
    partition_strategy = "dirichlet", # "dirichlet" | "category"
)

# ── MLP ─────────────────────────────────────────────────
MODEL = dict(
    hidden_dims        = [256, 128, 64],
    dropout            = 0.3,
)

# ── Drift Detection ──────────────────────────────────────
DRIFT = dict(
    detector           = "hddm_w",
    drift_confidence   = 0.001,
    warning_confidence = 0.005,
    local_threshold    = 0.5,
)

# ── DRC ─────────────────────────────────────────────────
# Theo Algorithm 3 (slide 7): K1=3, K2=8, R=3, δ=0.05.
# Stage 1 (Replay) : count ≤ K1
# Stage 2 (EWC)    : K1 < count ≤ K2
# Stage 3 (Escal.) : count > K2  → withheld + Recovery Monitor
DRC = dict(
    K1        = 3,      # số round dùng Replay trước khi leo thang
    K2        = 8,      # số round dùng EWC trước khi Escalation
    R         = 3,      # số round ổn định liên tiếp để tái gia nhập
    delta     = 0.05,   # re-entry margin: τ_re = τ_t + δ
    ema_alpha = 0.3,    # hệ số EMA cho Recovery Monitor
)

# ── Continual Learning ───────────────────────────────────
# B = 200 (Replay buffer), λ_EWC = 0.4 theo slide 7.
CL = dict(
    replay_buffer_size = 200,
    replay_batch_size  = 32,
    ewc_lambda         = 0.4,
    ewc_fisher_samples = 50,
)

# ── Differential Privacy ─────────────────────────────────
# σ = 1.2, δ = 1e-5 theo slide 7 (ε ≈ 1.0). Mặc định TẮT để so sánh
# 3 method công bằng; bật bằng cờ --dp (ablation E5 Privacy-utility).
PRIVACY = dict(
    enabled            = False,
    noise_multiplier   = 1.2,
    max_grad_norm      = 1.0,
    delta              = 1e-5,
)

# ── Server EWMA ──────────────────────────────────────────
# tau_min=0.5: chặn ngưỡng τ sụp về 0 (khiến reject hàng loạt client đang
#   thích nghi với drift hợp lệ → mô hình sụp đổ). Với τ≥0.5, ngưỡng reject
#   τ·1.5≥0.75 chỉ loại client drift cực mạnh, client "warning" vẫn được giữ
#   và down-weight (đúng thiết kế stable/warning/reject ở slide 5).
# min_participation: sàn tỉ lệ client luôn được tổng hợp mỗi vòng — nếu lọc
#   làm rớt quá sàn, giữ lại các client có drift thấp nhất (tránh collapse).
SERVER = dict(
    warmup_rounds         = 3,
    ewma_alpha            = 0.8,
    k_sigma               = 1.5,
    target_participation  = 0.7,
    eta                   = 0.05,
    tau_min               = 0.5,
    tau_max               = 1.0,
    min_participation     = 0.5,
    use_drift_filter      = True,
)

# ── Dataset-specific presets ──────────────────────────────
DATASET_PRESETS = {
    "cicmaldroid": {"num_features": 200, "num_tasks": 5},
    "drebin":      {"num_features": 1000, "num_tasks": 5},
    "synthetic":   {"num_features": 200,  "num_tasks": 5},
}
