"""
prepare_cicmaldroid.py
======================
Script hoàn chỉnh để:
  1. Download CICMalDroid 2020 từ Kaggle (kagglehub)
  2. Tự động khám phá cấu trúc file (tên file, cột, label encoding)
  3. Tiền xử lý: chọn features, encode label, cân bằng class
  4. Lưu ra data/cicmaldroid/features.csv đúng format cho pipeline

Chạy:
    pip install kagglehub pandas scikit-learn
    python prepare_cicmaldroid.py

Sau khi chạy xong, chạy training:
    python main.py --dataset cicmaldroid --tasks 5 --rounds 25
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.utils import resample

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── Cấu hình ─────────────────────────────────────────────
OUT_DIR       = "data/cicmaldroid"
OUT_FILE      = os.path.join(OUT_DIR, "features.csv")
NUM_FEATURES  = 200        # số features giữ lại
MAX_SAMPLES   = 12000      # toi da mau sau khi lay mau benign/malware
KAGGLE_HANDLE = "hasanccr92/cicmaldroid-2020"

PREFERRED_FILES = [
    "feature_vectors_syscallsbinders_frequency_5_Cat.csv",
    "feature_vectors_syscalls_frequency_5_Cat.csv",
]
DEFAULT_LOCAL_DIRS = [
    Path("..") / "CSV",
    Path("CSV"),
]

# Các tên cột label phổ biến trong CICMalDroid
POSSIBLE_LABEL_COLS = [
    "Label", "label", "class", "Class", "Category", "category",
    "type", "Type", "malware", "Malware", "target", "Target"
]

# Các giá trị được coi là BENIGN
BENIGN_VALUES = [
    "benign", "Benign", "BENIGN", "0", 0, "goodware", "Goodware",
    "normal", "Normal", "legitimate", "Legitimate"
]


# ══════════════════════════════════════════════════════════
# Step 1: Download
# ══════════════════════════════════════════════════════════
def download_dataset() -> str:
    """Download dataset từ Kaggle và trả về path."""
    print("\n" + "="*55)
    print("STEP 1: Download CICMalDroid 2020 từ Kaggle")
    print("="*55)
    try:
        import kagglehub
        path = kagglehub.dataset_download(KAGGLE_HANDLE)
        print(f"✓ Downloaded to: {path}")
        return path
    except Exception as e:
        print(f"✗ Download failed: {e}")
        print("\nGợi ý:")
        print("  1. Cài: pip install kagglehub")
        print("  2. Đăng nhập Kaggle: kaggle.json trong ~/.kaggle/")
        print("  3. Hoặc set env: KAGGLE_USERNAME, KAGGLE_KEY")
        sys.exit(1)


# ══════════════════════════════════════════════════════════
# Step 2: Khám phá cấu trúc
# ══════════════════════════════════════════════════════════
def explore_dataset(dataset_path: str) -> list:
    """Liệt kê tất cả file CSV trong dataset."""
    print("\n" + "="*55)
    print("STEP 2: Khám phá cấu trúc dataset")
    print("="*55)

    csv_files = []
    for root, dirs, files in os.walk(dataset_path):
        for f in files:
            if f.endswith(".csv"):
                full = os.path.join(root, f)
                size = os.path.getsize(full) / 1024 / 1024
                csv_files.append((full, size))
                print(f"  📄 {f}  ({size:.1f} MB)")

    if not csv_files:
        print("✗ Không tìm thấy file CSV!")
        print(f"  Nội dung thư mục {dataset_path}:")
        for item in os.listdir(dataset_path):
            print(f"    {item}")
        sys.exit(1)

    print(f"\n✓ Tìm thấy {len(csv_files)} file CSV")
    return csv_files


# ══════════════════════════════════════════════════════════
# Step 3: Đọc và phân tích từng file
# ══════════════════════════════════════════════════════════
def analyze_file(filepath: str) -> dict:
    """
    Đọc file CSV và phân tích:
    - Tìm cột label
    - Phân phối class
    - Kiểu dữ liệu features
    """
    print(f"\n  Phân tích: {os.path.basename(filepath)}")
    try:
        df = pd.read_csv(filepath, nrows=5000)  # đọc mẫu để phân tích
    except Exception as e:
        print(f"    ✗ Không đọc được: {e}")
        return None

    print(f"    Shape: {df.shape}")
    print(f"    Columns: {list(df.columns[:8])}{'...' if len(df.columns)>8 else ''}")

    # Tìm cột label
    label_col = None
    for col in POSSIBLE_LABEL_COLS:
        if col in df.columns:
            label_col = col
            break

    # Nếu không tìm được, thử cột cuối cùng
    if label_col is None:
        last_col = df.columns[-1]
        unique_vals = df[last_col].nunique()
        if unique_vals <= 10:  # cột cuối có ít giá trị → có thể là label
            label_col = last_col
            print(f"    ⚠ Dùng cột cuối '{label_col}' làm label")

    if label_col is None:
        print("    ✗ Không tìm thấy cột label")
        return None

    # Phân phối class
    dist = df[label_col].value_counts()
    print(f"    Label col: '{label_col}'")
    print(f"    Distribution: {dict(dist.head(6))}")

    # Đếm numeric features
    feature_cols = [c for c in df.columns if c != label_col]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    print(f"    Numeric features: {len(numeric_cols)}/{len(feature_cols)}")

    return {
        "filepath":    filepath,
        "label_col":   label_col,
        "n_rows":      len(df),
        "n_features":  len(numeric_cols),
        "dist":        dict(dist),
        "numeric_cols": numeric_cols,
    }


# ══════════════════════════════════════════════════════════
# Step 4: Chọn file tốt nhất
# ══════════════════════════════════════════════════════════
def select_best_file(csv_files: list) -> dict:
    """Chọn file CSV phù hợp nhất cho training."""
    print("\n" + "="*55)
    print("STEP 3: Phân tích các file CSV")
    print("="*55)

    by_name = {os.path.basename(p).lower(): (p, size) for p, size in csv_files}
    for preferred in PREFERRED_FILES:
        hit = by_name.get(preferred.lower())
        if not hit:
            continue
        info = analyze_file(hit[0])
        if info and info["n_features"] >= NUM_FEATURES:
            info["size_mb"] = hit[1]
            print(f"\n✓ Chọn file ưu tiên: {os.path.basename(info['filepath'])}")
            print(f"  Features: {info['n_features']}")
            print(f"  Label col: {info['label_col']}")
            return info

    candidates = []
    for filepath, size in csv_files:
        if os.path.basename(filepath).lower() in {
            name.lower() for name in PREFERRED_FILES
        }:
            continue
        info = analyze_file(filepath)
        if info and info["n_features"] >= 10:
            info["size_mb"] = size
            candidates.append(info)

    if not candidates:
        print("✗ Không có file nào phù hợp!")
        sys.exit(1)

    # Ưu tiên file có nhiều features nhất (nhưng không quá lớn)
    # Với CICMalDroid: file 470 features thường là tốt nhất
    best = sorted(candidates,
                  key=lambda x: (x["n_features"], -x["size_mb"]),
                  reverse=True)[0]

    print(f"\n✓ Chọn file: {os.path.basename(best['filepath'])}")
    print(f"  Features: {best['n_features']}")
    print(f"  Label col: {best['label_col']}")
    return best


# ══════════════════════════════════════════════════════════
# Step 5: Load full dataset
# ══════════════════════════════════════════════════════════
def load_full(info: dict) -> pd.DataFrame:
    """Load toàn bộ file đã chọn."""
    print("\n" + "="*55)
    print("STEP 4: Load toàn bộ dữ liệu")
    print("="*55)
    print(f"  Đang đọc {os.path.basename(info['filepath'])}...")
    df = pd.read_csv(info["filepath"])
    print(f"  ✓ Shape: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════
# Step 6: Tiền xử lý
# ══════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame, info: dict,
               malware_ratio: float = 1.0) -> tuple:
    """
    Tiền xử lý đầy đủ:
    1. Encode label → 0 (benign) / 1 (malware), giữ class_id gốc (1-5)
    2. Chọn numeric features
    3. Xử lý NaN, Inf
    4. Feature selection (top K bằng mutual info)
    5. Lay mau theo ty le malware:benign mong muon
    Returns: (X, y_binary, y_cls) numpy arrays
    """
    print("\n" + "="*55)
    print("STEP 5: Tiền xử lý")
    print("="*55)

    label_col  = info["label_col"]
    num_cols   = info["numeric_cols"]

    # ── 5a. Encode label ─────────────────────────────────
    print("  5a. Encode label...")
    raw_labels = df[label_col].astype(str).str.strip()
    unique_labels = raw_labels.unique()
    print(f"      Unique labels: {sorted(unique_labels)}")

    y_cls_raw = None  # class id goc (1-5), dung de partition category

    try:
        nums = raw_labels.astype(float)
        unique_nums = sorted(nums.unique())
        print(f"      Numeric values: {unique_nums}")

        if set(unique_nums) == {0.0, 1.0} or set(unique_nums) == {0, 1}:
            y = nums.astype(int).values
            y_cls_raw = y.copy()  # binary da la class
            print("      → Binary label (0=benign, 1=malware)")

        elif max(unique_nums) <= 5 and min(unique_nums) >= 1:
            # CICMalDroid: 1=Adware, 2=Banking, 3=SMS, 4=Riskware, 5=Benign
            from collections import Counter
            counts = Counter(nums.astype(int))
            print(f"      Class counts: {dict(counts)}")
            benign_cls = max(unique_nums)
            y = (nums != benign_cls).astype(int).values
            y_cls_raw = nums.astype(int).values  # giu class goc 1-5
            print(f"      → Class {int(benign_cls)} = Benign(0), 1-4 = Malware(1)")

        else:
            from collections import Counter
            counts = Counter(nums.astype(int))
            benign_cls = max(counts, key=counts.get)
            y = (nums != benign_cls).astype(int).values
            y_cls_raw = nums.astype(int).values
            print(f"      → Class {benign_cls} (nhieu nhat) = Benign, con lai = Malware")

    except ValueError:
        benign_strs = ["benign", "goodware", "normal", "legitimate", "0"]
        y = (~raw_labels.str.lower().isin(benign_strs)).astype(int).values
        y_cls_raw = y.copy()
        print("      → Text label encoding")

    y = pd.Series(y)
    print(f"      Benign: {(y==0).sum()}, Malware: {(y==1).sum()}")
    
    # ── 5b. Select numeric features ──────────────────────
    print("  5b. Chọn numeric features...")
    X = df[num_cols].copy()

    # Xử lý NaN và Inf
    X = X.replace([np.inf, -np.inf], np.nan)
    nan_ratio = X.isna().mean()
    good_cols = nan_ratio[nan_ratio < 0.5].index.tolist()
    X = X[good_cols]
    X = X.fillna(X.median())
    print(f"      Features sau clean: {X.shape[1]}")

    # Loại cột constant
    std = X.std()
    X = X[std[std > 1e-6].index]
    print(f"      Features sau loại constant: {X.shape[1]}")

    # ── 5c. Feature selection ─────────────────────────────
    n_select = min(NUM_FEATURES, X.shape[1])
    if X.shape[1] > NUM_FEATURES:
        print(f"  5c. Feature selection: {X.shape[1]} → {n_select}...")
        selector = SelectKBest(mutual_info_classif, k=n_select)
        X_arr = selector.fit_transform(X.values, y.values)
        selected_cols = X.columns[selector.get_support()].tolist()
        X = pd.DataFrame(X_arr, columns=selected_cols)
    else:
        n_select = X.shape[1]
        print(f"  5c. Giữ tất cả {n_select} features")

    # ── 5d. Cân bằng class ────────────────────────────────
    print("  5d. Lay mau class...")
    df_proc = X.copy()
    df_proc["__label__"] = y.values
    # thread class_id goc de bao ton sau khi resample
    df_proc["__class__"] = pd.Series(y_cls_raw, index=df_proc.index).values

    benign_df  = df_proc[df_proc["__label__"] == 0]
    malware_df = df_proc[df_proc["__label__"] == 1]

    print(f"      Benign raw: {len(benign_df)}, Malware raw: {len(malware_df)}")

    drop_cols = ["__label__", "__class__"]

    if len(benign_df) == 0 or len(malware_df) == 0:
        print("      WARNING: mot class rong — kiem tra lai label encoding!")
        print(f"      Unique labels: {df_proc['__label__'].unique()}")
        y_final   = df_proc["__label__"].values
        y_cls_final = df_proc["__class__"].values
        X_final   = df_proc.drop(columns=drop_cols).values
    else:
        malware_ratio = max(float(malware_ratio), 1e-6)
        n_benign = min(len(benign_df), int(MAX_SAMPLES / (1.0 + malware_ratio)))
        n_malware = min(len(malware_df), int(round(n_benign * malware_ratio)))

        # If malware is the limiting class, shrink benign to preserve the ratio.
        if n_malware < int(round(n_benign * malware_ratio)):
            n_benign = min(len(benign_df), int(n_malware / malware_ratio))

        n_benign = max(n_benign, 1)
        n_malware = max(n_malware, 1)

        print(f"      Target: {n_benign} benign + {n_malware} malware "
              f"(malware:benign={malware_ratio:g}:1)")

        benign_sample  = resample(benign_df,  n_samples=n_benign,
                                  random_state=42, replace=len(benign_df) < n_benign)
        malware_sample = resample(malware_df, n_samples=n_malware,
                                  random_state=42, replace=len(malware_df) < n_malware)

        balanced = pd.concat([benign_sample, malware_sample])
        balanced = balanced.sort_values("__class__").reset_index(drop=True)

        y_final     = balanced["__label__"].values
        y_cls_final = balanced["__class__"].values
        X_final     = balanced.drop(columns=drop_cols).values

    print(f"      Final: {X_final.shape[0]} samples, {X_final.shape[1]} features")
    if y_cls_final is not None:
        from collections import Counter
        print(f"      Class dist: {dict(sorted(Counter(y_cls_final.tolist()).items()))}")
    return X_final.astype(np.float32), y_final.astype(np.int64), y_cls_final.astype(np.int64)

# ══════════════════════════════════════════════════════════
# Step 7: Lưu ra CSV chuẩn
# ══════════════════════════════════════════════════════════
def save_features(X: np.ndarray, y: np.ndarray, y_cls: np.ndarray = None):
    """
    Luu ra CSV:
    - Co class_id: label | class_id | f0..fN
    - Khong co:    label | f0..fN
    """
    print("\n" + "="*55)
    print("STEP 6: Luu features.csv")
    print("="*55)

    os.makedirs(OUT_DIR, exist_ok=True)

    n_features = X.shape[1]
    feat_names = [f"f{i}" for i in range(n_features)]

    if y_cls is not None:
        # Format moi: label | class_id | f0..fN
        col_names = ["label", "class_id"] + feat_names
        data      = np.column_stack([y, y_cls, X])
    else:
        col_names = ["label"] + feat_names
        data      = np.column_stack([y, X])

    df_out = pd.DataFrame(data, columns=col_names)
    df_out["label"] = df_out["label"].astype(int)
    if y_cls is not None:
        df_out["class_id"] = df_out["class_id"].astype(int)
    df_out.to_csv(OUT_FILE, index=False)

    size_mb = os.path.getsize(OUT_FILE) / 1024 / 1024
    print(f"  Saved: {OUT_FILE}")
    print(f"  Shape: {df_out.shape}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Label distribution: {dict(df_out['label'].value_counts())}")
    if y_cls is not None:
        print(f"  Class dist: {dict(df_out['class_id'].value_counts().sort_index())}")


# ══════════════════════════════════════════════════════════
# Step 8: Verify
# ══════════════════════════════════════════════════════════
def verify():
    """Kiem tra file dau ra va test load vao pipeline."""
    print("\n" + "="*55)
    print("STEP 7: Verification")
    print("="*55)

    import pandas as pd
    df = pd.read_csv(OUT_FILE)
    y  = df["label"].values

    has_cls = "class_id" in df.columns
    feat_cols = [c for c in df.columns if c not in ("label", "class_id")]
    X = df[feat_cols].values

    print(f"  File: {OUT_FILE}")
    print(f"  Samples: {len(y)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Has class_id: {has_cls}")
    print(f"  Labels: {np.unique(y)} (0=benign, 1=malware)")
    print(f"  Class balance: {(y==0).sum()} benign / {(y==1).sum()} malware")
    if has_cls:
        from collections import Counter
        print(f"  Class dist: {dict(sorted(Counter(df['class_id'].tolist()).items()))}")
    print(f"  X range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"  NaN count: {np.isnan(X).sum()}")

    if np.isnan(X).sum() == 0 and len(np.unique(y)) == 2:
        print("\n  Dataset OK! San sang training.")
        print("    python main.py --compare --dataset cicmaldroid --tasks 5 --rounds 25")
    else:
        print("\n  WARNING: Dataset co van de, kiem tra lai!")


# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", type=str, default=None,
                        help="Đường dẫn tới file hoặc thư mục CSV local (bỏ qua download Kaggle)")
    parser.add_argument("--force", action="store_true",
                        help="Ghi đè features.csv nếu đã tồn tại")
    parser.add_argument("--malware_ratio", type=float, default=1.0,
                        help="Ty le malware:benign sau khi lay mau, vi du 2.0 = 2:1")
    args = parser.parse_args()

    print("CICMalDroid 2020 — Data Preparation Pipeline")
    print("=" * 55)

    # Kiểm tra xem đã có file chưa
    if os.path.exists(OUT_FILE) and not args.force:
        print(f"\n⚠ File đã tồn tại: {OUT_FILE}")
        ans = input("  Tạo lại không? [y/N]: ").strip().lower()
        if ans != "y":
            print("  Dùng file hiện có.")
            verify()
            return

    # Pipeline
    csv_files = None
    if args.local:
        local_path = os.path.abspath(args.local)
        if not os.path.exists(local_path):
            print(f"✗ Không tìm thấy file: {local_path}")
            sys.exit(1)
        if os.path.isfile(local_path):
            size = os.path.getsize(local_path) / 1024 / 1024
            csv_files = [(local_path, size)]
            print(f"\n[LOCAL] Dùng file: {local_path}")
        else:
            print(f"\n[LOCAL] Dùng thư mục: {local_path}")
            csv_files = explore_dataset(local_path)
    else:
        for local_dir in DEFAULT_LOCAL_DIRS:
            if local_dir.exists():
                print(f"\n[LOCAL] Phát hiện thư mục CSV: {local_dir}")
                csv_files = explore_dataset(str(local_dir))
                break
        if csv_files is None:
            dataset_path = download_dataset()
            csv_files = explore_dataset(dataset_path)

    best_info = select_best_file(csv_files)
    df        = load_full(best_info)
    X, y, y_cls = preprocess(df, best_info,
                             malware_ratio=args.malware_ratio)
    save_features(X, y, y_cls)
    verify()


if __name__ == "__main__":
    main()
