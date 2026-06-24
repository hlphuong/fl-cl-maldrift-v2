"""
data/dataset.py
Temporal split -> T domain tasks, Non-IID partition.

Hai chien luoc partition:
  "category" : Category-aware temporal split + client specialization
               (khi co class_id tu prepare_cicmaldroid.py)
  "dirichlet": Temporal split theo thu tu thoi gian + Dirichlet
               (fallback hoac khi khong co class_id)
"""
import os
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


# ── Dataset wrapper ─────────────────────────────────────
class MalwareDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Synthetic data ───────────────────────────────────────
def make_synthetic(n: int, d: int, seed: int = 42,
                   task_id: int = 0):
    rng = np.random.default_rng(seed + task_id * 17)
    shift = task_id * 0.4
    X0 = rng.normal( shift,     1.0, (n // 2, d)).astype(np.float32)
    X1 = rng.normal(-shift + 1, 1.0, (n // 2, d)).astype(np.float32)
    X = np.vstack([X0, X1])
    y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int64)
    idx = rng.permutation(n)
    return X[idx], y[idx], None


# ── CSV loaders ─────────────────────────────────────────
def load_drebin(data_dir: str, num_features: int, seed: int):
    path = os.path.join(data_dir, "drebin", "features.csv")
    if not os.path.exists(path):
        print("[WARN] Drebin not found -> synthetic fallback")
        return make_synthetic(3000, num_features, seed)
    import pandas as pd
    df = pd.read_csv(path)
    y = df.iloc[:, 0].values.astype(np.int64)
    X = df.iloc[:, 1:num_features + 1].values.astype(np.float32)
    return X, y, None


def load_cicmaldroid(data_dir: str, num_features: int, seed: int):
    path = os.path.join(data_dir, "cicmaldroid", "features.csv")
    if not os.path.exists(path):
        print("[WARN] CICMalDroid not found.")
        print("[WARN] Chay: python prepare_cicmaldroid.py --local <file.csv>")
        return make_synthetic(6000, num_features, seed)
    import pandas as pd
    df = pd.read_csv(path)
    y = df["label"].values.astype(np.int64)

    if "class_id" in df.columns:
        y_cls = df["class_id"].values.astype(np.int64)
        feat_cols = [c for c in df.columns if c not in ("label", "class_id")]
        X = df[feat_cols].iloc[:, :num_features].values.astype(np.float32)
    else:
        y_cls = None
        n_cols = min(num_features, df.shape[1] - 1)
        X = df.iloc[:, 1:n_cols + 1].values.astype(np.float32)

    print(f"[CICMalDroid] {len(y)} samples, {X.shape[1]} features")
    print(f"[CICMalDroid] Benign={int((y==0).sum())}, Malware={int((y==1).sum())}")
    if y_cls is not None:
        from collections import Counter
        dist = dict(sorted(Counter(y_cls.tolist()).items()))
        print(f"[CICMalDroid] Class dist: {dist}  (1=Adware 2=Banking 3=SMS 4=Riskware 5=Benign)")
    return X, y, y_cls


# ── Standard Temporal Split (fallback) ──────────────────
def temporal_split(X: np.ndarray, y: np.ndarray,
                   num_tasks: int, test_split: float,
                   val_split: float, seed: int) -> List[Dict]:
    n = len(y)
    chunk = n // num_tasks
    tasks = []
    for t in range(num_tasks):
        s = t * chunk
        e = (t + 1) * chunk if t < num_tasks - 1 else n
        Xt, yt = X[s:e], y[s:e]
        Xtr, Xte, ytr, yte = train_test_split(
            Xt, yt, test_size=test_split,
            random_state=seed + t,
            stratify=yt if len(np.unique(yt)) > 1 else None)
        val_ratio = val_split / (1 - test_split)
        Xtr, Xval, ytr, yval = train_test_split(
            Xtr, ytr, test_size=val_ratio,
            random_state=seed + t,
            stratify=ytr if len(np.unique(ytr)) > 1 else None)
        tasks.append({"train":     (Xtr, ytr),
                      "val":       (Xval, yval),
                      "test":      (Xte, yte),
                      "train_cls": None})
    return tasks


def _safe_stratify(labels: np.ndarray):
    vals, counts = np.unique(labels, return_counts=True)
    return labels if len(vals) > 1 and counts.min() >= 2 else None


def _task_split_dict(X: np.ndarray, y: np.ndarray, y_cls: np.ndarray,
                     idx_t: np.ndarray, test_split: float, val_split: float,
                     seed: int, task_id: int) -> Dict:
    Xt, yt, yt_cls = X[idx_t], y[idx_t], y_cls[idx_t]

    Xtr, Xte, ytr, yte, ycls_tr, _ = train_test_split(
        Xt, yt, yt_cls,
        test_size=test_split,
        random_state=seed + task_id,
        stratify=_safe_stratify(yt))

    val_ratio = val_split / (1 - test_split)
    Xtr, Xval, ytr, yval, ycls_tr, _ = train_test_split(
        Xtr, ytr, ycls_tr,
        test_size=val_ratio,
        random_state=seed + task_id,
        stratify=_safe_stratify(ytr))

    return {"train":     (Xtr, ytr),
            "val":       (Xval, yval),
            "test":      (Xte, yte),
            "train_cls": ycls_tr}


# ── Category-Aware Temporal Split ────────────────────────
def category_temporal_split(X: np.ndarray, y: np.ndarray,
                             y_cls: np.ndarray,
                             num_tasks: int, test_split: float,
                             val_split: float, seed: int) -> List[Dict]:
    """
    Tao temporal drift bang cach thay doi malware category dominance.

    Moi malware class phan bo:
      90% -> home task (task co index tuong ung voi class do)
      10% -> chia deu cho cac task con lai

    Task cuoi (neu num_tasks > so malware class): chi nhan phan "rest"
    -> phan phoi balanced, tao drift tu task truoc do.

    CICMalDroid:
      Task 0 dom: Adware (class 1)
      Task 1 dom: Banking (class 2)
      Task 2 dom: SMS (class 3)
      Task 3 dom: Riskware (class 4)
      Task 4    : Mixed (phan con lai cua tat ca class)
    """
    rng = np.random.default_rng(seed)

    mal_mask = (y == 1)
    ben_mask = (y == 0)
    mal_cls_vals = sorted(np.unique(y_cls[mal_mask]).tolist())
    n_mal = len(mal_cls_vals)

    DOM_FRAC = 0.90

    # Shuffle va chia index theo class
    idx_mal = {}
    for cv in mal_cls_vals:
        arr = np.where(mal_mask & (y_cls == cv))[0]
        rng.shuffle(arr)
        idx_mal[cv] = arr

    ben_idx = np.where(ben_mask)[0]
    rng.shuffle(ben_idx)

    # Gan malware samples vao tasks
    task_mal = defaultdict(list)
    for i_cls, cv in enumerate(mal_cls_vals):
        home = i_cls % num_tasks
        arr  = idx_mal[cv]
        n    = len(arr)
        n_home = int(DOM_FRAC * n)
        task_mal[home].extend(arr[:n_home].tolist())

        other_tasks = [t for t in range(num_tasks) if t != home]
        if other_tasks:
            rest_chunks = np.array_split(arr[n_home:], len(other_tasks))
            for t, chunk in zip(other_tasks, rest_chunks):
                task_mal[t].extend(chunk.tolist())

    # Keep each domain task roughly binary-balanced. If benign samples are split
    # evenly, the final "Mixed" task becomes almost all benign because it only
    # receives the 10% malware remainder from each family.
    mal_counts = np.array([len(task_mal[t]) for t in range(num_tasks)], dtype=float)
    if mal_counts.sum() > 0:
        ben_counts = np.floor(mal_counts / mal_counts.sum() * len(ben_idx)).astype(int)
        for t in np.argsort(-(mal_counts - ben_counts))[:len(ben_idx) - ben_counts.sum()]:
            ben_counts[t] += 1
    else:
        ben_counts = np.array([len(x) for x in np.array_split(ben_idx, num_tasks)])

    ben_chunks = []
    cur = 0
    for cnt in ben_counts:
        ben_chunks.append(ben_idx[cur:cur + int(cnt)])
        cur += int(cnt)
    if cur < len(ben_idx):
        ben_chunks[-1] = np.concatenate([ben_chunks[-1], ben_idx[cur:]])

    tasks = []
    for t in range(num_tasks):
        idx_t = np.array(task_mal[t] + ben_chunks[t].tolist())
        rng.shuffle(idx_t)

        Xt, yt, yt_cls = X[idx_t], y[idx_t], y_cls[idx_t]

        uniq = np.unique(yt)
        strat = yt if len(uniq) > 1 else None
        Xtr, Xte, ytr, yte, ycls_tr, _ = train_test_split(
            Xt, yt, yt_cls,
            test_size=test_split, random_state=seed + t, stratify=strat)

        val_ratio = val_split / (1 - test_split)
        uniq_tr = np.unique(ytr)
        strat_tr = ytr if len(uniq_tr) > 1 else None
        Xtr, Xval, ytr, yval, ycls_tr, _ = train_test_split(
            Xtr, ytr, ycls_tr,
            test_size=val_ratio, random_state=seed + t, stratify=strat_tr)

        cls_names = {1: "Adware", 2: "Banking", 3: "SMS", 4: "Riskware", 5: "Benign"}
        dom_cls = mal_cls_vals[t] if t < n_mal else None
        dom_name = cls_names.get(dom_cls, dom_cls) if dom_cls is not None else "Mixed"
        mal_cnt = int((yt == 1).sum())
        ben_cnt = int((yt == 0).sum())
        print(f"  Task {t} (dom={dom_name}): "
              f"{len(idx_t)} samples | Mal={mal_cnt} Ben={ben_cnt}")

        tasks.append({"train":     (Xtr, ytr),
                      "val":       (Xval, yval),
                      "test":      (Xte, yte),
                      "train_cls": ycls_tr})
    return tasks


def category_strict_split(X: np.ndarray, y: np.ndarray,
                          y_cls: np.ndarray,
                          num_tasks: int, test_split: float,
                          val_split: float, seed: int,
                          revisit_old: bool = False) -> List[Dict]:
    """
    Forgetting stress split.

    - Task 0..3: each task is dominated by exactly one malware family.
    - If num_tasks > #families:
        * category_strict: final task is a held-out mixed slice.
        * category_revisit: final task revisits held-out old families.
    This makes old-family forgetting easier to observe than the softer
    category-aware split.
    """
    rng = np.random.default_rng(seed)
    cls_names = {1: "Adware", 2: "Banking", 3: "SMS", 4: "Riskware", 5: "Benign"}

    mal_mask = (y == 1)
    ben_mask = (y == 0)
    mal_cls_vals = sorted(np.unique(y_cls[mal_mask]).tolist())
    n_mal = len(mal_cls_vals)
    if n_mal == 0:
        return temporal_split(X, y, num_tasks, test_split, val_split, seed)

    task_mal = defaultdict(list)
    has_extra_task = num_tasks > n_mal
    extra_task = num_tasks - 1 if has_extra_task else None
    holdout_frac = 0.25 if revisit_old else 0.15
    revisit_classes = set(mal_cls_vals[:2])

    for i_cls, cv in enumerate(mal_cls_vals):
        arr = np.where(mal_mask & (y_cls == cv))[0]
        rng.shuffle(arr)

        reserve = 0
        if has_extra_task:
            if revisit_old:
                reserve = int(holdout_frac * len(arr)) if cv in revisit_classes else 0
            else:
                reserve = int(holdout_frac * len(arr))
        reserve = min(reserve, max(0, len(arr) - 2))

        home_arr = arr[:-reserve] if reserve else arr
        reserve_arr = arr[-reserve:] if reserve else np.array([], dtype=int)

        home_task = i_cls if i_cls < min(n_mal, num_tasks) else i_cls % num_tasks
        if home_task == extra_task and has_extra_task:
            home_task = max(0, extra_task - 1)
        task_mal[home_task].extend(home_arr.tolist())
        if reserve and extra_task is not None:
            task_mal[extra_task].extend(reserve_arr.tolist())

    ben_idx = np.where(ben_mask)[0]
    rng.shuffle(ben_idx)
    mal_counts = np.array([len(task_mal[t]) for t in range(num_tasks)], dtype=float)
    if mal_counts.sum() > 0:
        ben_counts = np.floor(mal_counts / mal_counts.sum() * len(ben_idx)).astype(int)
        for t in np.argsort(-(mal_counts - ben_counts))[:len(ben_idx) - ben_counts.sum()]:
            ben_counts[t] += 1
    else:
        ben_counts = np.array([len(x) for x in np.array_split(ben_idx, num_tasks)])

    ben_chunks, cur = [], 0
    for cnt in ben_counts:
        ben_chunks.append(ben_idx[cur:cur + int(cnt)])
        cur += int(cnt)
    if cur < len(ben_idx):
        ben_chunks[-1] = np.concatenate([ben_chunks[-1], ben_idx[cur:]])

    tasks = []
    for t in range(num_tasks):
        idx_t = np.array(task_mal[t] + ben_chunks[t].tolist())
        if len(idx_t) == 0:
            continue
        rng.shuffle(idx_t)

        if revisit_old and has_extra_task and t == extra_task:
            dom_name = "RevisitOld"
        elif has_extra_task and t == extra_task:
            dom_name = "MixedHeldout"
        else:
            dom_cls = mal_cls_vals[t] if t < n_mal else None
            dom_name = cls_names.get(dom_cls, dom_cls) if dom_cls is not None else "Mixed"
        yt = y[idx_t]
        print(f"  Task {t} (dom={dom_name}): "
              f"{len(idx_t)} samples | Mal={int((yt == 1).sum())} "
              f"Ben={int((yt == 0).sum())}")

        tasks.append(_task_split_dict(
            X, y, y_cls, idx_t, test_split, val_split, seed, t))
    return tasks


# ── Specialization Matrix ─────────────────────────────────
def _make_spec_matrix(num_clients: int, cls_list: list) -> np.ndarray:
    """
    Ma tran specialization [num_clients x len(cls_list)].

    Cau truc voi 10 clients, 4 malware class, 1 benign:
      Client 0,1  : Adware   specialist  (55% Adware,  10% moi class khac, 15% Benign)
      Client 2,3  : Banking  specialist
      Client 4,5  : SMS      specialist
      Client 6,7  : Riskware specialist
      Client 8,9  : General  (20% moi class)

    Voi so client khac, tu dong dieu chinh:
      - spec_per_cls = max(1, (num_clients - n_general) // n_mal)
      - n_general    = max(1, num_clients % n_mal  hoac  2 neu du)
    """
    mal_cls = [c for c in cls_list if c != 5]
    n_mal   = len(mal_cls)
    n_cls   = len(cls_list)
    ben_idx = cls_list.index(5) if 5 in cls_list else None

    n_general    = max(1, num_clients - n_mal * 2)
    n_spec       = num_clients - n_general
    spec_per_cls = max(1, n_spec // n_mal)

    spec = np.zeros((num_clients, n_cls))
    r = 0
    for mal_c in mal_cls:
        if r >= num_clients - n_general:
            break
        mi = cls_list.index(mal_c)
        for _ in range(spec_per_cls):
            if r >= num_clients - n_general:
                break
            for ci, cls in enumerate(cls_list):
                if cls == mal_c:
                    spec[r, ci] = 0.55
                elif cls == 5:
                    spec[r, ci] = 0.15
                else:
                    spec[r, ci] = 0.10
            r += 1

    while r < num_clients:   # general clients
        spec[r] = 1.0 / n_cls
        r += 1

    return spec


# ── Category-Aware Client Partition ──────────────────────
def category_client_partition(X: np.ndarray, y: np.ndarray,
                               y_cls: np.ndarray,
                               num_clients: int,
                               seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Phan chia data cho N clients theo specialization matrix.
    Moi class duoc phan phoi theo trong so cua tung client.
    """
    rng      = np.random.default_rng(seed)
    cls_list = sorted(np.unique(y_cls).tolist())
    spec     = _make_spec_matrix(num_clients, cls_list)

    client_idx: Dict[int, list] = defaultdict(list)

    for ci, cls in enumerate(cls_list):
        arr = np.where(y_cls == cls)[0]
        if len(arr) == 0:
            continue
        rng.shuffle(arr)

        col   = spec[:, ci]
        total = col.sum()
        col   = col / total if total > 0 else np.ones(num_clients) / num_clients

        counts = (col * len(arr)).astype(int)
        counts[-1] += len(arr) - counts.sum()   # fix rounding

        cur = 0
        for c_idx, cnt in enumerate(counts):
            client_idx[c_idx].extend(arr[cur:cur + cnt].tolist())
            cur += cnt

    result = []
    for c in range(num_clients):
        idx = np.array(client_idx[c])
        if len(idx) == 0:
            idx = rng.choice(len(y), 10, replace=True)
        rng.shuffle(idx)
        result.append((X[idx], y[idx]))
    return result


# ── Dirichlet Partition (fallback) ───────────────────────
def dirichlet_partition(X: np.ndarray, y: np.ndarray,
                        num_clients: int, alpha: float,
                        seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    idx_per_client: Dict[int, list] = defaultdict(list)

    for cls in classes:
        idx_cls = np.where(y == cls)[0]
        rng.shuffle(idx_cls)
        props = rng.dirichlet([alpha] * num_clients)
        props = (props * len(idx_cls)).astype(int)
        props[-1] += len(idx_cls) - props.sum()
        cur = 0
        for c, cnt in enumerate(props):
            idx_per_client[c].extend(idx_cls[cur:cur + cnt].tolist())
            cur += cnt

    result = []
    for c in range(num_clients):
        idx = np.array(idx_per_client[c])
        if len(idx) == 0:
            idx = rng.choice(len(y), 10, replace=False)
        rng.shuffle(idx)
        result.append((X[idx], y[idx]))
    return result


# ── DataManager ──────────────────────────────────────────
class DataManager:
    """
    Giao dien chinh.
    Dung:
        dm = DataManager(cfg_data, cfg_fl)
        tasks = dm.prepare()
        # tasks[t][c] = {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """

    def __init__(self, data_cfg: dict, fl_cfg: dict):
        self.dcfg = data_cfg
        self.fcfg = fl_cfg
        self.scaler = StandardScaler()

    def prepare(self) -> List[List[Dict[str, DataLoader]]]:
        X, y, y_cls = self._load()
        sample_indices = self.dcfg.get("sample_indices")
        if sample_indices is not None:
            sample_indices = np.asarray(sample_indices, dtype=int)
            X = X[sample_indices]
            y = y[sample_indices]
            if y_cls is not None:
                y_cls = y_cls[sample_indices]
            print(f"[Data] K-Fold train subset: {len(sample_indices)} samples")

        X = self.scaler.fit_transform(X)
        self.dcfg["num_features"] = X.shape[1]

        task_strategy = self.dcfg.get("task_strategy", "temporal")
        partition_strategy = self.dcfg.get("partition_strategy", "dirichlet")

        use_category_tasks = (
            y_cls is not None
            and task_strategy in ("category", "category_strict", "category_revisit")
        )

        if use_category_tasks and task_strategy == "category":
            print("[Data] Task split: category-aware Domain-IL")
            task_splits = category_temporal_split(
                X, y, y_cls,
                num_tasks  = self.dcfg["num_tasks"],
                test_split = self.dcfg["test_split"],
                val_split  = self.dcfg["val_split"],
                seed       = self.dcfg["random_seed"],
            )
        elif use_category_tasks:
            revisit = task_strategy == "category_revisit"
            print("[Data] Task split: category strict"
                  + (" + old-family revisit" if revisit else ""))
            task_splits = category_strict_split(
                X, y, y_cls,
                num_tasks  = self.dcfg["num_tasks"],
                test_split = self.dcfg["test_split"],
                val_split  = self.dcfg["val_split"],
                seed       = self.dcfg["random_seed"],
                revisit_old = revisit,
            )
        else:
            print("[Data] Task split: temporal by index")
            task_splits = temporal_split(
                X, y,
                num_tasks  = self.dcfg["num_tasks"],
                test_split = self.dcfg["test_split"],
                val_split  = self.dcfg["val_split"],
                seed       = self.dcfg["random_seed"],
            )
        print(f"[Data] Client partition: {partition_strategy}")

        N  = self.fcfg["num_clients"]
        bs = self.fcfg["batch_size"]
        all_tasks = []

        for t, split in enumerate(task_splits):
            Xtr, ytr   = split["train"]
            Xval, yval = split["val"]
            Xte, yte   = split["test"]
            ycls_tr    = split.get("train_cls")

            if partition_strategy == "category" and ycls_tr is not None:
                parts = category_client_partition(
                    Xtr, ytr, ycls_tr, N,
                    seed=self.dcfg["random_seed"] + t)
            else:
                parts = dirichlet_partition(
                    Xtr, ytr, N,
                    alpha=self.dcfg["non_iid_alpha"],
                    seed=self.dcfg["random_seed"] + t)

            client_loaders = []
            for c in range(N):
                Xc, yc = parts[c]
                # drop_last chi khi co du samples
                do_drop = len(Xc) >= bs
                client_loaders.append({
                    "train": DataLoader(MalwareDataset(Xc, yc),
                                        batch_size=bs, shuffle=True,
                                        drop_last=do_drop),
                    "val":   DataLoader(MalwareDataset(Xval, yval),
                                        batch_size=256, shuffle=False),
                    "test":  DataLoader(MalwareDataset(Xte, yte),
                                        batch_size=256, shuffle=False),
                })
            all_tasks.append(client_loaders)

        print(f"[Data] {len(all_tasks)} tasks x {N} clients ready.")
        return all_tasks

    def _load(self):
        ds = self.dcfg["dataset"].lower()
        d  = self.dcfg["data_dir"]
        nf = self.dcfg["num_features"]
        sd = self.dcfg["random_seed"]
        if ds == "drebin":
            return load_drebin(d, nf, sd)
        elif ds == "cicmaldroid":
            return load_cicmaldroid(d, nf, sd)
        else:
            print("[Data] Using synthetic dataset.")
            return make_synthetic(5000, nf, sd)
