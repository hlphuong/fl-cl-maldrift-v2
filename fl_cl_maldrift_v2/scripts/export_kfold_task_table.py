import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ["FedAvg", "FL-MalDrift", "FL-CL-MalDrift"]
TASK_ORDER = [0, 1, 2, 3, 4]
TASK_METRICS = [
    ("acc", "Acc"),
    ("precision", "Prec"),
    ("recall", "Rec"),
    ("f1", "F1"),
]
OVERALL_METRICS = [
    ("ACC", "Acc"),
    ("Precision", "Prec"),
    ("Recall", "Rec"),
    ("F1", "F1"),
    ("Forgetting", "Forget"),
    ("BWT", "BWT"),
    ("FWT", "FWT"),
]
ALL_DISPLAY_METRICS = ["Acc", "Prec", "Rec", "F1", "Forget", "BWT", "FWT"]
PM = "\u00b1"


def fmt_value(value: float) -> str:
    return f"{float(value):.4f}"


def fmt_mean_std(values) -> str:
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.4f} {PM} {values.std():.4f}"


def build_table(result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_task_path = result_dir / "kfold_per_task.csv"
    per_fold_path = result_dir / "kfold_per_fold.csv"
    if not per_task_path.exists():
        raise FileNotFoundError(f"Missing {per_task_path}")
    if not per_fold_path.exists():
        raise FileNotFoundError(f"Missing {per_fold_path}")

    per_task = pd.read_csv(per_task_path)
    per_fold = pd.read_csv(per_fold_path)

    formatted_rows = []
    raw_rows = []

    for fold in sorted(per_task["fold"].unique()):
        fold_df = per_task[per_task["fold"] == fold]
        for task in TASK_ORDER:
            task_df = fold_df[fold_df["task"] == task]
            for method in METHOD_ORDER:
                row = task_df[task_df["method"] == method]
                if row.empty:
                    continue
                row = row.iloc[0]
                formatted = {
                    "Scenario": f"Fold {fold}",
                    "Task": f"Task {task}",
                    "Method": method,
                }
                raw = dict(formatted)
                for dst in ALL_DISPLAY_METRICS:
                    formatted[dst] = ""
                    raw[dst] = np.nan
                for src, dst in TASK_METRICS:
                    formatted[dst] = fmt_value(row[src])
                    raw[dst] = float(row[src])
                formatted_rows.append(formatted)
                raw_rows.append(raw)

        fold_overall = per_fold[per_fold["fold"] == fold]
        for method in METHOD_ORDER:
            row = fold_overall[fold_overall["method"] == method]
            if row.empty:
                continue
            row = row.iloc[0]
            formatted = {
                "Scenario": f"Fold {fold}",
                "Task": "Overall",
                "Method": method,
            }
            raw = dict(formatted)
            for src, dst in OVERALL_METRICS:
                formatted[dst] = fmt_value(row[src])
                raw[dst] = float(row[src])
            formatted_rows.append(formatted)
            raw_rows.append(raw)

    for task in TASK_ORDER:
        task_df = per_task[per_task["task"] == task]
        for method in METHOD_ORDER:
            method_df = task_df[task_df["method"] == method]
            if method_df.empty:
                continue
            formatted = {
                "Scenario": f"Mean {PM} Std",
                "Task": f"Task {task}",
                "Method": method,
            }
            raw = dict(formatted)
            for dst in ALL_DISPLAY_METRICS:
                formatted[dst] = ""
                raw[dst] = np.nan
            for src, dst in TASK_METRICS:
                values = method_df[src].to_numpy(dtype=float)
                formatted[dst] = fmt_mean_std(values)
                raw[dst] = float(values.mean())
            formatted_rows.append(formatted)
            raw_rows.append(raw)

    for method in METHOD_ORDER:
        method_df = per_fold[per_fold["method"] == method]
        if method_df.empty:
            continue
        formatted = {
            "Scenario": f"Mean {PM} Std",
            "Task": "Overall",
            "Method": method,
        }
        raw = dict(formatted)
        for src, dst in OVERALL_METRICS:
            values = method_df[src].to_numpy(dtype=float)
            formatted[dst] = fmt_mean_std(values)
            raw[dst] = float(values.mean())
        formatted_rows.append(formatted)
        raw_rows.append(raw)

    columns = ["Scenario", "Task", "Method"] + ALL_DISPLAY_METRICS
    return pd.DataFrame(formatted_rows, columns=columns), pd.DataFrame(raw_rows, columns=columns)


def render_table(display_df: pd.DataFrame, raw_df: pd.DataFrame, out_png: Path):
    plot_df = display_df.copy()
    last_scenario = None
    last_task_key = None
    for i, row in plot_df.iterrows():
        scenario = row["Scenario"]
        task_key = (row["Scenario"], row["Task"])
        if scenario == last_scenario:
            plot_df.at[i, "Scenario"] = ""
        else:
            last_scenario = scenario
            last_task_key = None
        if task_key == last_task_key:
            plot_df.at[i, "Task"] = ""
        else:
            last_task_key = task_key

    n_rows = len(plot_df)
    fig_h = max(8.0, 0.23 * (n_rows + 4))
    fig_w = 13.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title("K-Fold Task Comparison (5 folds, 5 clients)",
                 fontsize=13, fontweight="bold", pad=12)

    table = ax.table(
        cellText=plot_df.values,
        colLabels=plot_df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.0, 0.0, 1.0, 0.965],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.2)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#4040a0")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_facecolor("#eeeeff")
            cell.set_text_props(weight="bold", color="#1f2f8f")
        elif raw_df.iloc[row - 1]["Scenario"].startswith("Mean"):
            cell.set_facecolor("#f7f7ff")

    metric_cols = {
        "Acc": 3,
        "Prec": 4,
        "Rec": 5,
        "F1": 6,
        "Forget": 7,
        "BWT": 8,
        "FWT": 9,
    }
    for (scenario, task), grp in raw_df.groupby(["Scenario", "Task"], sort=False):
        idxs = grp.index.tolist()
        for metric, col in metric_cols.items():
            vals = pd.to_numeric(grp[metric], errors="coerce")
            if vals.isna().all():
                continue
            best = vals.min() if metric == "Forget" else vals.max()
            for idx in idxs:
                value = raw_df.at[idx, metric]
                if pd.notna(value) and abs(value - best) < 1e-12:
                    table[(idx + 1, col)].set_text_props(weight="bold")

    col_widths = {
        0: 0.14,
        1: 0.10,
        2: 0.17,
        3: 0.105,
        4: 0.105,
        5: 0.105,
        6: 0.105,
        7: 0.105,
        8: 0.105,
        9: 0.105,
    }
    for (row, col), cell in table.get_celld().items():
        if col in col_widths:
            cell.set_width(col_widths[col])

    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--out_prefix", default="kfold_task_table")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    display_df, raw_df = build_table(result_dir)

    out_csv = result_dir / f"{args.out_prefix}.csv"
    out_png = result_dir / f"{args.out_prefix}.png"
    display_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    render_table(display_df, raw_df, out_png)

    print(f"CSV: {out_csv}")
    print(f"PNG: {out_png}")


if __name__ == "__main__":
    main()
