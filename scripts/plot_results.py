import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DISPLAY_NAME = {
    "iforest": "Isolation Forest",
    "lstm": "LSTM-AE",
    "gcn": "GCN-AE",
    "hier_hgnn": "Improved HGNN-AE",
    "hgnn": "Original HGNN-AE",
}


def model_label(model_key: str) -> str:
    return DISPLAY_NAME.get(model_key, model_key)


def ensure_fig_dir(fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)


def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    for c in ["precision", "recall", "f1"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["precision", "recall", "f1"])
    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("model", as_index=False).agg(
        mean_precision=("precision", "mean"),
        std_precision=("precision", "std"),
        mean_recall=("recall", "mean"),
        std_recall=("recall", "std"),
        mean_f1=("f1", "mean"),
        std_f1=("f1", "std"),
        runs=("f1", "count"),
    )
    grouped = grouped.sort_values(by="mean_f1", ascending=False).reset_index(drop=True)
    grouped["std_precision"] = grouped["std_precision"].fillna(0.0)
    grouped["std_recall"] = grouped["std_recall"].fillna(0.0)
    grouped["std_f1"] = grouped["std_f1"].fillna(0.0)
    grouped["model_display"] = grouped["model"].map(model_label)
    return grouped


def plot_f1_bar(summary: pd.DataFrame, fig_dir: Path) -> None:
    x = np.arange(len(summary))
    means = summary["mean_f1"].values
    stds = summary["std_f1"].values
    labels = summary["model_display"].tolist()

    plt.figure(figsize=(8.2, 4.8))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel("F1-score")
    plt.title("Average F1-score Across Machines")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / "f1_bar_with_std.png", dpi=200)
    plt.close()


def plot_prf_grouped(summary: pd.DataFrame, fig_dir: Path) -> None:
    labels = summary["model_display"].tolist()
    x = np.arange(len(labels))
    w = 0.25

    p = summary["mean_precision"].values
    r = summary["mean_recall"].values
    f = summary["mean_f1"].values

    plt.figure(figsize=(9.0, 5.0))
    plt.bar(x - w, p, width=w, label="Precision")
    plt.bar(x, r, width=w, label="Recall")
    plt.bar(x + w, f, width=w, label="F1")
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel("Score")
    plt.title("Precision / Recall / F1 Comparison")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / "prf_grouped_bar.png", dpi=200)
    plt.close()


def plot_machine_f1(df: pd.DataFrame, fig_dir: Path) -> None:
    pivot = df.pivot_table(index="machine", columns="model", values="f1", aggfunc="mean")
    pivot = pivot.sort_index()
    model_cols = list(pivot.columns)
    x = np.arange(len(pivot.index))

    plt.figure(figsize=(10.0, 5.2))
    for m in model_cols:
        plt.plot(x, pivot[m].values, marker="o", linewidth=1.8, label=model_label(m))

    plt.xticks(x, pivot.index.tolist(), rotation=30, ha="right")
    plt.ylabel("F1-score")
    plt.title("Per-machine F1-score by Model")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "f1_per_machine_line.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot and summarize batch experiment results.")
    parser.add_argument("--results-csv", type=str, default="results/results.csv")
    parser.add_argument("--summary-csv", type=str, default="results/summary.csv")
    parser.add_argument("--fig-dir", type=str, default="results/figures")
    args = parser.parse_args()

    results_csv = PROJECT_ROOT / args.results_csv
    summary_csv = PROJECT_ROOT / args.summary_csv
    fig_dir = PROJECT_ROOT / args.fig_dir
    ensure_fig_dir(fig_dir)

    if not results_csv.exists():
        raise FileNotFoundError(f"results CSV not found: {results_csv}")

    df = pd.read_csv(results_csv)
    df_ok = clean_results(df)
    if len(df_ok) == 0:
        raise ValueError("No valid successful rows in results.csv")

    summary = build_summary(df_ok)
    summary_out = summary[
        [
            "model",
            "model_display",
            "mean_precision",
            "std_precision",
            "mean_recall",
            "std_recall",
            "mean_f1",
            "std_f1",
            "runs",
        ]
    ]
    summary_out.to_csv(summary_csv, index=False)

    plot_f1_bar(summary, fig_dir)
    plot_prf_grouped(summary, fig_dir)
    plot_machine_f1(df_ok, fig_dir)

    print(f"Loaded results from: {results_csv}")
    print(f"Saved summary to   : {summary_csv}")
    print(f"Saved figures to   : {fig_dir}")
    print("\nSummary preview:")
    print(summary_out.to_string(index=False))


if __name__ == "__main__":
    main()
