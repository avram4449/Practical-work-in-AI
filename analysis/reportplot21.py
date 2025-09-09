import os
import sys
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

n_classes = 12
tox21_class_names = [
    "NR.AhR", "NR.AR", "NR.AR.LBD", "NR.Aromatase", "NR.ER", "NR.ER.LBD",
    "NR.PPAR.gamma", "SR.ARE", "SR.ATAD5", "SR.HSE", "SR.MMP", "SR.p53"
]

def load_results(parent_dir: str) -> pd.DataFrame:
    """Read all test_predictions.csv files under parent_dir/runX/loop-Y and compute per-class AUC."""
    all_rows = []

    for run_dir in glob.glob(os.path.join(parent_dir, "*")):
        if not os.path.isdir(run_dir):
            continue

        stored_path = os.path.join(run_dir, "stored.npz")
        if not os.path.exists(stored_path):
            continue

        # num_samples is [num_iterations] array
        try:
            num_samples = np.load(stored_path)["num_samples"]
        except Exception:
            num_samples = None

        loop_dirs = sorted(
            [d for d in os.listdir(run_dir) if d.startswith("loop-") and os.path.isdir(os.path.join(run_dir, d))],
            key=lambda x: int(x.split("-")[1])
        )

        for i, loop_dir in enumerate(loop_dirs):
            pred_path = os.path.join(run_dir, loop_dir, "test_predictions.csv")
            if not os.path.exists(pred_path):
                continue

            try:
                df = pd.read_csv(pred_path)
            except Exception:
                continue

            num_labeled = None
            if num_samples is not None and i < len(num_samples):
                num_labeled = int(num_samples[i])

            for class_idx in range(n_classes):
                y_true_col = f"true_target_{class_idx}"
                y_pred_col = f"pred_target_{class_idx}"
                if y_true_col in df.columns and y_pred_col in df.columns:
                    y_true = df[y_true_col].values
                    y_pred = df[y_pred_col].values
                    auc = np.nan
                    # Only compute AUC if we have both classes present
                    vals = np.unique(y_true)
                    if vals.size >= 2:
                        try:
                            auc = roc_auc_score(y_true, y_pred)
                        except Exception:
                            auc = np.nan

                    all_rows.append(
                        {
                            "run": os.path.basename(run_dir),
                            "iteration": i,
                            "num_labeled": num_labeled,
                            "class": class_idx,
                            "auc": auc,
                        }
                    )

    results_df = pd.DataFrame(all_rows)
    return results_df


def aggregate(results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per (iteration,class,run): mean/std/count AUC and per-iteration num_labeled stats."""
    df = results_df.dropna(subset=["auc"]).copy()

    # Per-iteration/class/run aggregates
    grouped = df.groupby(["iteration", "class", "run"])
    aggr = grouped["auc"].agg(["mean", "std", "count"]).reset_index()
    aggr = aggr.rename(columns={"mean": "mean_auc", "std": "std_auc", "count": "n_runs"})
    aggr["std_auc"] = aggr["std_auc"].fillna(0.0)

    # Best AUC per class/method (over all runs & iterations)
    best = df.groupby(["class", "run"])["auc"].max().reset_index(name="best_auc")
    aggr = aggr.merge(best, on=["class", "run"], how="left")
    aggr["deviation"] = aggr["best_auc"] - aggr["mean_auc"]

    # Per-iteration labeled count (use median across runs in case it differs)
    nl_aggr = (
        results_df.dropna(subset=["num_labeled"])
        .groupby(["iteration"])["num_labeled"]
        .agg(["median", "min", "max"])
        .reset_index()
        .rename(columns={"median": "num_labeled_median", "min": "num_labeled_min", "max": "num_labeled_max"})
    )

    return aggr, nl_aggr


def plot_grid(aggr: pd.DataFrame, nl_aggr: pd.DataFrame, out_png: str | None = None, show_values: bool = False):
    classes = list(range(n_classes))
    iters = sorted(aggr["iteration"].unique())
    x = np.array(iters, dtype=int)

    def method_from_run(run_name):
        run_name = run_name.lower()
        if "bald" in run_name:
            return "BALD"
        elif "random" in run_name:
            return "Random"
        else:
            return "Other"

    if "run" in aggr.columns:
        aggr["method"] = aggr["run"].map(method_from_run)
    else:
        aggr["method"] = "Unknown"

    method_colors = {"BALD": "#1f77b4", "Random": "#ff7f0e"}
    method_labels = {"BALD": "BALD", "Random": "Random"}

    # Aggregate over runs: mean and std per (iteration, class, method)
    aggr2 = (
        aggr.groupby(["iteration", "class", "method"])
        .agg(mean_auc=("mean_auc", "mean"), std_auc=("mean_auc", "std"))
        .reset_index()
    )

    fig, axes = plt.subplots(3, 4, figsize=(20, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    ylim = (0.5, 1.0)

    for c in classes:
        ax = axes[c]
        ax.set_title(tox21_class_names[c], fontsize=14)
        ax.set_xticks(x)
        ax.grid(True, linestyle=":", alpha=0.4)

        # Find min/max for y-axis for this class across both methods
        ymins, ymaxs = [], []
        for method in method_colors.keys():
            cdf = aggr2[(aggr2["class"] == c) & (aggr2["method"] == method)].sort_values("iteration")
            if cdf.empty:
                continue
            m = cdf["mean_auc"].to_numpy()
            s = cdf["std_auc"].to_numpy()
            ymins.append((m - s).min())
            ymaxs.append((m + s).max())
        # Set y-axis limits with a small margin
        if ymins and ymaxs:
            ymin = max(0.0, min(ymins) - 0.05)
            ymax = min(1.0, max(ymaxs) + 0.05)
            ax.set_ylim(ymin, ymax)
        else:
            ax.set_ylim(0.0, 1.0)

        # First, plot shaded std bands for both methods
        for method, color in method_colors.items():
            cdf = aggr2[(aggr2["class"] == c) & (aggr2["method"] == method)].sort_values("iteration")
            if cdf.empty:
                continue
            xs = cdf["iteration"].to_numpy()
            m = cdf["mean_auc"].to_numpy()
            s = cdf["std_auc"].to_numpy()
            ax.fill_between(xs, m - s, m + s, color=color, alpha=0.12, zorder=1)

        # Then, plot mean lines and markers for both methods (on top)
        for method, color in method_colors.items():
            cdf = aggr2[(aggr2["class"] == c) & (aggr2["method"] == method)].sort_values("iteration")
            if cdf.empty:
                continue
            xs = cdf["iteration"].to_numpy()
            m = cdf["mean_auc"].to_numpy()
            ax.plot(xs, m, color=color, marker="o", markersize=5, linewidth=2, alpha=0.85, label=f"{method} Mean AUC", zorder=2)

    # Shared legend
    handles, labels = [], []
    for method, color in method_colors.items():
        handles.append(plt.Line2D([0], [0], color=color, marker="o", linewidth=2))
        labels.append(f"{method} Mean AUC")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=13,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.suptitle(
        "Tox21 per-class Mean AUC for BALD vs Random in 5500 samples(shaded: ±1 std)",
        y=0.99,
        fontsize=18,
    )
    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.94])

    if out_png:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.savefig(out_png, dpi=200)
        print(f"Saved figure to: {out_png}")

    plt.show()


def main():
    # Parent dir can be passed as CLI arg; default to your current experiments folder
    if len(sys.argv) > 1:
        parent_dir = sys.argv[1]
    else:
        parent_dir = r"C:\FILES\jku\Semester 6\Practical work\realistic-al-open-source\experiments\test"

    print(f"Scanning runs under: {parent_dir}")
    results_df = load_results(parent_dir)
    if results_df.empty:
        print("No results found.")
        return

    # Save CSV with raw rows for reference
    out_dir = os.path.join(parent_dir, "_aggregate")
    os.makedirs(out_dir, exist_ok=True)
    raw_csv = os.path.join(out_dir, "all_results_raw.csv")
    results_df.to_csv(raw_csv, index=False)
    print(f"Saved raw results to: {raw_csv}")

    aggr, nl_aggr = aggregate(results_df)

    # Save the aggregated CSV
    aggr_csv = os.path.join(out_dir, "per_class_iteration_mean_std.csv")
    aggr.to_csv(aggr_csv, index=False)
    print(f"Saved aggregated results to: {aggr_csv}")

    # Plot
    out_png = os.path.join(out_dir, "tox21_auc_grid.png")
    plot_grid(aggr, nl_aggr, out_png=out_png, show_values=True)


if __name__ == "__main__":
    main()