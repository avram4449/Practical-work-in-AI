import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

parent_dir = r"C:\FILES\jku\Semester 6\Practical work\realistic-al-open-source\experiments\test"

n_classes = 12
tox21_class_names = [
    "NR.AhR", "NR.AR", "NR.AR.LBD", "NR.Aromatase", "NR.ER", "NR.ER.LBD",
    "NR.PPAR.gamma", "SR.ARE", "SR.ATAD5", "SR.HSE", "SR.MMP", "SR.p53"
]

# Collect all test_predictions.csv files from all runs and iterations
all_results = []
for run_dir in glob.glob(os.path.join(parent_dir, "*")):
    if not os.path.isdir(run_dir):
        continue
    stored_path = os.path.join(run_dir, "stored.npz")
    if not os.path.exists(stored_path):
        continue
    num_samples = np.load(stored_path)["num_samples"]
    loop_dirs = sorted([d for d in os.listdir(run_dir) if d.startswith("loop-")], key=lambda x: int(x.split('-')[1]))
    for i, loop_dir in enumerate(loop_dirs):
        pred_path = os.path.join(run_dir, loop_dir, "test_predictions.csv")
        if not os.path.exists(pred_path):
            continue
        df = pd.read_csv(pred_path)
        num_labeled = num_samples[i] if i < len(num_samples) else None
        for class_idx in range(n_classes):
            y_true_col = f"true_target_{class_idx}"
            y_pred_col = f"pred_target_{class_idx}"
            if y_true_col in df.columns and y_pred_col in df.columns:
                y_true = df[y_true_col].values
                y_pred = df[y_pred_col].values
                if np.unique(y_true).size == 2:
                    auc = roc_auc_score(y_true, y_pred)
                else:
                    auc = np.nan
                all_results.append({
                    "run": os.path.basename(run_dir),
                    "iteration": i,
                    "num_labeled": num_labeled,
                    "class": class_idx,
                    "auc": auc
                })

# Convert to DataFrame
results_df = pd.DataFrame(all_results)

# Compute mean, std, and best per-class AUC across runs for each iteration
grouped = results_df.groupby(["iteration", "class"])
mean_auc = grouped["auc"].mean().reset_index(name="mean_auc")
std_auc = grouped["auc"].std().reset_index(name="std_auc")

# Compute best AUC per class (across all runs and iterations)
best_auc = results_df.groupby("class")["auc"].max().reset_index(name="best_auc")

# Merge for deviation
mean_auc = mean_auc.merge(best_auc, on="class")
mean_auc["deviation"] = mean_auc["best_auc"] - mean_auc["mean_auc"]

# --- Plot all classes in one figure ---
fig = go.Figure()

for class_idx in range(n_classes):
    class_data = mean_auc[mean_auc["class"] == class_idx]
    std_data = std_auc[std_auc["class"] == class_idx]
    # Mean AUC with error bars
    fig.add_trace(go.Scatter(
        x=class_data["iteration"],
        y=class_data["mean_auc"],
        error_y=dict(type='data', array=std_data["std_auc"], visible=True),
        mode='lines+markers',
        name=f'{tox21_class_names[class_idx]} Mean AUC',
        hovertemplate='%{y:.6f} ± %{error_y.array:.4f}<extra>%{fullData.name}</extra>'
    ))
    # Deviation from best as dashed line (no error bars)
    fig.add_trace(go.Scatter(
        x=class_data["iteration"],
        y=class_data["deviation"],
        mode='lines+markers',
        name=f'{tox21_class_names[class_idx]} Deviation',
        line=dict(dash='dot'),
        hovertemplate='%{y:.6f}<extra>%{fullData.name}</extra>'
    ))

fig.update_layout(
    title="Mean AUC and Deviation from Best per Class vs. Iteration",
    xaxis_title="Iteration",
    yaxis_title="Value",
    legend_title="Tox21 Class / Metric",
    hovermode="x unified"
)
fig.show()