import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Directory containing all loop-* subfolders
base_dir = r"C:\FILES\jku\Semester 6\Practical work\realistic-al-open-source\experiments\test\2025-08-08_20-03-52-128075"
n_classes = 12

# Load true number of labeled samples per iteration
stored = np.load(os.path.join(base_dir, "stored.npz"))
num_samples = stored["num_samples"]  # shape: (num_iterations,)

# Find all loop directories
loop_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("loop-")], key=lambda x: int(x.split('-')[1]))

results = []
for i, loop_dir in enumerate(loop_dirs):
    pred_path = os.path.join(base_dir, loop_dir, "test_predictions.csv")
    if not os.path.exists(pred_path):
        continue
    df = pd.read_csv(pred_path)
    # Use the actual number of labeled samples for this iteration
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
            results.append({"num_labeled": num_labeled, "class": class_idx, "auc": auc})

# Convert to DataFrame
results_df = pd.DataFrame(results)

tox21_class_names = [
    "NR.AhR", "NR.AR", "NR.AR.LBD", "NR.Aromatase", "NR.ER", "NR.ER.LBD",
    "NR.PPAR.gamma", "SR.ARE", "SR.ATAD5", "SR.HSE", "SR.MMP", "SR.p53"
]

# Plot
plt.figure(figsize=(10, 6))
for class_idx in range(n_classes):
    class_data = results_df[results_df["class"] == class_idx]
    plt.plot(class_data["num_labeled"], class_data["auc"], marker="o", label=tox21_class_names[class_idx])

# Compute and plot average (macro) AUC across all classes for each iteration
avg_auc_per_iter = results_df.groupby("num_labeled")["auc"].mean().reset_index()

# Calculate overall average AUC (across all classes and all iterations)
overall_avg_auc = results_df["auc"].mean()

plt.plot(
    avg_auc_per_iter["num_labeled"],
    avg_auc_per_iter["auc"],
    marker="s",
    color="black",
    linewidth=2,
    label=f"Average AUC (macro)\nOverall avg: {overall_avg_auc:.4f}"
)

plt.xlabel("Number of labeled samples")
plt.ylabel("AUC")
plt.title("Per-class AUC vs. Number of Labeled Samples")
plt.ylim(0, 1)
plt.grid(True)
plt.legend(title="Tox21 Class", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()