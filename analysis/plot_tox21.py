import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, log_loss

RESULTS_DIR = r"C:\FILES\jku\Semester 6\Practical work\realistic-al-open-source\experiments\test\2025-07-02_14-59-40-835377\loop-19"
CSV_PATH = os.path.join(RESULTS_DIR, "test_predictions.csv")

TARGET_NAMES = [
    "NR.AhR", "NR.AR", "NR.AR.LBD", "NR.Aromatase", "NR.ER", "NR.ER.LBD",
    "NR.PPAR.gamma", "SR.ARE", "SR.ATAD5", "SR.HSE", "SR.MMP", "SR.p53"
]

def plot_all_metrics(csv_path, n_targets=12, target_names=None):
    if target_names is None:
        target_names = [f"Target {i}" for i in range(n_targets)]
    df = pd.read_csv(csv_path)
    aucs, accs, losses = [], [], []
    plt.figure(figsize=(10, 8))
    for i in range(n_targets):
        y_true = df[f"true_target_{i}"]
        y_score = df[f"pred_target_{i}"]
        if y_true.nunique() < 2:
            aucs.append(np.nan)
            accs.append(np.nan)
            losses.append(np.nan)
            continue
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        aucs.append(auc)
        accs.append(accuracy_score(y_true, (y_score > 0.5).astype(int)))
        losses.append(log_loss(y_true, y_score, labels=[0,1]))
        plt.plot(fpr, tpr, label=f"{target_names[i]} (AUC={auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves per Target")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Log Loss
    plt.figure(figsize=(10, 4))
    plt.bar(target_names, losses)
    plt.xlabel("Target")
    plt.ylabel("Log Loss")
    plt.title("Per-target Log Loss")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Accuracy
    plt.figure(figsize=(10, 4))
    plt.bar(target_names, accs)
    plt.xlabel("Target")
    plt.ylabel("Accuracy")
    plt.title("Per-target Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # ROC AUC
    plt.figure(figsize=(10, 4))
    plt.bar(target_names, aucs)
    plt.xlabel("Target")
    plt.ylabel("ROC AUC")
    plt.title("Per-target ROC AUC")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_all_metrics(CSV_PATH, n_targets=12, target_names=TARGET_NAMES)

