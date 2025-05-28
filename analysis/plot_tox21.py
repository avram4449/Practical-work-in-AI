import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set this to your experiment folder containing stored.npz or test_metrics.csv
RESULTS_DIR = r"c:\FILES\jku\Semester 6\Practical work\realistic-al-open-source\experiments\test\2025-05-27_16-15-54-940046"

def annotate_points(x, y, ax, fmt="{:.9f}"):
    for xi, yi in zip(x, y):
        ax.annotate(fmt.format(yi), (xi, yi), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8, color='black')

def plot_from_stored_npz(results_dir):
    npz_path = os.path.join(results_dir, "stored.npz")
    if not os.path.exists(npz_path):
        print(f"stored.npz not found in {results_dir}")
        return
    data = np.load(npz_path)
    num_samples = data["num_samples"]
    test_acc = data["test_acc"]
    val_acc = data["val_acc"]
    test_loss = data["test_loss"] if "test_loss" in data else None
    val_loss = data["val_loss"] if "val_loss" in data else None

    # Accuracy plot
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    plt.plot(num_samples, test_acc, marker='o', label="Test Accuracy")
    plt.plot(num_samples, val_acc, marker='s', label="Validation Accuracy")
    annotate_points(num_samples, test_acc, ax)
    annotate_points(num_samples, val_acc, ax)
    plt.xlabel("Number of Labelled Samples")
    plt.ylabel("Accuracy")
    plt.title("Active Learning Progress: Accuracy")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Loss plot (if available)
    if test_loss is not None or val_loss is not None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        if test_loss is not None:
            plt.plot(num_samples, test_loss, marker='o', label="Test Loss", color='red')
            annotate_points(num_samples, test_loss, ax, fmt="{:.4f}")
        if val_loss is not None:
            plt.plot(num_samples, val_loss, marker='s', label="Validation Loss", color='orange')
            annotate_points(num_samples, val_loss, ax, fmt="{:.4f}")
        plt.xlabel("Number of Labelled Samples")
        plt.ylabel("Loss")
        plt.title("Active Learning Progress: Loss")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

def plot_from_test_metrics(results_dir):
    csv_path = os.path.join(results_dir, "test_metrics.csv")
    if not os.path.exists(csv_path):
        print(f"test_metrics.csv not found in {results_dir}")
        return
    df = pd.read_csv(csv_path)
    if "num_labelled" in df.columns:
        x = df["num_labelled"]
    elif "iteration" in df.columns:
        x = df["iteration"]
    else:
        x = range(len(df))

    # Accuracy plot
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    if "test/acc" in df.columns:
        plt.plot(x, df["test/acc"], marker='o', label="Test Accuracy")
        annotate_points(x, df["test/acc"], ax)
    if "val/acc" in df.columns:
        plt.plot(x, df["val/acc"], marker='s', label="Validation Accuracy")
        annotate_points(x, df["val/acc"], ax)
    plt.xlabel("Number of Labelled Samples" if "num_labelled" in df.columns else "Iteration")
    plt.ylabel("Accuracy")
    plt.title("Active Learning Progress: Accuracy")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Loss plot
    if "test/loss" in df.columns or "val/loss" in df.columns:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        if "test/loss" in df.columns:
            plt.plot(x, df["test/loss"], marker='o', label="Test Loss", color='red')
            annotate_points(x, df["test/loss"], ax, fmt="{:.4f}")
        if "val/loss" in df.columns:
            plt.plot(x, df["val/loss"], marker='s', label="Validation Loss", color='orange')
            annotate_points(x, df["val/loss"], ax, fmt="{:.4f}")
        plt.xlabel("Number of Labelled Samples" if "num_labelled" in df.columns else "Iteration")
        plt.ylabel("Loss")
        plt.title("Active Learning Progress: Loss")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Try both plotting methods
    plot_from_stored_npz(RESULTS_DIR)
    plot_from_test_metrics(RESULTS_DIR)