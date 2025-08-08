import os
import pandas as pd

RESULTS_DIR = r"C:\FILES\jku\Semester 6\Practical work\realistic-al-open-source\experiments\test\2025-06-10_15-46-59-382744\loop-19"
CSV_PATH = os.path.join(RESULTS_DIR, "test_predictions.csv")

TARGET_NAMES = [
    "NR.AhR", "NR.AR", "NR.AR.LBD", "NR.Aromatase", "NR.ER", "NR.ER.LBD",
    "NR.PPAR.gamma", "SR.ARE", "SR.ATAD5", "SR.HSE", "SR.MMP", "SR.p53"
]

def show_predicted_positives(csv_path, target_names):
    df = pd.read_csv(csv_path)
    for i, target in enumerate(target_names):
        pred_col = f"pred_target_{i}"
        if pred_col in df.columns:
            # Count how many were predicted as positive (pred > 0.5)
            num_pred_pos = (df[pred_col] > 0.5).sum()
            print(f"Target: {target} - Number predicted as positive: {num_pred_pos}")

if __name__ == "__main__":
    show_predicted_positives(CSV_PATH, TARGET_NAMES)