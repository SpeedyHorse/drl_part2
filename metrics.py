"""
Extended Python script to compute and save classification metrics from a confusion matrix CSV.

Features:
- Reads confusion matrix CSV (first column = true labels)
- Computes per-class Precision, Recall, F1-score, Support
- Computes overall Accuracy, macro- and micro- averages
- Saves detailed report to a CSV or text file

Usage:
    python compute_metrics.py <confusion_matrix.csv> <output_report.csv>
"""

import sys
import pandas as pd
import numpy as np

def load_confusion_matrix(csv_path):
    """
    Load confusion matrix from CSV.
    Assumes first column and header row contain integer class labels.
    """
    df = pd.read_csv(csv_path, index_col=0)
    # Drop any unnamed extra columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.index = df.index.astype(int)
    df.columns = df.columns.astype(int)
    return df.values.T, list(df.index)

def compute_metrics(cm, labels):
    """
    Compute per-class and overall classification metrics.
    Returns:
      - df_metrics: DataFrame with Label, Support, Precision, Recall, F1-score
      - summary: dict of overall Accuracy, macro/micro Precision, Recall, F1
    """
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    support = np.sum(cm, axis=1)

    precision = np.divide(
        TP, TP + FP,
        out=np.zeros_like(TP, dtype=float),
        where=(TP + FP) > 0
    )
    recall = np.divide(
        TP, TP + FN,
        out=np.zeros_like(TP, dtype=float),
        where=(TP + FN) > 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(TP, dtype=float),
        where=(precision + recall) > 0
    )

    total_correct = TP.sum()
    total_samples = cm.sum()
    accuracy = total_correct / total_samples

    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()

    micro_precision = total_correct / (total_correct + FP.sum())
    micro_recall = total_correct / (total_correct + FN.sum())
    micro_f1 = (
        2 * micro_precision * micro_recall /
        (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0 else 0
    )

    df_metrics = pd.DataFrame({
        'Label': labels,
        'Support': support,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1
    })

    summary = {
        'Accuracy': accuracy,
        'Macro Precision': macro_precision,
        'Macro Recall': macro_recall,
        'Macro F1-score': macro_f1,
        'Micro Precision': micro_precision,
        'Micro Recall': micro_recall,
        'Micro F1-score': micro_f1
    }

    return df_metrics, summary

def save_report(df_metrics, summary, output_path):
    """
    Save per-class metrics to CSV and summary metrics to text file.
    """
    base, ext = output_path.rsplit('.', 1)
    metrics_file = f"{base}_classes.{ext}"
    df_metrics.to_csv(metrics_file, index=False)

    summary_file = f"{base}_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Overall Metrics:\n")
        for key, value in summary.items():
            f.write(f"{key}: {value:.4f}\n")

    print(f"Per-class metrics saved to {metrics_file}")
    print(f"Summary metrics saved to {summary_file}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python compute_metrics.py <confusion_matrix.csv> <output_report.csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_report = sys.argv[2]

    cm, labels = load_confusion_matrix(input_csv)
    df_metrics, summary = compute_metrics(cm, labels)
    save_report(df_metrics, summary, output_report)

if __name__ == '__main__':
    main()

