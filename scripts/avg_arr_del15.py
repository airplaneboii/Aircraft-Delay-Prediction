# Simple script to compute classification baselines for ARR_DEL15
# Configure the variables below and run with: python scripts/avg_arr_del15.py

DATA_PATH = "data/datasets/2M_20251221_210514.csv"  # path to CSV file or directory
HEAD_ROWS = None  # set to an int to limit rows for quick checks, or None to read all
IGNORE_CANCELLED = True  # if True, drop rows where CANCELLED == 1
SAVE_PLOT = True  # save distribution plot to disk
PLOT_PATH = "scripts/arr_del15_distribution.png"

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_csv(path, nrows=None):
    if os.path.isdir(path):
        # pick latest CSV in directory
        files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith('.csv')]
        if not files:
            raise FileNotFoundError(f"No CSV files found in directory: {path}")
        path = max(files, key=os.path.getmtime)
    return pd.read_csv(path, nrows=nrows)


def _metrics_from_labels(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.logical_and(y_true == 1, y_pred == 1).sum())
    tn = int(np.logical_and(y_true == 0, y_pred == 0).sum())
    fp = int(np.logical_and(y_true == 0, y_pred == 1).sum())
    fn = int(np.logical_and(y_true == 1, y_pred == 0).sum())
    eps = 1e-12
    accuracy = float((tp + tn) / max(1, (tp + tn + fp + fn)))
    precision = float(tp / (tp + fp + eps))
    recall = float(tp / (tp + fn + eps))
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
    }


def main():
    df = load_csv(DATA_PATH, nrows=HEAD_ROWS)

    if IGNORE_CANCELLED and "CANCELLED" in df.columns:
        df = df[df["CANCELLED"] == 0]

    # Prefer an existing ARR_DEL15 column; if missing try to build from ARR_DELAY
    if "ARR_DEL15" not in df.columns:
        if "ARR_DELAY" in df.columns:
            df["ARR_DEL15"] = (pd.to_numeric(df["ARR_DELAY"], errors='coerce') >= 15).astype(int)
            print("Column ARR_DEL15 not found; deriving from ARR_DELAY >= 15")
        else:
            print("Column ARR_DEL15 not found in dataset and ARR_DELAY is not available to derive it.")
            return

    df["ARR_DEL15"] = df["ARR_DEL15"].fillna(0).astype(int)
    vals = df["ARR_DEL15"].values.astype(int)

    total = len(vals)
    if total == 0:
        print("No rows available for ARR_DEL15 analysis")
        return

    pos = int((vals == 1).sum())
    neg = int((vals == 0).sum())
    pos_frac = float(pos) / total

    print(f"ARR_DEL15 stats from {DATA_PATH} (n={total}):")
    print(f"  positives = {pos} ({pos_frac*100:.2f}%)")
    print(f"  negatives = {neg} ({100 - pos_frac*100:.2f}%)")

    # Baseline predictors
    preds_all_zero = np.zeros_like(vals)
    preds_all_one = np.ones_like(vals)

    m_zero = _metrics_from_labels(vals, preds_all_zero)
    m_one = _metrics_from_labels(vals, preds_all_one)

    print("\nBaseline metrics:")
    print("  Predict all-zero (never delayed):")
    print(f"    Accuracy  = {m_zero['accuracy']:.4f}")
    print(f"    Precision = {m_zero['precision']:.4f}")
    print(f"    Recall    = {m_zero['recall']:.4f}")
    print(f"    F1        = {m_zero['f1']:.4f}")

    print("  Predict all-one (always delayed):")
    print(f"    Accuracy  = {m_one['accuracy']:.4f}")
    print(f"    Precision = {m_one['precision']:.4f}")
    print(f"    Recall    = {m_one['recall']:.4f}")
    print(f"    F1        = {m_one['f1']:.4f}")

    # Predict majority class (simple classifier)
    majority = 1 if pos > neg else 0
    preds_majority = np.full_like(vals, fill_value=majority)
    m_major = _metrics_from_labels(vals, preds_majority)
    print("\nPredict majority class:")
    print(f"  majority class = {majority} (fraction {max(pos_frac, 1-pos_frac):.3f})")
    print(f"    Accuracy  = {m_major['accuracy']:.4f}")
    print(f"    Precision = {m_major['precision']:.4f}")
    print(f"    Recall    = {m_major['recall']:.4f}")
    print(f"    F1        = {m_major['f1']:.4f}")

    # Random predictor (Bernoulli using pos fraction) - empirical estimate
    rng = np.random.RandomState(0)
    trials = 1000
    accs = []
    for _ in range(trials):
        rpred = rng.binomial(1, pos_frac, size=total)
        accs.append(float((rpred == vals).mean()))
    print(f"\nRandom Bernoulli predictor (p=pos_frac) - empirical accuracy ~ {np.mean(accs):.4f} (stdev {np.std(accs):.4f})")

    # If ARR_DELAY present, show AUC of ARR_DELAY as a scoring function
    if "ARR_DELAY" in df.columns:
        try:
            from sklearn.metrics import roc_auc_score
            scores = pd.to_numeric(df["ARR_DELAY"], errors='coerce').fillna(-1).values.astype(float)
            # Use raw ARR_DELAY as a score (higher -> more likely delayed)
            auc = roc_auc_score(vals, scores)
            print(f"\nUsing ARR_DELAY as a ranking score -> ROC AUC = {auc:.4f}")
        except Exception:
            pass

    # Plot counts and optional ARR_DELAY hist by class
    if SAVE_PLOT:
        os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))

        # Top: bar counts
        axes[0].bar([0, 1], [neg, pos], color=['C2', 'C3'], alpha=0.8)
        axes[0].set_xticks([0, 1])
        axes[0].set_xticklabels(['No delay (<15m)', 'Delay >=15m'])
        axes[0].set_ylabel('Count')
        axes[0].set_title('ARR_DEL15 class counts')

        # Bottom: ARR_DELAY histograms if available
        if "ARR_DELAY" in df.columns:
            arr = pd.to_numeric(df["ARR_DELAY"], errors='coerce')
            arr_pos = arr[vals == 1].dropna()
            arr_neg = arr[vals == 0].dropna()
            bins = 80
            axes[1].hist([arr_neg, arr_pos], bins=bins, color=['C2', 'C3'], alpha=0.6, label=['no delay', 'delay'], stacked=False, density=False)
            axes[1].set_xlim(left=-5, right=max(60, np.percentile(arr.dropna(), 95)))
            axes[1].set_xlabel('ARR_DELAY (minutes)')
            axes[1].legend()
            axes[1].set_title('ARR_DELAY distribution by class')
        else:
            axes[1].text(0.5, 0.5, 'ARR_DELAY not available for histogram', ha='center', va='center')
            axes[1].set_axis_off()

        plt.tight_layout()
        try:
            fig.savefig(PLOT_PATH)
            print(f"Saved distribution plot to {PLOT_PATH}")
        except Exception as e:
            print(f"Failed to save plot: {e}")
        plt.close(fig)


if __name__ == "__main__":
    main()
