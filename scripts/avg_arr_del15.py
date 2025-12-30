import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

# Default split ratios (train, val, test)
SPLIT_RATIOS = (0.8, 0.1, 0.1)
# Default random seed for reproducible splits
RANDOM_SEED = 42

# Simple script to compute classification baselines for ARR_DEL15
# Configure the variables below and run with: python scripts/avg_arr_del15.py

DATA_PATH = "data/datasets/1Y_20251221_221451.csv"  # path to CSV file or directory
HEAD_ROWS = None  # set to an int to limit rows for quick checks, or None to read all
IGNORE_CANCELLED = True  # if True, drop rows where CANCELLED == 1
SAVE_PLOT = True  # save distribution plot to disk
PLOT_PATH = "scripts/arr_del15_distribution.png"


def load_csv(path, nrows=None):
    if os.path.isdir(path):
        # pick latest CSV in directory
        files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith(".csv")
        ]
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
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute ARR_DEL15 stats and per-split classification baselines")
    parser.add_argument("--data-path", default=DATA_PATH, help="Path to CSV file or directory")
    parser.add_argument("--head-rows", type=int, default=HEAD_ROWS, help="Limit rows read for quick checks")
    parser.add_argument(
        "--ignore-cancelled",
        dest="ignore_cancelled",
        action="store_true",
        default=IGNORE_CANCELLED,
        help="Ignore cancelled flights (requires CANCELLED column)",
    )
    parser.add_argument(
        "--save-plot",
        dest="save_plot",
        action="store_true",
        default=SAVE_PLOT,
        help="Save distribution plot to disk",
    )
    parser.add_argument("--plot-path", default=PLOT_PATH, help="Path to save plot")
    parser.add_argument(
        "--splits",
        nargs=3,
        type=float,
        default=list(SPLIT_RATIOS),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Split ratios (fractions summing to 1 or percentages summing to 100). Default 80/10/10.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for shuffling before splitting")

    args = parser.parse_args()

    # One-shot script — use simple prints instead of configuring logging
    df = load_csv(args.data_path, nrows=args.head_rows)

    if args.ignore_cancelled and "CANCELLED" in df.columns:
        df = df[df["CANCELLED"] == 0]

    # Prefer an existing ARR_DEL15 column; if missing try to build from ARR_DELAY
    if "ARR_DEL15" not in df.columns:
        if "ARR_DELAY" in df.columns:
            df["ARR_DEL15"] = (
                pd.to_numeric(df["ARR_DELAY"], errors="coerce") >= 15
            ).astype(int)
            print("Column ARR_DEL15 not found; deriving from ARR_DELAY >= 15")
        else:
            print(
                "Column ARR_DEL15 not found in dataset and ARR_DELAY is not available to derive it."
            )
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

    print(f"ARR_DEL15 stats from {args.data_path} (n={total}):")
    print(f"  positives = {pos} ({pos_frac*100:.2f}%)")
    print(f"  negatives = {neg} ({100 - pos_frac*100:.2f}%)")

    # Global baselines (computed on full dataset)
    preds_all_zero = np.zeros_like(vals)
    preds_all_one = np.ones_like(vals)

    m_zero = _metrics_from_labels(vals, preds_all_zero)
    m_one = _metrics_from_labels(vals, preds_all_one)

    print("\nGlobal baseline metrics:")
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

    # Predict majority class (on full data)
    majority = 1 if pos > neg else 0
    preds_majority = np.full_like(vals, fill_value=majority)
    m_major = _metrics_from_labels(vals, preds_majority)
    print("\nPredict majority class (on full data):")
    print(f"  majority class = {majority} (fraction {max(pos_frac, 1-pos_frac):.3f})")
    print(f"    Accuracy  = {m_major['accuracy']:.4f}")
    print(f"    Precision = {m_major['precision']:.4f}")
    print(f"    Recall    = {m_major['recall']:.4f}")
    print(f"    F1        = {m_major['f1']:.4f}")

    # Random predictor (Bernoulli using pos fraction) - empirical estimate on full data
    rng = np.random.RandomState(args.seed)
    trials = 1000
    accs = []
    for _ in range(trials):
        rpred = rng.binomial(1, pos_frac, size=total)
        accs.append(float((rpred == vals).mean()))
    print(
        f"\nRandom Bernoulli predictor (p=pos_frac) - empirical accuracy ~ {np.mean(accs):.4f} (stdev {np.std(accs):.4f})"
    )

    # If ARR_DELAY present, show AUC of ARR_DELAY as a scoring function
    if "ARR_DELAY" in df.columns:
        try:
            from sklearn.metrics import roc_auc_score

            scores = (
                pd.to_numeric(df["ARR_DELAY"], errors="coerce")
                .fillna(-1)
                .values.astype(float)
            )
            # Use raw ARR_DELAY as a score (higher -> more likely delayed)
            auc = roc_auc_score(vals, scores)
            print(f"\nUsing ARR_DELAY as a ranking score -> ROC AUC = {auc:.4f}")
        except Exception:
            pass

    # --- Splitting into train/val/test and evaluating baselines ---
    splits = np.array(args.splits, dtype=float)
    if splits.sum() > 1.0 + 1e-6:
        # allow percentages (e.g., 80 10 10) or unnormalized inputs
        splits = splits / splits.sum()
    else:
        # normalize small floating rounding errors
        splits = splits / max(splits.sum(), 1.0)

    n = len(vals)
    indices = np.arange(n)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(indices)

    train_n = int(round(splits[0] * n))
    val_n = int(round(splits[1] * n))
    # Ensure sensible allocation (rest goes to test)
    if train_n < 1 and n >= 1:
        train_n = 1
    if val_n < 0:
        val_n = 0

    train_idx = indices[:train_n]
    val_idx = indices[train_n: train_n + val_n]
    test_idx = indices[train_n + val_n :]

    train_vals = vals[train_idx]
    val_vals = vals[val_idx]
    test_vals = vals[test_idx]

    print(f"\nData splits (train/val/test): {splits[0]:.3f}/{splits[1]:.3f}/{splits[2]:.3f}  n={n:,}")
    print(f"  counts -> train={len(train_vals):,}, val={len(val_vals):,}, test={len(test_vals):,}")

    # Evaluate classification baselines using training-derived parameters
    def eval_class_baselines(y_true, train_majority, train_pos_frac, rng_seed=None):
        metrics = {}
        if len(y_true) == 0:
            return metrics
        # Predict train majority
        preds = np.full_like(y_true, fill_value=train_majority)
        metrics['train_majority'] = _metrics_from_labels(y_true, preds)
        # Empirical random Bernoulli accuracy (using same seed for reproducibility if provided)
        rng_local = np.random.RandomState(rng_seed if rng_seed is not None else 0)
        trials = 500
        accs_local = []
        for _ in range(trials):
            rpred = rng_local.binomial(1, train_pos_frac, size=len(y_true))
            accs_local.append(float((rpred == y_true).mean()))
        metrics['rand_mean_acc'] = float(np.mean(accs_local))
        metrics['rand_std_acc'] = float(np.std(accs_local))
        return metrics

    train_pos_frac = float((train_vals == 1).sum()) / max(1, len(train_vals))
    train_major = 1 if (train_vals == 1).sum() > (train_vals == 0).sum() else 0

    for name, arr in [("train", train_vals), ("val", val_vals), ("test", test_vals)]:
        print(f"\n{name.capitalize()} (n={len(arr):,}):")
        if len(arr) == 0:
            print("  (no samples)")
            continue
        pos_c = int((arr == 1).sum())
        print(f"  positives = {pos_c} ({pos_c/len(arr)*100:.2f}%)")
        print(f"  negatives = {int((arr==0).sum())}")

        metrics = eval_class_baselines(arr, train_major, train_pos_frac, rng_seed=args.seed)
        if 'train_majority' in metrics:
            m = metrics['train_majority']
            print("  Baseline using train majority class (predict train majority):")
            print(f"    Accuracy  = {m['accuracy']:.4f}")
            print(f"    Precision = {m['precision']:.4f}")
            print(f"    Recall    = {m['recall']:.4f}")
            print(f"    F1        = {m['f1']:.4f}")
            print(f"  Random Bernoulli (p=train_pos_frac={train_pos_frac:.3f}): emp acc~{metrics['rand_mean_acc']:.4f} ± {metrics['rand_std_acc']:.4f}")

    # Plot counts and a more informative visualization (wide layout for readability)
    if args.save_plot:
        os.makedirs(os.path.dirname(args.plot_path), exist_ok=True)
        # ROC/PR plotting removed per user request (sklearn imports not required) 

        # Use a wider layout: main column for counts+distribution, side column for ROC and summary
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 2], wspace=0.28, hspace=0.25)

        # Top-left: counts
        ax_counts = fig.add_subplot(gs[0, 0])
        ax_counts.bar([0, 1], [neg, pos], color=["C2", "C3"], alpha=0.9)
        ax_counts.set_xticks([0, 1])
        ax_counts.set_xticklabels(["No delay (<15m)", "Delay >=15m"], fontsize=9)
        ax_counts.set_ylabel("Count")
        ax_counts.set_title("ARR_DEL15 class counts")
        for i, v in enumerate([neg, pos]):
            pct = v / total * 100
            ax_counts.text(i, v + max(1, total * 0.01), f"{v:,}\n{pct:.1f}%", ha="center", va="bottom", fontsize=9)

        # Top-right: empty area to reduce visual clutter
        ax_top = fig.add_subplot(gs[0, 1])
        ax_top.axis("off")

        # Bottom-left: long distribution with KDE and boxplot inset
        ax_dist = fig.add_subplot(gs[1, 0])
        if "ARR_DELAY" in df.columns:
            arr = pd.to_numeric(df["ARR_DELAY"], errors="coerce").fillna(np.nan)
            arr_pos = arr[vals == 1].dropna()
            arr_neg = arr[vals == 0].dropna()
            bins = 120
            ax_dist.hist(
                [arr_neg, arr_pos],
                bins=bins,
                color=["C2", "C3"],
                alpha=0.45,
                label=["no delay", "delay"],
                density=True,
            )
            # KDE overlays (pandas for robustness)
            try:
                if len(arr_neg) > 1:
                    pd.Series(arr_neg).plot.kde(ax=ax_dist, color="C2", lw=1)
                if len(arr_pos) > 1:
                    pd.Series(arr_pos).plot.kde(ax=ax_dist, color="C3", lw=1)
            except Exception:
                pass
            ax_dist.set_xlim(left=0, right=max(60, np.nanpercentile(arr, 95)))
            ax_dist.set_xlabel("ARR_DELAY (minutes)")
            ax_dist.set_title("ARR_DELAY density by class (KDE + histogram)")
            ax_dist.legend(fontsize=9)

            # Wider inset boxplot to show spread
            ax_box = fig.add_axes([0.66, 0.18, 0.26, 0.12])
            ax_box.boxplot([arr_neg, arr_pos], vert=False, widths=0.6)
            ax_box.set_yticklabels(["no delay", "delay"], fontsize=8)
            ax_box.set_xlabel("")
            ax_box.set_xticks([])
        else:
            ax_dist.text(0.5, 0.5, "ARR_DELAY not available for distribution", ha="center", va="center")
            ax_dist.set_axis_off()

        # Bottom-right: summary (compact)
        ax_summary = fig.add_subplot(gs[1, 1])
        ax_summary.axis("off")

        # Summary textbox with baseline metrics (placed in main axes area)
        summary_lines = [
            f"Totals: n={total:,}, pos={pos:,} ({pos_frac*100:.1f}%)",
            f"All-zero: acc={m_zero['accuracy']:.3f}, f1={m_zero['f1']:.3f}",
            f"All-one:  acc={m_one['accuracy']:.3f}, f1={m_one['f1']:.3f}",
            f"Majority:  acc={m_major['accuracy']:.3f}, f1={m_major['f1']:.3f}",
            f"Random (emp): acc~{np.mean(accs):.3f} ± {np.std(accs):.3f}",
        ]
        summary_text = "\n".join(summary_lines)
        ax_summary.text(0.01, 0.98, summary_text, fontsize=10, va="top", family="monospace")

        # Save with higher DPI and tight bounding box to avoid clipping
        plt.tight_layout()
        try:
            fig.savefig(args.plot_path, dpi=180, bbox_inches="tight")
            print(f"Saved distribution plot to {args.plot_path}")
        except Exception as e:
            print(f"Failed to save plot: {e}")
        plt.close(fig)


if __name__ == "__main__":
    main()
