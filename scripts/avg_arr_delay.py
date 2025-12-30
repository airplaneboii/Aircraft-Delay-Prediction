import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Simple script to compute average ARR_DELAY for a dataset
# Configure the variables below and run with: python scripts/avg_arr_delay.py

DATA_PATH = "data/datasets/12M_20251221_221451.csv"  # path to CSV file or directory
HEAD_ROWS = None  # set to an int to limit rows for quick checks, or None to read all
IGNORE_CANCELLED = True  # if True, drop rows where CANCELLED == 1
CLIP_NEGATIVE = True  # if True, clip negative ARR_DELAY to 0 before averaging
SAVE_PLOT = True  # save distribution plot to disk
PLOT_PATH = "scripts/arr_delay_distribution.png"


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


def main():
    df = load_csv(DATA_PATH, nrows=HEAD_ROWS)

    if IGNORE_CANCELLED and "CANCELLED" in df.columns:
        df = df[df["CANCELLED"] == 0]

    if "ARR_DELAY" not in df.columns:
        print("Column ARR_DELAY not found in dataset")
        return

    df["ARR_DELAY"] = pd.to_numeric(df["ARR_DELAY"], errors="coerce")
    if CLIP_NEGATIVE:
        df["ARR_DELAY"] = df["ARR_DELAY"].clip(lower=0)

    valid = df["ARR_DELAY"].dropna()
    count = len(valid)
    if count == 0:
        print("No valid ARR_DELAY values found.")
        return

    mean = valid.mean()
    median = valid.median()
    std = valid.std()

    print(f"ARR_DELAY stats from {DATA_PATH} (n={count}):")
    print(f"  mean   = {mean:.4f} minutes")
    print(f"  median = {median:.4f} minutes")
    print(f"  std    = {std:.4f} minutes")

    # Baseline predictions: always predict the mean, and always predict zero
    vals = valid.values.astype(float)

    # Helper metric functions
    def mse(y_true, y_pred):
        return float(((y_true - y_pred) ** 2).mean())

    def mae(y_true, y_pred):
        return float(np.abs(y_true - y_pred).mean())

    def rmse(y_true, y_pred):
        return float(np.sqrt(mse(y_true, y_pred)))

    def r2_score(y_true, y_pred):
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        if ss_tot == 0:
            return float("nan")
        return float(1.0 - ss_res / ss_tot)

    # Mean predictor
    mean_pred = mean
    mean_preds = np.full_like(vals, fill_value=mean_pred, dtype=float)
    mse_mean = mse(vals, mean_preds)
    mae_mean = mae(vals, mean_preds)
    rmse_mean = rmse(vals, mean_preds)
    r2_mean = r2_score(vals, mean_preds)

    # Zero predictor
    zero_preds = np.zeros_like(vals)
    mse_zero = mse(vals, zero_preds)
    mae_zero = mae(vals, zero_preds)
    rmse_zero = rmse(vals, zero_preds)
    r2_zero = r2_score(vals, zero_preds)

    print("\nBaseline metrics:")
    print("  Predict mean:")
    print(f"    MSE  = {mse_mean:.4f}")
    print(f"    MAE  = {mae_mean:.4f}")
    print(f"    RMSE = {rmse_mean:.4f}")
    print(f"    R2   = {r2_mean if not np.isnan(r2_mean) else 'N/A'}")
    print("  Predict zero:")
    print(f"    MSE  = {mse_zero:.4f}")
    print(f"    MAE  = {mae_zero:.4f}")
    print(f"    RMSE = {rmse_zero:.4f}")
    print(f"    R2   = {r2_zero if not np.isnan(r2_zero) else 'N/A'}")

    # Additional descriptive stats
    p10 = np.percentile(vals, 10)
    p25 = np.percentile(vals, 25)
    p50 = np.percentile(vals, 50)
    p75 = np.percentile(vals, 75)
    p90 = np.percentile(vals, 90)
    skew = float(pd.Series(vals).skew())

    print("\nAdditional stats:")
    print(f"  min    = {vals.min():.4f}")
    print(f"  10%    = {p10:.4f}")
    print(f"  25%    = {p25:.4f}")
    print(f"  median = {p50:.4f}")
    print(f"  75%    = {p75:.4f}")
    print(f"  90%    = {p90:.4f}")
    print(f"  max    = {vals.max():.4f}")
    print(f"  skew   = {skew:.4f}")

    # Plot distribution and boxplot
    if SAVE_PLOT:
        os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
        fig, axes = plt.subplots(
            2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]}
        )

        # Histogram + KDE (using pandas for KDE)
        # Prepare transformed values for log1p view
        log_vals = np.log1p(vals)

        # Compute focus limit (95th percentile) for the linear view
        try:
            p95 = float(np.percentile(vals, 95))
            p75 = float(np.percentile(vals, 75))
            xmax = float(max(p95 * 1.1, p75 * 2.0, mean * 3.0))
            if xmax <= 0 or not np.isfinite(xmax):
                xmax = float(np.max(vals))
        except Exception:
            xmax = float(np.max(vals))

        # Fraction zeros and percentiles for annotations
        frac_zero = float((vals == 0).sum()) / len(vals)
        p10 = float(np.percentile(vals, 10))
        p50 = float(np.percentile(vals, 50))
        p90 = float(np.percentile(vals, 90))

        # Layout: left = log1p histogram+KDE, right = linear histogram + boxplot
        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(
            2, 2, width_ratios=[1, 1], height_ratios=[3, 1], hspace=0.25, wspace=0.3
        )
        ax_log = fig.add_subplot(gs[:, 0])
        ax_lin = fig.add_subplot(gs[0, 1])
        ax_box = fig.add_subplot(gs[1, 1])

        # Left: log1p histogram + KDE
        ax_log.hist(log_vals, bins=80, density=True, alpha=0.6, color="C0")
        try:
            pd.Series(log_vals).plot.kde(ax=ax_log, color="C1")
        except Exception:
            pass
        ax_log.axvline(
            np.log1p(mean), color="k", linestyle="--", label=f"mean={mean:.2f} min"
        )
        ax_log.axvline(
            np.log1p(median), color="m", linestyle=":", label=f"median={median:.2f} min"
        )
        ax_log.set_title("ARR_DELAY (log1p scale)")
        ax_log.set_xlabel("log1p(ARR_DELAY)")
        ax_log.legend()

        # Right-top: linear histogram focused to 95th percentile region
        ax_lin.hist(
            vals, bins=80, range=(0, xmax), density=False, alpha=0.6, color="C2"
        )
        try:
            # overlay a KDE on the truncated range by re-evaluating values within range
            in_range = vals[vals <= xmax]
            if len(in_range) > 0:
                pd.Series(in_range).plot.kde(ax=ax_lin, color="C3")
        except Exception:
            pass
        ax_lin.axvline(mean, color="k", linestyle="--", label=f"mean={mean:.2f}")
        ax_lin.axvline(0.0, color="r", linestyle=":", label="zero")
        ax_lin.set_title(f"ARR_DELAY (0 to {int(xmax)} min, ~95% focus)")
        ax_lin.set_xlabel("ARR_DELAY (minutes)")
        ax_lin.legend()

        # Annotate percentiles and fraction at zero
        pct_text = f"n={len(vals):,}\n0s={frac_zero*100:.1f}%\n10%={p10:.0f}\nmedian={p50:.0f}\n90%={p90:.0f}"
        ax_lin.text(
            0.98,
            0.95,
            pct_text,
            transform=ax_lin.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.7),
        )

        # Right-bottom: boxplot aligned to same x-range
        ax_box.boxplot(vals, vert=False, widths=0.6)
        ax_box.set_xlabel("ARR_DELAY (minutes)")
        ax_box.set_yticks([])
        try:
            ax_box.set_xlim(left=0, right=xmax)
        except Exception:
            pass

        # Adjust x-axis to focus on main mass of the distribution
        try:
            p95 = np.percentile(vals, 95)
            p75 = np.percentile(vals, 75)
            xmax = float(max(p95 * 1.2, p75 * 2.0, mean * 3.0))
            if xmax <= 0 or not np.isfinite(xmax):
                xmax = float(np.max(vals))
        except Exception:
            xmax = float(np.max(vals))

        # Use linear x-axis focused on the main mass of the distribution
        try:
            axes[0].set_xlim(left=0, right=xmax)
        except Exception:
            pass

        # Boxplot (aligned to same x-range when linear)
        axes[1].boxplot(vals, vert=False, widths=0.6)
        axes[1].set_xlabel("ARR_DELAY (minutes)")
        axes[1].set_yticks([])
        try:
            axes[1].set_xlim(left=0, right=xmax)
        except Exception:
            pass

        plt.tight_layout()
        try:
            fig.savefig(PLOT_PATH)
            print(f"Saved distribution plot to {PLOT_PATH}")
        except Exception as e:
            print(f"Failed to save plot: {e}")
        plt.close(fig)


if __name__ == "__main__":
    main()
