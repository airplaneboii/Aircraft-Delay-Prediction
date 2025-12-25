import os
import pandas as pd
import numpy as np
from .splitter import split_file_to_list

def load_data(
    path: str,
    mode: str = "train",
    task_type: str = "regression",
    max_rows: int | None = None,
    split_dim: tuple[int, int, int] = (80, 10, 10),
    normalize_cols_file: str | None = None,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load the merged CSV into a DataFrame.

    Note: Data cleaning (dropping rows with missing essential fields and filling non-essential numeric NaNs) 
    should be performed during merging to save time when running multiple tests.
    This loader only reads the CSV and applies a development row cap.
    Eventually some data processing and conversion that can't be stored in a CSV can be done here.

    Returns:
        (df, train_index, val_index, test_index, norm_stats)
    """

    # Read CSV
    df = pd.read_csv(path)

    # Optional row cap for quicker testing
    if max_rows and max_rows > 0:
        df = df.head(int(max_rows)).reset_index(drop=True)
        print(f"Row cap enabled: using first {int(max_rows)} rows.")

    # Remove cancelled flights for delay prediction
    if "CANCELLED" in df.columns:
        df = df[df["CANCELLED"] == 0].reset_index(drop=True)

    # Ensure arrival delays are non-negative: set negative ARR_DELAY to 0
    if "ARR_DELAY" in df.columns:
        df["ARR_DELAY"] = pd.to_numeric(df["ARR_DELAY"], errors="coerce")
        df["ARR_DELAY"] = df["ARR_DELAY"].clip(lower=0).fillna(0)
    # Create target column based on task type
    if task_type == "regression":
        if "ARR_DELAY" in df.columns:
            df["y"] = df["ARR_DELAY"]
        else:
            df["y"] = 0
    else:
        if "ARR_DEL15" in df.columns:
            df["y"] = df["ARR_DEL15"]
        else:
            df["y"] = 0

    # Split data
    num_data = len(df)
    i_train = int(num_data * split_dim[0] / 100)
    i_val = int(num_data * (split_dim[0] + split_dim[1]) / 100)

    train_index = np.arange(0, i_train)
    val_index = np.arange(i_train, i_val)
    test_index = np.arange(i_val, num_data)

    # --- Normalization: compute mu/sigma on TRAIN split only, then apply to all splits ---
    # Avoids data leakage by fitting statistics on train and reusing for val/test.
    # - If a normalize_cols_file is provided and exists, use the listed columns (one per line).
    # - If the file is missing or empty, do NOT normalize anything (user opted out).
    num_cols = []
    if normalize_cols_file and os.path.isfile(normalize_cols_file):
        try:
            requested = split_file_to_list(normalize_cols_file)
            # Keep only columns that exist in df
            num_cols = [c for c in requested if c in df.columns]
        except Exception:
            num_cols = []

    norm_stats = {"mu": {}, "sigma": {}}
    if num_cols:
        # Fit statistics on training subset only
        train_slice = df.loc[train_index, num_cols]
        mu = train_slice.mean()
        sigma = train_slice.std(ddof=0)
        sigma_safe = sigma.replace({0: 1.0})  # guard against zero variance

        # Apply train-derived stats to all rows (train/val/test)
        df[num_cols] = (df[num_cols] - mu) / sigma_safe

        # Store stats as plain floats for JSON-compatibility
        norm_stats["mu"] = {c: float(mu[c]) for c in num_cols}
        norm_stats["sigma"] = {c: float(sigma_safe[c]) for c in num_cols}

        print(f"Normalized columns (fit on train): {', '.join(num_cols)}")
    else:
        norm_stats = {"mu": {}, "sigma": {}}
        print("No normalization applied (no valid columns listed in normalize.txt)")



#    # for now
#    dates = df["FL_DATE"].unique().sort_values()
#    dates_train = dates[:int(len(dates) * split_dim[0] / 100)]
#    dates_val = dates[int(len(dates) * split_dim[0] / 100):int(len(dates) * (split_dim[0] + split_dim[1]) / 100)]
#    dates_test = dates[int(len(dates) * (split_dim[0] + split_dim[1]) / 100):]
#
#    set_train = df["FL_DATE"].isin(dates_train)
#    set_val = df["FL_DATE"].isin(dates_val)
#    set_test = df["FL_DATE"].isin(dates_test)

    return df, train_index, val_index, test_index, norm_stats
