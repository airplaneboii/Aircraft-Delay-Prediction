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
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load the merged CSV into a DataFrame.

    Note: Data cleaning (dropping rows with missing essential fields and filling non-essential numeric NaNs) 
    should be performed during merging to save time when running multiple tests.
    This loader only reads the CSV and applies a development row cap.
    Eventually some data processing and conversion that can't be stored in a CSV can be done here.
    """

    # Read CSV
    df = pd.read_csv(path)

    # Optional row cap for quicker testing
    if max_rows and max_rows > 0:
        df = df.head(int(max_rows)).reset_index(drop=True)
        print(f"Row cap enabled: using first {int(max_rows)} rows.")

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

    # --- Normalization: compute mu/sigma on selected columns and normalize dataset ---
    # Determine which columns to normalize:
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
        mu = df[num_cols].mean()
        sigma = df[num_cols].std(ddof=0)
        # avoid zero std
        sigma_safe = sigma.replace({0: 1.0})
        # apply normalization in-place
        df[num_cols] = (df[num_cols] - mu) / sigma_safe
        # store stats as plain floats for JSON-compatibility
        norm_stats["mu"] = {c: float(mu[c]) for c in num_cols}
        norm_stats["sigma"] = {c: float(sigma_safe[c]) for c in num_cols}
        # Inform which columns were normalized
        print(f"Normalized columns: {', '.join(num_cols)}")
    else:
        norm_stats = {"mu": {}, "sigma": {}}
        print("No normalization applied (no valid columns listed in normalize.txt)")

    # Split data
    num_data = len(df)
    i_train = int(num_data * split_dim[0] / 100)
    i_val = int(num_data * (split_dim[0] + split_dim[1]) / 100)

    df_train = df.iloc[:i_train].reset_index(drop=True)
    df_val = df.iloc[i_train:i_val].reset_index(drop=True)
    df_test = df.iloc[i_val:].reset_index(drop=True)

#    # for now
#    dates = df["FL_DATE"].unique().sort_values()
#    dates_train = dates[:int(len(dates) * split_dim[0] / 100)]
#    dates_val = dates[int(len(dates) * split_dim[0] / 100):int(len(dates) * (split_dim[0] + split_dim[1]) / 100)]
#    dates_test = dates[int(len(dates) * (split_dim[0] + split_dim[1]) / 100):]
#
#    set_train = df["FL_DATE"].isin(dates_train)
#    set_val = df["FL_DATE"].isin(dates_val)
#    set_test = df["FL_DATE"].isin(dates_test)

    return df_train, df_val, df_test, norm_stats
