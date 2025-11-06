import pandas as pd
import torch

def load_data(
        path: str,
        mode: str = "train",
        task_type: str = "regression",
        development: bool = True,
        ) -> pd.DataFrame:
    
    # Read CSV
    df = pd.read_csv(path)

    # For development only
    if development:
        max_rows = 500000
        df = df.head(max_rows).reset_index(drop=True)
        print(f"Development mode: using first {max_rows} rows.")

    # Remove rows with missing essential data
    df = df.dropna(subset=["ORIGIN", "DEST", "TAIL_NUM"])
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    df = df.dropna(subset=["FL_DATE"])

    # Target variable
    if task_type == "regression":
        df["y"] = df["ARR_DELAY"].fillna(0)
    else:
        df["y"] = df["ARR_DEL15"].fillna(0)

    df["y"] = df["y"].fillna(0)

    # Split data
    # TODO: Implement proper splitting
    if mode == "train":
        print("Not implemented: train mode data filtering.")
    elif mode == "test":
        print("Not implemented: test mode data filtering.")

    return df
