import pandas as pd
import torch

def load_data(
        path: str,
        mode: str = "develop",
        task_type: str = "regression",
        ) -> pd.DataFrame:
    
    # Read CSV
    df = pd.read_csv(path)

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
