import pandas as pd

def load_data(
        path: str,
        mode: str = "train",
        task_type: str = "regression",
        development: bool = True,
        ) -> pd.DataFrame:
    """Load the merged CSV into a DataFrame.

    Note: Data cleaning (dropping rows with missing essential fields and filling non-essential numeric NaNs) 
    should be performed during merging to save time when running multiple tests.
    This loader only reads the CSV and applies a development row cap.
    Eventually some data processing and conversion that can't be stored in a CSV can be done here.
    """

    # Read CSV
    df = pd.read_csv(path)

    # For development only
    if development:
        max_rows = 500000
        df = df.head(max_rows).reset_index(drop=True)
        print(f"Development mode: using first {max_rows} rows.")

    # Create target column based on task type, but do not perform cleaning here.
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

    # Split data (placeholder)
    if mode == "train":
        print("Loaded data in train mode.")
    elif mode == "test":
        print("Loaded data in test mode.")

    return df
