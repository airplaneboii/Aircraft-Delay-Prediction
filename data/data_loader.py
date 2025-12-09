import pandas as pd

def load_data(
        path: str,
        mode: str = "train",
        task_type: str = "regression",
        development: bool = True,
        split_dim: tuple[int, int, int] = (80, 10, 10)
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

    return df_train, df_val, df_test
