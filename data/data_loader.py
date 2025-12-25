import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from .splitter import split_file_to_list
from src.utils import hhmm_to_minutes

# Suppress pandas FutureWarning about dtype assignment during normalization
# warnings.filterwarnings('ignore', category=FutureWarning, message='.*Setting an item of incompatible dtype.*')


def load_data(
    path: str,
    mode: str = "train",
    task_type: str = "regression",
    max_rows: int | None = None,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    normalize_cols_file: str | None = None,
    unit: int = 60,  # Time unit in minutes (default 1 hour)
    learn_window: int = 10,  # Length of learning window in units
    pred_window: int = 1,  # Length of prediction window in units
    window_stride: int = 1,  # Stride between windows in units
    normalize: bool = True,
) -> tuple[pd.DataFrame, dict, list, dict]:
    """Load data and prepare for sliding window temporal training.
    
    Returns:
        df: Full dataframe sorted by time
        norm_stats: Normalization statistics
        window_splits: Dict with keys 'train'/'val'/'test' -> list of window dicts
        split_indices: Dict with numpy arrays for chronological split indices: {"train_idx","val_idx","test_idx"}
    """
    
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

    if task_type == "regression":
        df["y"] = df["ARR_DELAY"] if "ARR_DELAY" in df.columns else 0
    else:
        df["y"] = df["ARR_DEL15"] if "ARR_DEL15" in df.columns else 0

    # Build timestamps
    if "FL_DATE" not in df.columns or not pd.api.types.is_datetime64_any_dtype(df["FL_DATE"]):
        df["FL_DATE"] = pd.to_datetime(
            df[["YEAR", "MONTH", "DAY_OF_MONTH"]].rename(columns={"YEAR": "year", "MONTH": "month", "DAY_OF_MONTH": "day"})
        )
    
    df["dep_minutes"] = df["CRS_DEP_TIME"].apply(hhmm_to_minutes)
    df["dep_timestamp"] = df["FL_DATE"] + pd.to_timedelta(df["dep_minutes"], unit="m")
    
    # Sort by departure timestamp
    df = df.sort_values("dep_timestamp").reset_index(drop=True)
    
    # Create time bins based on unit size
    min_time = df["dep_timestamp"].min()
    df["time_unit"] = ((df["dep_timestamp"] - min_time).dt.total_seconds() / (unit * 60)).astype(int)
    
    print(f"Time unit: {unit} minutes")
    print(f"Total time units: {df['time_unit'].max() + 1}")
    print(f"Dataset spans: {df['dep_timestamp'].min()} to {df['dep_timestamp'].max()}")
    
    # Split chronologically into train/val/test
    n = len(df)
    train_end = int(n * split_ratios[0])
    val_end = int(n * (split_ratios[0] + split_ratios[1]))
    
    train_indices = np.arange(0, train_end)
    val_indices = np.arange(train_end, val_end)
    test_indices = np.arange(val_end, n)
    
    print(f"\nChronological split: train={len(train_indices)} ({split_ratios[0]*100:.0f}%), "
          f"val={len(val_indices)} ({split_ratios[1]*100:.0f}%), "
          f"test={len(test_indices)} ({split_ratios[2]*100:.0f}%)")
    
    # Normalize using train statistics
    norm_stats = {"mu": {}, "sigma": {}}
    if normalize and normalize_cols_file and os.path.isfile(normalize_cols_file):
        try:
            requested = split_file_to_list(normalize_cols_file)
            num_cols = [c for c in requested if c in df.columns]
            
            if num_cols:
                print(f"Normalizing {len(num_cols)} columns using train split statistics")
                df[num_cols] = df[num_cols].astype('float32')
                
                # Compute stats from train portion only
                train_slice = df.loc[train_indices, num_cols]
                mu = train_slice.mean()
                sigma = train_slice.std(ddof=0)
                sigma_safe = sigma.replace({0: 1.0})
                
                # Apply to entire dataframe
                df.loc[:, num_cols] = (df[num_cols] - mu) / sigma_safe
                
                norm_stats["mu"] = {c: float(mu[c]) for c in num_cols}
                norm_stats["sigma"] = {c: float(sigma_safe[c]) for c in num_cols}
                # Print all normalization statistics for visibility
                print("Normalization complete. Per-column statistics (μ, σ):")
                for c in num_cols:
                    mu_val = norm_stats["mu"].get(c, "N/A")
                    sigma_val = norm_stats["sigma"].get(c, "N/A")
                    if isinstance(mu_val, (int, float)) and isinstance(sigma_val, (int, float)):
                        print(f"  {c}: μ={mu_val:.4f}, σ={sigma_val:.4f}")
                    else:
                        print(f"  {c}: μ={mu_val}, σ={sigma_val}")
        except Exception as e:
            print(f"Warning: normalization failed: {e}")
    
    # Store split masks on dataframe
    df["split"] = "test"  # default
    df.loc[train_indices, "split"] = "train"
    df.loc[val_indices, "split"] = "val"
    
    # Create sliding windows for all splits (train, val, test)
    total_window_size = learn_window + pred_window
    
    print(f"\nSliding window configuration:")
    print(f"  Learn window: {learn_window} units ({learn_window * unit} minutes)")
    print(f"  Pred window: {pred_window} units ({pred_window * unit} minutes)")
    print(f"  Total window: {total_window_size} units ({total_window_size * unit} minutes)")
    print(f"  Stride: {window_stride} units ({window_stride * unit} minutes)")
    
    def generate_windows_for_split(split_indices, split_name):
        """Generate sliding windows for a given split."""
        if len(split_indices) == 0:
            return []
        
        split_df = df.iloc[split_indices].copy()
        split_time_units = split_df["time_unit"].values
        
        min_unit = split_time_units.min()
        max_unit = split_time_units.max()
        
        windows = []
        num_possible_windows = max(1, (max_unit - min_unit + 1 - total_window_size) // window_stride + 1)
        
        for i in range(num_possible_windows):
            window_start = min_unit + i * window_stride
            learn_end = window_start + learn_window
            pred_end = learn_end + pred_window
            
            if pred_end > max_unit + 1:
                break
            
            # Get indices for learn and pred windows
            learn_mask = (split_time_units >= window_start) & (split_time_units < learn_end)
            pred_mask = (split_time_units >= learn_end) & (split_time_units < pred_end)
            
            learn_idx = split_indices[learn_mask]
            pred_idx = split_indices[pred_mask]
            
            if len(learn_idx) > 0 and len(pred_idx) > 0:
                windows.append({
                    'window_id': len(windows),
                    'split': split_name,
                    'learn_indices': learn_idx,
                    'pred_indices': pred_idx,
                    'time_range': (window_start, pred_end - 1),
                    'learn_count': len(learn_idx),
                    'pred_count': len(pred_idx),
                })
        
        return windows
    
    # Generate windows for each split
    train_windows = generate_windows_for_split(train_indices, 'train')
    val_windows = generate_windows_for_split(val_indices, 'val')
    test_windows = generate_windows_for_split(test_indices, 'test')
    
    # Collect all windows
    window_splits = {
        'train': train_windows,
        'val': val_windows,
        'test': test_windows,
    }
    
    print(f"Generated sliding windows:")
    print(f"  Train: {len(train_windows)} windows")
    if train_windows:
        print(f"    First: units [{train_windows[0]['time_range'][0]}, {train_windows[0]['time_range'][1]}], "
              f"learn={train_windows[0]['learn_count']}, pred={train_windows[0]['pred_count']}")
        print(f"    Last: units [{train_windows[-1]['time_range'][0]}, {train_windows[-1]['time_range'][1]}], "
              f"learn={train_windows[-1]['learn_count']}, pred={train_windows[-1]['pred_count']}")
    print(f"  Val: {len(val_windows)} windows")
    print(f"  Test: {len(test_windows)} windows")
    
    split_indices = {
        'train_idx': train_indices,
        'val_idx': val_indices,
        'test_idx': test_indices,
    }

    return df, norm_stats, window_splits, split_indices
