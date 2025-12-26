import os
import pandas as pd
import numpy as np
from .splitter import split_file_to_list
from src.utils import hhmm_to_minutes


def load_data(
    path: str,
    task_type: str = "regression",
    max_rows: int | None = None,
    normalize_cols_file: str | None = None,
    normalize: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Load and preprocess flight data.
    
    Data splitting and normalization are done after graph building for flexibility.
    
    Args:
        path: Path to CSV file
        task_type: 'regression' or 'classification'
        max_rows: Optional row limit for testing
        normalize_cols_file: Path to file listing columns to normalize (deferred)
        normalize: Whether to enable normalization (deferred)
    
    Returns:
        df: Full dataframe sorted by time
        norm_config: Dict with normalization configuration for later use
    """
    
    if max_rows and max_rows > 0:
        df = pd.read_csv(path, nrows=int(max_rows))
        df = df.reset_index(drop=True)
        print(f"Row cap enabled: reading first {int(max_rows)} rows.")
    else:
        df = pd.read_csv(path)

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
    
    print(f"Dataset: {len(df)} flights from {df['dep_timestamp'].min()} to {df['dep_timestamp'].max()}")
    
    # Store normalization config for later use after splitting
    norm_config = {
        "enabled": normalize,
        "cols_file": normalize_cols_file,
    }
    
    return df, norm_config


def compute_split_and_normalize(
    df: pd.DataFrame,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    norm_config: dict = None,
) -> tuple[dict, dict]:
    """Compute chronological train/val/test split and normalize using train statistics.
    
    This is called after graph building to allow flexible split ratios.
    
    Args:
        df: Full dataframe sorted by time
        split_ratios: (train, val, test) ratios
        norm_config: Dict with 'enabled' and 'cols_file' from load_data
    
    Returns:
        split_indices: Dict with train_idx, val_idx, test_idx numpy arrays
        norm_stats: Normalization statistics (mu, sigma per column)
    """
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
    if norm_config and norm_config.get("enabled") and norm_config.get("cols_file"):
        normalize_cols_file = norm_config["cols_file"]
        if os.path.isfile(normalize_cols_file):
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
                    print("Normalization complete.")
            except Exception as e:
                print(f"Warning: normalization failed: {e}")
    
    split_indices = {
        'train_idx': train_indices,
        'val_idx': val_indices,
        'test_idx': test_indices,
    }

    return split_indices, norm_stats


def compute_splits_from_graph(graph, split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)):
    """Compute chronological train/val/test split directly from graph and update masks.
    
    This uses the graph's flight timestamps to compute splits, making it independent
    of the original dataframe. Works with both newly built and loaded graphs.
    
    Args:
        graph: HeteroData graph with flight.timestamp
        split_ratios: (train, val, test) ratios
    
    Returns:
        split_indices: Dict with train_idx, val_idx, test_idx numpy arrays
    """
    import torch
    import numpy as np
    
    # Get number of flights from graph
    n_flights = graph["flight"].x.size(0)
    
    # Compute split points chronologically (timestamps are already sorted since data was sorted)
    train_end = int(n_flights * split_ratios[0])
    val_end = int(n_flights * (split_ratios[0] + split_ratios[1]))
    
    train_indices = np.arange(0, train_end)
    val_indices = np.arange(train_end, val_end)
    test_indices = np.arange(val_end, n_flights)
    
    print(f"\nComputing splits from graph ({n_flights} flights):")
    print(f"  Train: {len(train_indices)} ({split_ratios[0]*100:.0f}%)")
    print(f"  Val: {len(val_indices)} ({split_ratios[1]*100:.0f}%)")
    print(f"  Test: {len(test_indices)} ({split_ratios[2]*100:.0f}%)")
    
    # Update graph masks
    train_mask = torch.zeros(n_flights, dtype=torch.bool)
    train_mask[train_indices] = True
    graph["flight"].train_mask = train_mask
    
    val_mask = torch.zeros(n_flights, dtype=torch.bool)
    val_mask[val_indices] = True
    graph["flight"].val_mask = val_mask
    
    test_mask = torch.zeros(n_flights, dtype=torch.bool)
    test_mask[test_indices] = True
    graph["flight"].test_mask = test_mask
    
    split_indices = {
        'train_idx': train_indices,
        'val_idx': val_indices,
        'test_idx': test_indices,
    }
    
    return split_indices


def compute_windows_from_graph(graph, unit, learn_window, pred_window, window_stride):
    """
    Compute sliding window indices from graph timestamps after loading/building.
    
    Windows are computed across ALL flights, then assigned to splits based on
    where their prediction indices fall. This preserves the full temporal continuity.
    
    Args:
        graph: HeteroData graph with flight.timestamp and flight.{train/val/test}_mask
        unit: Time unit size in minutes
        learn_window: Number of units for learning window
        pred_window: Number of units for prediction window
        window_stride: Stride between windows in units
    
    Returns:
        window_splits: Dict with 'train'/'val'/'test' window lists
    """
    import torch
    import numpy as np
    
    # Get timestamps from graph. Prefer absolute minutes if available to preserve true span
    if hasattr(graph["flight"], "timestamp_min"):
        timestamps_min = graph["flight"].timestamp_min.cpu().numpy()
        # Compute unit index directly from absolute minutes
        time_units = (timestamps_min // unit).astype(int)
    else:
        # Fallback: use normalized [0,1] timestamp (may compress span to ~1 day)
        timestamps = graph["flight"].timestamp.cpu().numpy()
        min_timestamp = timestamps.min()
        # Approximate units using 24h scale (may undercount if only normalized is available)
        time_units = (((timestamps - min_timestamp) * 1440).astype(int) // unit)
    
    total_window_size = learn_window + pred_window
    min_unit = time_units.min()
    max_unit = time_units.max()
    
    print(f"\nRecomputing sliding windows from graph:")
    print(f"  Time unit: {unit} minutes")
    print(f"  Learn window: {learn_window} units ({learn_window * unit} minutes)")
    print(f"  Pred window: {pred_window} units ({pred_window * unit} minutes)")
    print(f"  Stride: {window_stride} units")
    print(f"  Total time units in graph: {max_unit + 1}")
    
    # Get split masks
    train_mask = graph["flight"].train_mask.cpu().numpy()
    val_mask = graph["flight"].val_mask.cpu().numpy()
    test_mask = graph["flight"].test_mask.cpu().numpy()
    
    # Generate windows across ALL data
    windows_all = []
    num_possible_windows = max(1, (max_unit - min_unit + 1 - total_window_size) // window_stride + 1)
    
    for i in range(num_possible_windows):
        window_start = min_unit + i * window_stride
        learn_end = window_start + learn_window
        pred_end = learn_end + pred_window
        
        if pred_end > max_unit + 1:
            break
        
        # Find all flights in learn and pred windows
        learn_mask = (time_units >= window_start) & (time_units < learn_end)
        pred_mask = (time_units >= learn_end) & (time_units < pred_end)
        
        learn_idx = np.where(learn_mask)[0]
        pred_idx = np.where(pred_mask)[0]
        
        if len(learn_idx) > 0 and len(pred_idx) > 0:
            # Assign window to split based on where MAJORITY of pred indices fall
            pred_in_train = np.sum(train_mask[pred_idx])
            pred_in_val = np.sum(val_mask[pred_idx])
            pred_in_test = np.sum(test_mask[pred_idx])
            
            # Assign to split with most pred nodes
            if pred_in_train >= pred_in_val and pred_in_train >= pred_in_test:
                split = 'train'
            elif pred_in_val >= pred_in_test:
                split = 'val'
            else:
                split = 'test'
            
            windows_all.append({
                'window_id': len(windows_all),
                'split': split,
                'learn_indices': learn_idx,
                'pred_indices': pred_idx,
                'time_range': (int(window_start), int(pred_end - 1)),
                'pred_start_unit': int(learn_end),
                'learn_count': int(len(learn_idx)),
                'pred_count': int(len(pred_idx)),
            })
    
    # Group windows by split
    train_windows = [w for w in windows_all if w['split'] == 'train']
    val_windows = [w for w in windows_all if w['split'] == 'val']
    test_windows = [w for w in windows_all if w['split'] == 'test']
    
    window_splits = {
        'train': train_windows,
        'val': val_windows,
        'test': test_windows,
    }
    
    print(f"  Generated {len(windows_all)} total windows across timeline")
    print(f"  Split assignment: train={len(train_windows)}, val={len(val_windows)}, test={len(test_windows)}")
    
    return window_splits
