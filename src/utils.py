import logging
import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from torch_geometric.data import HeteroData

"""Utility helpers for logging, graph operations, normalization, metrics, and system stats."""

try:
    from torch_geometric.profile.utils import get_data_size  # available in recent PyG
except Exception:
    get_data_size = None

try:
    import psutil
except Exception:
    psutil = None


def ensure_dir(directory: str) -> None:
    """Create directory `directory` if it does not exist.

    This helper is tolerant of existing directories and avoids raising an
    exception when the directory already exists.
    """
    os.makedirs(directory, exist_ok=True)


# Logging setup
def setup_logging(verbosity: int = 0, logfile: Optional[str] = None) -> logging.Logger:
    """Configure a single named logger for the application.

    This avoids repeated calls to basicConfig and provides a consistent
    `train` logger across modules. If the logger already has handlers we
    only update its level (idempotent), which is useful for tests.

    verbosity: 0=WARNING, 1=INFO (INFO logs go to stdout), 2=DEBUG
    """
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO

    logger = logging.getLogger("train")

    # If the logger already configured, just set the level and adjust stream handlers
    if logger.handlers:
        logger.setLevel(level)
        # If user requested INFO, ensure StreamHandlers use stdout instead of stderr
        if level == logging.INFO:
            for h in logger.handlers:
                if isinstance(h, logging.StreamHandler):
                    try:
                        if getattr(h, "stream", None) is sys.stderr:
                            h.stream = sys.stdout
                    except Exception:
                        pass
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"
    )

    # Send INFO-level console output to stdout (instead of default stderr).
    stream_target = sys.stdout if level == logging.INFO else None
    stream_handler = logging.StreamHandler(stream=stream_target)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if logfile:
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent double logging in interactive environments
    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    """Return the configured top-level application logger (name: 'train')."""
    return logging.getLogger("train")


################ To work on GPU if available ##################
def move_graph_to_device(graph, device):
    """Move all tensor attributes of a HeteroData graph to the target device."""
    for node_type in graph.node_types:
        for key, val in graph[node_type].items():
            if torch.is_tensor(val):
                graph[node_type][key] = val.to(device)
    for edge_type in graph.edge_types:
        for key, val in graph[edge_type].items():
            if torch.is_tensor(val):
                graph[edge_type][key] = val.to(device)
    return graph


# Graph helpers
def hhmm_to_minutes(hhmm):
    """Convert HHMM integer/str to minutes since midnight; safe for NaNs."""
    if pd.isna(hhmm):
        return 0
    try:
        hhmm_int = int(hhmm)
    except Exception:
        return 0
    hours = hhmm_int // 100
    minutes = hhmm_int % 100
    return hours * 60 + minutes


# Normalization utilities
def normalize_with_idx(arr, fit_idx):
    """Normalize array with mean/std fitted on fit_idx rows; returns (normalized, mean, std_safe)."""
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    fit_idx = np.asarray(fit_idx, dtype=int)
    if fit_idx.size == 0:
        fit_idx = np.arange(arr.shape[0], dtype=int)
    mu = arr[fit_idx].mean(axis=0, keepdims=True)
    std = arr[fit_idx].std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    std_safe = std + 1e-6
    normalized = (arr - mu) / std_safe
    return normalized, mu, std_safe


# Evaluation metrics
def regression_metrics(
    y_true: torch.Tensor, y_pred: torch.Tensor, norm_stats: dict = None
) -> dict:

    y_pred_np = y_pred.detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()

    # Only denormalize if norm_stats has the expected structure
    if norm_stats is not None and isinstance(norm_stats, dict):
        if "mu" in norm_stats and "sigma" in norm_stats:
            if "ARR_DELAY" in norm_stats["mu"] and "ARR_DELAY" in norm_stats["sigma"]:
                mu = norm_stats["mu"]["ARR_DELAY"]
                sigma = norm_stats["sigma"]["ARR_DELAY"]
                y_pred_np = (y_pred_np * sigma) + mu
                y_true_np = (y_true_np * sigma) + mu

    mse = mean_squared_error(y_true_np, y_pred_np)
    rmse = mse**0.5
    mae = mean_absolute_error(y_true_np, y_pred_np)
    r2 = r2_score(y_true_np, y_pred_np)

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


def classification_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, args) -> dict:

    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    # If logits, apply sigmoid and threshold
    if np.any((y_pred_np > 1) | (y_pred_np < 0)):
        y_pred_np = 1 / (1 + np.exp(-y_pred_np))  # sigmoid
    y_pred_np = (y_pred_np >= args.border).astype(int)

    accuracy = accuracy_score(y_true_np, y_pred_np)

    f1 = f1_score(y_true_np, y_pred_np, average="weighted")
    precision = precision_score(
        y_true_np, y_pred_np, average="binary"
    )  # positive class precision
    recall = recall_score(
        y_true_np, y_pred_np, average="binary"
    )  # positive class recall

    return {
        "Accuracy": accuracy,
        "F1_Score": f1,
        "Recall": recall,
        "Precision": precision,
    }


# System stats
def get_available_gpu_memory():
    """Get available GPU VRAM in MB. Returns None if CUDA not available."""
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            props = torch.cuda.get_device_properties(device)
            allocated = torch.cuda.memory_allocated(device) / 1024**2
            total = props.total_memory / 1024**2
            available = total - allocated
            return {
                "available_mb": available,
                "total_mb": total,
                "allocated_mb": allocated,
            }
        except Exception:
            return None
    return None


def get_available_system_memory():
    """Get available system RAM in MB. Returns None if psutil unavailable."""
    if psutil:
        try:
            mem = psutil.virtual_memory()
            return {
                "available_mb": mem.available / 1024**2,
                "total_mb": mem.total / 1024**2,
                "used_mb": mem.used / 1024**2,
            }
        except Exception:
            return None
    return None


def print_available_memory():
    """Print available GPU VRAM and system RAM."""
    logger = get_logger()
    logger.info("Available Memory:")

    gpu_info = get_available_gpu_memory()
    if gpu_info:
        logger.info(
            f"  GPU VRAM:  {gpu_info['available_mb']:8.1f} MB / {gpu_info['total_mb']:.1f} MB"
        )
    else:
        logger.info("  GPU VRAM:  Not available (CPU mode)")

    ram_info = get_available_system_memory()
    if ram_info:
        logger.info(
            f"  System RAM: {ram_info['available_mb']:8.1f} MB / {ram_info['total_mb']:.1f} MB"
        )
    else:
        logger.info("  System RAM: Unable to query")


# Training/validation/test stats
def compute_epoch_stats(
    epoch, args, graph, labels_cat, preds_cat, epoch_losses, epoch_start_time, logger
):
    """Compute metrics and log resource usage for training or testing epochs.

    This function performs logging adjusted for the mode (train/val/test).
    Extracts mode from args.mode for appropriate logging behavior.
    """
    epoch_time = time.time() - epoch_start_time
    mode = args.mode

    # Metrics
    if args.prediction_type == "regression":
        norm_stats = getattr(graph, "norm_stats", None) or getattr(
            args, "norm_stats", None
        )
        metrics_results = regression_metrics(labels_cat, preds_cat, norm_stats)
    else:
        metrics_results = classification_metrics(labels_cat, preds_cat, args)

    avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)

    # Resource usage
    gpu_mem_peak = None
    if torch.cuda.is_available():
        try:
            gpu_mem_peak = torch.cuda.max_memory_allocated() / 1024**2
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            gpu_mem_peak = None

    cpu_mem_mb = None
    cpu_pct = None
    if psutil:
        try:
            p = psutil.Process()
            cpu_mem_mb = p.memory_info().rss / 1024**2
            cpu_pct = psutil.cpu_percent(interval=None)
        except Exception:
            cpu_mem_mb = cpu_pct = None

    # Build structured stats dictionary
    stats = {
        "mode": mode,
        "epoch": epoch + 1 if mode == "train" else None,
        "loss": avg_loss,
        "time_s": epoch_time,
        "gpu_mem_mb": gpu_mem_peak,
        "proc_mem_mb": cpu_mem_mb,
        "cpu_pct": cpu_pct,
    }
    stats.update(metrics_results)

    # Human-friendly log string
    info_parts = []
    if mode == "train":
        info_parts.append(f"Epoch {epoch+1}/{args.epochs}")
    elif mode in ("val", "test"):
        info_parts.append(mode.upper())

    info_parts.append(f"loss: {avg_loss:.4f}")
    if args.prediction_type == "regression":
        info_parts.append(
            f"MSE: {metrics_results['MSE']:.4f}, MAE: {metrics_results['MAE']:.4f}, "
            f"RMSE: {metrics_results['RMSE']:.4f}, R2: {metrics_results['R2']:.4f}"
        )
    else:
        info_parts.append(
            f"Accuracy: {metrics_results['Accuracy']:.4f}, Precision: {metrics_results['Precision']:.4f}, "
            f"Recall: {metrics_results['Recall']:.4f}, F1_Score: {metrics_results['F1_Score']:.4f}"
        )

    info_parts.append(f"time: {epoch_time:.2f}s")

    if gpu_mem_peak is not None:
        info_parts.append(f"gpu_mem: {gpu_mem_peak:.1f} MB")

    if cpu_mem_mb is not None and cpu_pct is not None:
        info_parts.append(f"proc_mem: {cpu_mem_mb:.1f} MB")
        info_parts.append(f"cpu%: {cpu_pct:.1f}")

    logger.info(" - ".join(info_parts))
    return stats


# Graph statistics
def print_graph_stats(data: HeteroData):
    """Print detailed graph statistics (node/edge counts, feature shapes, approximate memory)."""
    try:

        def _sizeof(t: torch.Tensor) -> int:
            return t.numel() * t.element_size()

        node_bytes = {}
        edge_bytes = {}
        total_nodes = 0
        total_edges = 0
        logger = get_logger()
        logger.info("\n" + "=" * 70)
        logger.info(" " * 27 + "GRAPH STATISTICS")
        logger.info("=" * 70)

        # Nodes
        logger.info("NODE TYPES:")
        node_total_bytes = 0
        for ntype in data.node_types:
            x = getattr(data[ntype], "x", None)
            if isinstance(x, torch.Tensor):
                n_nodes = x.size(0)
                feat_shape = tuple(x.shape)
                dtype = x.dtype
                node_bytes[ntype] = _sizeof(x)
                node_total_bytes += node_bytes[ntype]
            else:
                n_nodes = getattr(data[ntype], "num_nodes", "unknown")
                feat_shape = None
                dtype = None
                node_bytes[ntype] = 0
            if isinstance(n_nodes, int):
                total_nodes += n_nodes
            mb = node_bytes[ntype] / (1024 * 1024)
            logger.info(
                f"  {ntype:12s}: count={n_nodes:7}, shape={str(feat_shape):20s}, dtype={str(dtype):15s}, memory={mb:8.2f} MB"
            )

        logger.info(f"\n  >>> TOTAL NODES: {total_nodes:,}")
        logger.info(f"  >>> NODE MEMORY: {node_total_bytes / (1024*1024):.2f} MB")

        # Edges
        logger.info("\nEDGE TYPES:")
        edge_total_bytes = 0
        for etype in data.edge_types:
            eidx = getattr(data[etype], "edge_index", None)
            if isinstance(eidx, torch.Tensor):
                num_edges = eidx.size(1)
                shape = tuple(eidx.shape)
                edge_bytes[etype] = _sizeof(eidx)
                edge_total_bytes += edge_bytes[etype]
            else:
                num_edges = 0
                shape = None
                edge_bytes[etype] = 0
            total_edges += num_edges
            mb = edge_bytes[etype] / (1024 * 1024)
            etype_str = str(etype)[:40]
            logger.info(
                f"  {etype_str:42s}: edges={num_edges:7,}, shape={str(shape):20s}, memory={mb:8.2f} MB"
            )

        logger.info(f"\n  >>> TOTAL EDGES: {total_edges:,}")
        logger.info(f"  >>> EDGE MEMORY: {edge_total_bytes / (1024*1024):.2f} MB")

        # Labels / other tensors (regression/classification stored separately)
        label_bytes = 0
        y_reg = getattr(data["flight"], "y_reg", None)
        y_cls = getattr(data["flight"], "y_cls", None)
        any_label_printed = False
        if isinstance(y_reg, torch.Tensor):
            lb = _sizeof(y_reg)
            label_bytes += lb
            mb = lb / (1024 * 1024)
            logger.info("\nLABELS:")
            logger.info(
                f"  flight.y_reg: shape={tuple(y_reg.shape)}, dtype={y_reg.dtype}, memory={mb:.2f} MB"
            )
            any_label_printed = True
        if isinstance(y_cls, torch.Tensor):
            lb = _sizeof(y_cls)
            label_bytes += lb
            mb = lb / (1024 * 1024)
            if not any_label_printed:
                logger.info("\nLABELS:")
            logger.info(
                f"  flight.y_cls: shape={tuple(y_cls.shape)}, dtype={y_cls.dtype}, memory={mb:.2f} MB"
            )
            any_label_printed = True
        if not any_label_printed:
            logger.info("LABELS: None stored on flight nodes")

        # Grand totals
        total_bytes = node_total_bytes + edge_total_bytes + label_bytes
        logger.info("\n" + "=" * 70)
        logger.info(
            f"TOTAL MEMORY: {total_bytes / (1024*1024):.2f} MB ({total_bytes / 1024:.1f} KB, {total_bytes} bytes)"
        )
        logger.info("=" * 70 + "\n")

        if get_data_size is not None:
            try:
                builtin_bytes = get_data_size(data)
                builtin_mb = builtin_bytes / (1024 * 1024)
                logger.info(f"PyG get_data_size() validation: {builtin_mb:.2f} MB")
            except Exception as inner_e:
                logger.debug("get_data_size() failed: %s", inner_e)
        else:
            logger.info("Note: get_data_size() not available in this PyG version.")
    except Exception as e:
        logger.warning("Warning: failed to compute graph stats: %s", e)


# Neighbor fanout resolution
def resolve_fanouts(model, fanouts):
    """Resolve neighbor fanouts to match model depth.

    - If model exposes `num_layers`, use that as depth; otherwise infer from `convs` length; fallback 1.
    - If `fanouts` is None, return full-neighbor sampling `[-1] * depth`.
    - If length mismatch, pad or trim to match depth.
    """
    depth = getattr(model, "num_layers", None)
    if depth is None or depth <= 0:
        convs = getattr(model, "convs", None)
        try:
            depth = len(convs) if convs is not None else 1
        except Exception:
            depth = 1

    if fanouts is None:
        return [-1] * depth

    # normalize provided fanouts to depth
    try:
        length = len(fanouts)
    except Exception:
        return [-1] * depth

    if length != depth and length > 0:
        if length < depth:
            fanouts = fanouts + [fanouts[-1]] * (depth - length)
        else:
            fanouts = fanouts[:depth]
    return fanouts


def get_labels(data, prediction_type: str, mask=None):
    """Return labels for `data['flight']` according to prediction_type.

    - If `prediction_type=='regression'` returns regression targets (y_reg or y).
    - If classification returns classification targets (y_cls if present, else thresholded ARR_DELAY).

    If `mask` is provided, it is applied to the flattened label vector before return.
    Returned tensor is 1-D (squeezed) and for classification is a float tensor (0./1.) suitable
    for `BCEWithLogitsLoss`.
    """
    y_reg = getattr(data["flight"], "y_reg", None)
    y_cls = getattr(data["flight"], "y_cls", None)
    y_default = getattr(data["flight"], "y", None)

    if prediction_type == "regression":
        if y_reg is None and y_default is None:
            raise ValueError("No regression labels found on graph (y_reg or y)")
        src = y_reg if y_reg is not None else y_default
        out = src.squeeze(-1)
        if mask is not None:
            return out[mask]
        return out
    else:
        # classification
        if y_cls is not None:
            out = y_cls.view(-1)
        else:
            # Fallback: threshold ARR_DELAY >= 15 minutes
            if y_default is None:
                raise ValueError(
                    "No classification labels (y_cls) or ARR_DELAY (y) available"
                )
            out = (y_default.squeeze(-1) >= 15).long()
        # Return float (0./1.) for BCEWithLogitsLoss compatibility
        outf = out.float()
        if mask is not None:
            return outf[mask]
        return outf
