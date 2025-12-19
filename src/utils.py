import os
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, recall_score, precision_score
from torch_geometric.data import HeteroData
import logging
from typing import Optional
import numpy as np

try:
    from torch_geometric.profile.utils import get_data_size  # available in recent PyG
except Exception:
    get_data_size = None

try:
    import psutil
except Exception:
    psutil = None


###################### SETUP LOGGING ######################
def setup_logging(verbosity: int = 0, logfile: Optional[str] = None) -> logging.Logger:
    """Configure root logger for the application.

    verbosity: 0=WARNING, 1=INFO, 2=DEBUG
    """
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO

    handlers = [logging.StreamHandler()]
    if logfile:
        handlers.append(logging.FileHandler(logfile))

    logging.basicConfig(level=level,
                        format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S",
                        handlers=handlers)
    return logging.getLogger("train")


def ensure_dir(
        directory: str
        ) -> None:
    
    if not os.path.exists(directory):
        os.makedirs(directory)


###################### EVALUATION METRICS ######################
def regression_metrics(
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        norm_stats: dict = None
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
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_true_np, y_pred_np)
    r2 = r2_score(y_true_np, y_pred_np)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

def classification_metrics(
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        args
    ) -> dict:

    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    # If logits, apply sigmoid and threshold
    if np.any((y_pred_np > 1) | (y_pred_np < 0)):
        y_pred_np = 1 / (1 + np.exp(-y_pred_np))  # sigmoid
    y_pred_np = (y_pred_np >= args.border).astype(int)

    accuracy = accuracy_score(y_true_np, y_pred_np)

    f1 = f1_score(y_true_np, y_pred_np, average='weighted')
    precision = precision_score(y_true_np, y_pred_np, average='binary')  # positive class precision
    recall = recall_score(y_true_np, y_pred_np, average='binary')  # positive class recall

    return {
        "Accuracy": accuracy,
        "F1_Score": f1,
        "Recall": recall,
        "Precision": precision
    }

###############################
# To work on GPU if available #
###############################
def get_available_gpu_memory():
    """Get available GPU VRAM in MB. Returns None if CUDA not available."""
    if torch.cuda.is_available():
        try:
            device = torch.device('cuda')
            props = torch.cuda.get_device_properties(device)
            allocated = torch.cuda.memory_allocated(device) / 1024**2
            total = props.total_memory / 1024**2
            available = total - allocated
            return {"available_mb": available, "total_mb": total, "allocated_mb": allocated}
        except Exception:
            return None
    return None


def get_available_system_memory():
    """Get available system RAM in MB. Returns None if psutil unavailable."""
    if psutil:
        try:
            mem = psutil.virtual_memory()
            return {"available_mb": mem.available / 1024**2, "total_mb": mem.total / 1024**2, "used_mb": mem.used / 1024**2}
        except Exception:
            return None
    return None


def print_available_memory():
    """Print available GPU VRAM and system RAM."""
    print("\nAvailable Memory:")
    
    gpu_info = get_available_gpu_memory()
    if gpu_info:
        print(f"  GPU VRAM:  {gpu_info['available_mb']:8.1f} MB / {gpu_info['total_mb']:.1f} MB")
    else:
        print(f"  GPU VRAM:  Not available (CPU mode)")
    
    ram_info = get_available_system_memory()
    if ram_info:
        print(f"  System RAM: {ram_info['available_mb']:8.1f} MB / {ram_info['total_mb']:.1f} MB")
    else:
        print(f"  System RAM: Unable to query")


def move_graph_to_device(graph, device):
    for node_type in graph.node_types:
        graph[node_type].x = graph[node_type].x.to(device)
        if "y" in graph[node_type]:
            graph[node_type].y = graph[node_type].y.to(device)
    for edge_type in graph.edge_types:
        graph[edge_type].edge_index = graph[edge_type].edge_index.to(device)
    return graph

################ TRAINING/VALIDATION/TESTING STATS ####################
def compute_epoch_stats(epoch, args, graph, labels_cat, preds_cat, epoch_losses, epoch_start, logger):
    """Compute metrics and log resource usage for training or testing epochs.

    This function performs logging adjusted for the mode (train/val/test).
    Extracts mode from args.mode for appropriate logging behavior.
    """
    import time
    epoch_time = time.time() - epoch_start
    mode = args.mode

    # Metrics
    if args.prediction_type == "regression":
        norm_stats = getattr(graph, "norm_stats", None) or getattr(args, "norm_stats", None)
        metrics_results = regression_metrics(labels_cat, preds_cat, norm_stats)
        metrics_str = (
            f"MSE: {metrics_results['MSE']:.4f}, MAE: {metrics_results['MAE']:.4f}, "
            f"RMSE: {metrics_results['RMSE']:.4f}, R2: {metrics_results['R2']:.4f}"
        )
    else:
        metrics_results = classification_metrics(labels_cat, preds_cat, args)
        metrics_str = f"Accuracy: {metrics_results['Accuracy']:.4f}, Precision: {metrics_results['Precision']:.4f}, Recall: {metrics_results['Recall']:.4f}, F1_Score: {metrics_results['F1_Score']:.4f}"

    avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)

    # Resource usage
    gpu_mem_cur = None
    gpu_mem_peak = None
    if torch.cuda.is_available():
        try:
            gpu_mem_cur = torch.cuda.memory_allocated() / 1024**2
            gpu_mem_peak = torch.cuda.max_memory_allocated() / 1024**2
        except Exception:
            gpu_mem_cur = gpu_mem_peak = None

    cpu_info = None
    if psutil:
        try:
            p = psutil.Process()
            mem_mb = p.memory_info().rss / 1024**2
            cpu_pct = psutil.cpu_percent(interval=None)
            cpu_info = (mem_mb, cpu_pct)
        except Exception:
            cpu_info = None

    # Build info parts based on mode
    info_parts = []
    
    if mode == "train":
        info_parts.append(f"Epoch {epoch+1}/{args.epochs}")
    elif mode in ("val", "test"):
        info_parts.append(mode.upper())
    
    info_parts.append(f"loss: {avg_loss:.4f}")
    info_parts.append(metrics_str)
    info_parts.append(f"time: {epoch_time:.2f}s")
    
    if gpu_mem_peak is not None:
        info_parts.append(f"gpu_mem: {gpu_mem_peak:.1f} MB")
    
    if cpu_info is not None:
        info_parts.append(f"proc_mem: {cpu_info[0]:.1f} MB")
        info_parts.append(f"cpu%: {cpu_info[1]:.1f}")

    logger.info(" - ".join(info_parts))


################ GRAPH STATISTICS ####################
def print_graph_stats(data: HeteroData):
    """Print detailed graph statistics (node/edge counts, feature shapes, approximate memory)."""
    try:
        def _sizeof(t: torch.Tensor) -> int:
            return t.numel() * t.element_size()

        node_bytes = {}
        edge_bytes = {}
        total_nodes = 0
        total_edges = 0
        print("\n" + "="*70)
        print(" "*27 + "GRAPH STATISTICS")
        print("="*70)

        # Nodes
        print("\nNODE TYPES:")
        node_total_bytes = 0
        for ntype in data.node_types:
            x = getattr(data[ntype], 'x', None)
            if isinstance(x, torch.Tensor):
                n_nodes = x.size(0)
                feat_shape = tuple(x.shape)
                dtype = x.dtype
                node_bytes[ntype] = _sizeof(x)
                node_total_bytes += node_bytes[ntype]
            else:
                n_nodes = getattr(data[ntype], 'num_nodes', 'unknown')
                feat_shape = None
                dtype = None
                node_bytes[ntype] = 0
            if isinstance(n_nodes, int):
                total_nodes += n_nodes
            mb = node_bytes[ntype] / (1024 * 1024)
            print(f"  {ntype:12s}: count={n_nodes:7}, shape={str(feat_shape):20s}, dtype={str(dtype):15s}, memory={mb:8.2f} MB")

        print(f"\n  >>> TOTAL NODES: {total_nodes:,}")
        print(f"  >>> NODE MEMORY: {node_total_bytes / (1024*1024):.2f} MB")

        # Edges
        print("\nEDGE TYPES:")
        edge_total_bytes = 0
        for etype in data.edge_types:
            eidx = getattr(data[etype], 'edge_index', None)
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
            print(f"  {etype_str:42s}: edges={num_edges:7,}, shape={str(shape):20s}, memory={mb:8.2f} MB")

        print(f"\n  >>> TOTAL EDGES: {total_edges:,}")
        print(f"  >>> EDGE MEMORY: {edge_total_bytes / (1024*1024):.2f} MB")

        # Labels / other tensors
        y = getattr(data['flight'], 'y', None)
        label_bytes = 0
        if isinstance(y, torch.Tensor):
            label_bytes = _sizeof(y)
            mb = label_bytes / (1024 * 1024)
            print(f"\nLABELS:")
            print(f"  flight.y: shape={tuple(y.shape)}, dtype={y.dtype}, memory={mb:.2f} MB")

        # Grand totals
        total_bytes = node_total_bytes + edge_total_bytes + label_bytes
        print("\n" + "="*70)
        print(f"TOTAL MEMORY: {total_bytes / (1024*1024):.2f} MB ({total_bytes / 1024:.1f} KB, {total_bytes} bytes)")
        print("="*70 + "\n")

        if get_data_size is not None:
            try:
                builtin_bytes = get_data_size(data)
                builtin_mb = builtin_bytes / (1024 * 1024)
                print(f"PyG get_data_size() validation: {builtin_mb:.2f} MB")
            except Exception as inner_e:
                print("Note: get_data_size() failed:", inner_e)
        else:
            print("Note: get_data_size() not available in this PyG version.")
    except Exception as e:
        print("Warning: failed to compute graph stats:", e)


################ NEIGHBOR FANOUT RESOLUTION ####################
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
