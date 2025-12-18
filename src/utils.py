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

    if norm_stats is not None:
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
def move_graph_to_device(graph, device):
    for node_type in graph.node_types:
        graph[node_type].x = graph[node_type].x.to(device)
        if "y" in graph[node_type]:
            graph[node_type].y = graph[node_type].y.to(device)
    for edge_type in graph.edge_types:
        graph[edge_type].edge_index = graph[edge_type].edge_index.to(device)
    return graph


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
