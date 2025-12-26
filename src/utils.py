import os
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, recall_score, precision_score
from torch_geometric.data import HeteroData
import logging
from typing import Optional
import numpy as np
import pandas as pd

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


################ NORMALIZATION ####################
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

################ TRAINING/VALIDATION/TESTING STATS ####################
def compute_epoch_stats(epoch, args, graph, labels_cat, preds_cat, epoch_losses, epoch_start_time, logger):
    """Compute metrics and log resource usage for training or testing epochs.

    This function performs logging adjusted for the mode (train/val/test).
    Extracts mode from args.mode for appropriate logging behavior.
    """
    import time
    epoch_time = time.time() - epoch_start_time
    mode = args.mode

    # Metrics
    if args.prediction_type == "regression":
        norm_stats = getattr(graph, "norm_stats", None) or getattr(args, "norm_stats", None)
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

        # Labels / other tensors (regression/classification stored separately)
        label_bytes = 0
        y_reg = getattr(data['flight'], 'y_reg', None)
        y_cls = getattr(data['flight'], 'y_cls', None)
        any_label_printed = False
        if isinstance(y_reg, torch.Tensor):
            lb = _sizeof(y_reg)
            label_bytes += lb
            mb = lb / (1024 * 1024)
            print(f"\nLABELS:")
            print(f"  flight.y_reg: shape={tuple(y_reg.shape)}, dtype={y_reg.dtype}, memory={mb:.2f} MB")
            any_label_printed = True
        if isinstance(y_cls, torch.Tensor):
            lb = _sizeof(y_cls)
            label_bytes += lb
            mb = lb / (1024 * 1024)
            if not any_label_printed:
                print(f"\nLABELS:")
            print(f"  flight.y_cls: shape={tuple(y_cls.shape)}, dtype={y_cls.dtype}, memory={mb:.2f} MB")
            any_label_printed = True
        if not any_label_printed:
            print("\nLABELS: None stored on flight nodes")

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


def get_labels(data, prediction_type: str, mask=None):
    """Return labels for `data['flight']` according to prediction_type.

    - If `prediction_type=='regression'` returns regression targets (y_reg or y).
    - If classification returns classification targets (y_cls if present, else thresholded ARR_DELAY).

    If `mask` is provided, it is applied to the flattened label vector before return.
    Returned tensor is 1-D (squeezed) and for classification is a float tensor (0./1.) suitable
    for `BCEWithLogitsLoss`.
    """
    y_reg = getattr(data['flight'], 'y_reg', None)
    y_cls = getattr(data['flight'], 'y_cls', None)
    y_default = getattr(data['flight'], 'y', None)

    if prediction_type == 'regression':
        if y_reg is None and y_default is None:
            raise ValueError('No regression labels found on graph (y_reg or y)')
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
                raise ValueError('No classification labels (y_cls) or ARR_DELAY (y) available')
            out = (y_default.squeeze(-1) >= 15).long()
        # Return float (0./1.) for BCEWithLogitsLoss compatibility
        outf = out.float()
        if mask is not None:
            return outf[mask]
        return outf


class WindowSubgraphBuilder:
    """
    Iterative builder for sliding window subgraphs that reuses computation.
    
    This builds subgraphs containing flights in a time window plus all edges
    between them and all connected nodes. Uses an incremental approach where
    consecutive windows reuse overlapping data to minimize rebuilding.
    """
    
    def __init__(self, graph, unit: int | None = None, learn_window: int | None = None, pred_window: int | None = None, window_stride: int | None = None):
        """
        Initialize the builder with the full graph.
        
        Args:
            graph: Full HeteroData graph
            unit: Time unit size in minutes (optional, enables per-unit caching)
            learn_window: learn window length in units (optional)
            pred_window: pred window length in units (optional)
            window_stride: stride in units (optional)
        """
        self.graph = graph
        self.graph_device = graph["flight"].x.device
        self.last_window_flights = None
        self.last_subgraph_data = None
        # Cache CPU edge indices for faster masking without torch.isin
        self.edge_index_cpu = {}
        for etype in graph.edge_types:
            self.edge_index_cpu[etype] = graph[etype].edge_index.cpu()
        self.num_flights = graph["flight"].x.size(0)

        # Optional per-unit cache for faster window assembly
        self.unit = unit
        self.learn_window = learn_window
        self.pred_window = pred_window
        self.window_stride = window_stride
        self.flights_by_unit = None
        if unit is not None and hasattr(graph["flight"], "timestamp_min"):
            try:
                timestamps_min = graph["flight"].timestamp_min.cpu().numpy()
                time_units = (timestamps_min // unit).astype(int)
                min_u = time_units.min()
                max_u = time_units.max()
                buckets = [[] for _ in range(max_u - min_u + 1)]
                for idx, tu in enumerate(time_units):
                    buckets[tu - min_u].append(idx)
                self.flights_by_unit = {
                    (min_u + offset): torch.tensor(arr, dtype=torch.long) if len(arr) > 0 else torch.tensor([], dtype=torch.long)
                    for offset, arr in enumerate(buckets)
                }
                self.unit_min = min_u
            except Exception:
                self.flights_by_unit = None

        # Build CSR-like adjacency to access neighbors per flight in O(1) ranges
        # For edge types where src=='flight': map flight -> neighbor nodes of dst_type
        # For edge types where dst=='flight': map flight -> neighbor nodes of src_type
        self.csr_src = {}  # dst_type -> {ptr, col}
        self.csr_dst = {}  # src_type -> {ptr, col}
        def _build_csr(rows: torch.Tensor, cols: torch.Tensor, n_rows: int):
            # Sort by row
            order = torch.argsort(rows, stable=True)
            rows_sorted = rows[order]
            cols_sorted = cols[order]
            counts = torch.bincount(rows_sorted, minlength=n_rows)
            ptr = torch.empty(n_rows + 1, dtype=torch.long)
            ptr[0] = 0
            torch.cumsum(counts, dim=0, out=ptr[1:])
            return ptr, cols_sorted
        for et in graph.edge_types:
            s, _, d = et
            ei = self.edge_index_cpu[et]
            if s == 'flight':
                ptr, col = _build_csr(ei[0], ei[1], self.num_flights)
                self.csr_src.setdefault(d, {})['ptr'] = ptr
                self.csr_src[d]['col'] = col
            if d == 'flight':
                ptr, col = _build_csr(ei[1], ei[0], self.num_flights)
                self.csr_dst.setdefault(s, {})['ptr'] = ptr
                self.csr_dst[s]['col'] = col

        # Node refcounts for non-flight node types (used in rolling updates)
        self.node_refcounts = {}
        for ntype in graph.node_types:
            if ntype == 'flight':
                continue
            n_nodes = getattr(graph[ntype], 'num_nodes', None)
            x = getattr(graph[ntype], 'x', None)
            if n_nodes is None and isinstance(x, torch.Tensor):
                n_nodes = x.size(0)
            if n_nodes is None:
                continue
            self.node_refcounts[ntype] = torch.zeros(n_nodes, dtype=torch.int32)
        
    def build_subgraph(self, learn_indices, pred_indices, device=None, window_time_range=None):
        """
        Build an induced subgraph for a sliding window.
        
        Extracts:
        - Flight nodes in the window (learn + pred)
        - All edges between these flights and ANY other node type
        - All nodes connected to these flights via edges
        
        Args:
            learn_indices: numpy array of flight indices in learning window
            pred_indices: numpy array of flight indices in prediction window
            device: Optional device to move subgraph to
            
        Returns:
            subgraph: Induced HeteroData subgraph
            local_pred_mask: Boolean tensor marking pred flights in subgraph ordering
        """
        import torch
        import numpy as np
        from torch_geometric.data import HeteroData
        
        # Combine learn and pred indices for the full window (CPU, sorted, unique)
        if window_time_range is not None and self.flights_by_unit is not None:
            start_u, end_u = window_time_range  # inclusive end
            # Collect flights from cached per-unit buckets
            flights_list = []
            for u in range(start_u, end_u + 1):
                bucket = self.flights_by_unit.get(u, None)
                if bucket is not None and bucket.numel() > 0:
                    flights_list.append(bucket)
            if flights_list:
                window_flights_tensor = torch.unique(torch.cat(flights_list), sorted=True)
            else:
                window_flights_tensor = torch.tensor([], dtype=torch.long)
        else:
            window_flights = np.unique(np.concatenate([learn_indices, pred_indices]))
            window_flights_tensor = torch.from_numpy(window_flights).long().cpu()
            window_flights_tensor = torch.unique(window_flights_tensor, sorted=True)

        # Rolling update using CSR and refcounts if we have per-unit windows
        node_selections = {}
        node_selections['flight'] = window_flights_tensor

        if window_time_range is not None and self.flights_by_unit is not None:
            # Determine added/removed flights vs last window
            if self.last_window_flights is None:
                added = window_flights_tensor
                removed = torch.tensor([], dtype=torch.long)
            else:
                old = self.last_window_flights
                added = torch.unique(window_flights_tensor[~torch.isin(window_flights_tensor, old)], sorted=True)
                removed = torch.unique(old[~torch.isin(old, window_flights_tensor)], sorted=True)

            # Update refcounts for neighbors per added/removed flights
            def _update_counts(flights: torch.Tensor, delta: int):
                if flights.numel() == 0:
                    return
                # flight -> dst_type neighbors
                for dst_type, csr in self.csr_src.items():
                    ptr, col = csr['ptr'], csr['col']
                    # Collect neighbors for all flights
                    neigh = []
                    for f in flights.tolist():
                        s, e = int(ptr[f].item()), int(ptr[f+1].item())
                        if e > s:
                            neigh.append(col[s:e])
                    if neigh:
                        ncat = torch.unique(torch.cat(neigh))
                        self.node_refcounts[dst_type].index_add_(0, ncat, torch.full((ncat.numel(),), delta, dtype=torch.int32))
                # src_type -> flight neighbors
                for src_type, csr in self.csr_dst.items():
                    ptr, col = csr['ptr'], csr['col']
                    neigh = []
                    for f in flights.tolist():
                        s, e = int(ptr[f].item()), int(ptr[f+1].item())
                        if e > s:
                            neigh.append(col[s:e])
                    if neigh:
                        ncat = torch.unique(torch.cat(neigh))
                        self.node_refcounts[src_type].index_add_(0, ncat, torch.full((ncat.numel(),), delta, dtype=torch.int32))

            _update_counts(added, +1)
            _update_counts(removed, -1)

            # Build node selections from positive refcounts
            for ntype, counts in self.node_refcounts.items():
                sel = torch.nonzero(counts > 0, as_tuple=False).view(-1)
                node_selections[ntype] = sel
        else:
            # Fallback: compute neighbors fresh via masking (still efficient using flight_mask)
            flight_mask = torch.zeros(self.num_flights, dtype=torch.bool)
            flight_mask[window_flights_tensor] = True
            for edge_type in self.graph.edge_types:
                src_type, rel_name, dst_type = edge_type
                edge_index = self.edge_index_cpu[edge_type]
                if src_type == 'flight':
                    ei0, ei1 = edge_index[0], edge_index[1]
                    connected = torch.unique(ei1[flight_mask[ei0]])
                    if connected.numel() > 0:
                        node_selections[dst_type] = torch.unique(torch.cat([
                            node_selections.get(dst_type, torch.tensor([], dtype=torch.long)), connected
                        ]))
                if dst_type == 'flight':
                    ei0, ei1 = edge_index[0], edge_index[1]
                    connected = torch.unique(ei0[flight_mask[ei1]])
                    if connected.numel() > 0:
                        node_selections[src_type] = torch.unique(torch.cat([
                            node_selections.get(src_type, torch.tensor([], dtype=torch.long)), connected
                        ]))
        
        # Build induced subgraph using PyG's subgraph method
        # HeteroData.subgraph expects CPU LongTensor indices; ensure sorted for stable ordering
        for ntype in list(node_selections.keys()):
            node_selections[ntype] = node_selections[ntype].cpu().long()
            node_selections[ntype] = torch.unique(node_selections[ntype], sorted=True)
        subgraph = self.graph.subgraph(node_selections)
        
        # Compute local pred_mask
        pred_indices_tensor = torch.from_numpy(pred_indices).long().cpu()
        pred_indices_tensor = torch.unique(pred_indices_tensor, sorted=True)
        # Mask corresponds to the order of node_selections['flight'] which we controlled
        local_pred_mask = torch.isin(node_selections['flight'], pred_indices_tensor)
        
        if device is not None:
            subgraph = subgraph.to(device)
            local_pred_mask = local_pred_mask.to(device)
        
        # Store for reuse on next window
        self.last_window_flights = window_flights_tensor
        self.last_subgraph_data = node_selections
        
        return subgraph, local_pred_mask


def build_window_subgraph(graph, learn_indices, pred_indices, device=None):
    """
    Build an induced subgraph for a sliding window (standalone function).
    
    This is a convenience wrapper around WindowSubgraphBuilder for single-use cases.
    For iterative window processing, use WindowSubgraphBuilder directly to reuse state.
    
    Args:
        graph: Full HeteroData graph
        learn_indices: numpy array of flight indices in learning window
        pred_indices: numpy array of flight indices in prediction window
        device: Optional device to move subgraph to
    
    Returns:
        subgraph: Induced HeteroData subgraph
        local_pred_mask: Boolean tensor marking pred flights in subgraph ordering
    """
    builder = WindowSubgraphBuilder(graph)
    return builder.build_subgraph(learn_indices, pred_indices, device)
