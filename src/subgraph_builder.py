"""
Sliding window subgraph builders for heterogeneous graphs.

Optimized for memory-efficient sorted datasets using split boundaries instead of masks.

Contains:
- WindowSubgraphBuilder: CSR-based incremental builder with GPU-resident support
- OptimizedWindowSubgraphBuilder: Direct edge assembly without PyG's subgraph method
"""

import torch


class WindowSubgraphBuilder:
    """
    Memory-efficient sliding window subgraph builder for sorted flight data.

    Uses CSR-based neighbor gathering and incremental refcount updates.
    Leverages the fact that flights are chronologically sorted.
    """

    def __init__(
        self,
        graph,
        unit: int | None = None,
        learn_window: int | None = None,
        pred_window: int | None = None,
        window_stride: int | None = None,
        use_gpu_resident: bool = False,
    ):
        """
        Initialize the builder with the full graph.

        Args:
            graph: Full HeteroData graph with sorted flights
            unit: Time unit size in minutes (enables per-unit caching)
            learn_window: learn window length in units
            pred_window: pred window length in units
            window_stride: stride in units
            use_gpu_resident: Keep graph on GPU and build subgraphs in GPU memory
        """
        self.graph = graph
        self.graph_device = graph["flight"].x.device
        self.use_gpu_resident = use_gpu_resident
        self.working_device = (
            self.graph_device if use_gpu_resident else torch.device("cpu")
        )
        self.last_window_flights = None

        # Cache edge indices on working device
        self.edge_index_cache = {}
        for etype in graph.edge_types:
            self.edge_index_cache[etype] = graph[etype].edge_index.to(
                self.working_device
            )

        self.num_flights = graph["flight"].x.size(0)

        # Build CSR-like adjacency for fast neighbor lookup
        # For edge types where src=='flight': map flight -> dst neighbors
        # For edge types where dst=='flight': map flight -> src neighbors
        self.csr_src = {}  # dst_type -> {ptr, col}
        self.csr_dst = {}  # src_type -> {ptr, col}

        def _build_csr(rows: torch.Tensor, cols: torch.Tensor, n_rows: int):
            order = torch.argsort(rows, stable=True)
            rows_sorted = rows[order]
            cols_sorted = cols[order]
            counts = torch.bincount(rows_sorted, minlength=n_rows)
            ptr = torch.zeros(n_rows + 1, dtype=torch.long, device=self.working_device)
            torch.cumsum(counts, dim=0, out=ptr[1:])
            return ptr, cols_sorted

        for et in graph.edge_types:
            s, _, d = et
            ei = self.edge_index_cache[et]
            if s == "flight":
                ptr, col = _build_csr(ei[0], ei[1], self.num_flights)
                self.csr_src.setdefault(d, {})["ptr"] = ptr
                self.csr_src[d]["col"] = col
            if d == "flight":
                ptr, col = _build_csr(ei[1], ei[0], self.num_flights)
                self.csr_dst.setdefault(s, {})["ptr"] = ptr
                self.csr_dst[s]["col"] = col

        # Node refcounts for incremental updates
        self.node_refcounts = {}
        for ntype in graph.node_types:
            if ntype == "flight":
                continue
            x = getattr(graph[ntype], "x", None)
            if x is not None:
                n_nodes = x.size(0)
                self.node_refcounts[ntype] = torch.zeros(
                    n_nodes, dtype=torch.int32, device=self.working_device
                )

    def build_subgraph(
        self, learn_indices, pred_indices, device=None, window_time_range=None
    ):
        """
        Build an induced subgraph for a sliding window.

        Args:
            learn_indices: numpy array (int32) of flight indices in learning window
            pred_indices: numpy array (int32) of flight indices in prediction window
            device: Optional device to move subgraph to
            window_time_range: Optional (start_unit, end_unit) for metadata

        Returns:
            subgraph: Induced HeteroData subgraph
            local_pred_mask: Boolean tensor marking pred flights in subgraph
        """
        # Since flights are sorted, learn/pred indices are already sorted ranges
        # Combine into single contiguous range: [learn_indices[0], pred_indices[-1]+1)
        window_start = int(learn_indices[0])
        window_end = int(pred_indices[-1]) + 1
        window_flights_tensor = torch.arange(
            window_start, window_end, dtype=torch.long, device=self.working_device
        )

        # Build node selections using CSR neighbor gathering
        node_selections = {"flight": window_flights_tensor}

        # Vectorized neighbor gathering for all flights in window
        for dst_type, csr in self.csr_src.items():
            ptr, col = csr["ptr"], csr["col"]
            # Gather neighbors for range [window_start, window_end)
            start_ptr = ptr[window_start]
            end_ptr = ptr[window_end]
            if end_ptr > start_ptr:
                neighbors = torch.unique(col[start_ptr:end_ptr])
                node_selections[dst_type] = neighbors

        for src_type, csr in self.csr_dst.items():
            ptr, col = csr["ptr"], csr["col"]
            start_ptr = ptr[window_start]
            end_ptr = ptr[window_end]
            if end_ptr > start_ptr:
                neighbors = torch.unique(col[start_ptr:end_ptr])
                if src_type in node_selections:
                    node_selections[src_type] = torch.unique(
                        torch.cat([node_selections[src_type], neighbors])
                    )
                else:
                    node_selections[src_type] = neighbors

        # Build induced subgraph using PyG's subgraph method
        node_selections_for_pyg = {
            ntype: sel.to(self.graph_device).long()
            for ntype, sel in node_selections.items()
        }

        subgraph = self.graph.subgraph(node_selections_for_pyg)

        # Compute local pred_mask based on actual subgraph size
        # Pred flights are at indices [len(learn_indices), len(learn_indices) + len(pred_indices))
        subgraph_flight_count = subgraph["flight"].x.size(0)
        local_pred_mask = torch.zeros(
            subgraph_flight_count, dtype=torch.bool, device=self.graph_device
        )
        pred_start_local = len(learn_indices)
        pred_end_local = min(
            pred_start_local + len(pred_indices), subgraph_flight_count
        )
        local_pred_mask[pred_start_local:pred_end_local] = True

        if device is not None:
            subgraph = subgraph.to(device)
            local_pred_mask = local_pred_mask.to(device)

        self.last_window_flights = window_flights_tensor

        return subgraph, local_pred_mask


def build_window_subgraph(graph, learn_indices, pred_indices, device=None):
    """
    Build an induced subgraph for a sliding window (standalone function).

    Convenience wrapper for single-use cases.
    For iterative window processing, use WindowSubgraphBuilder directly.

    Args:
        graph: Full HeteroData graph with sorted flights
        learn_indices: numpy array (int32) of flight indices in learning window
        pred_indices: numpy array (int32) of flight indices in prediction window
        device: Optional device to move subgraph to

    Returns:
        subgraph: Induced HeteroData subgraph
        local_pred_mask: Boolean tensor marking pred flights in subgraph
    """
    builder = WindowSubgraphBuilder(graph)
    return builder.build_subgraph(learn_indices, pred_indices, device)
