"""
Sliding window subgraph builders for heterogeneous graphs.

Optimized for memory-efficient sorted datasets using split boundaries instead of masks.

Contains:
- WindowSubgraphBuilder: CSR-based incremental builder with GPU-resident support
- OptimizedWindowSubgraphBuilder: Direct edge assembly without PyG's subgraph method
"""

import torch
import numpy as np
from torch_geometric.data import HeteroData


class WindowSubgraphBuilder:
    """
    Memory-efficient sliding window subgraph builder for sorted flight data.
    
    Uses CSR-based neighbor gathering and incremental refcount updates.
    Leverages the fact that flights are chronologically sorted.
    """
    
    def __init__(self, graph, unit: int | None = None, learn_window: int | None = None, pred_window: int | None = None, window_stride: int | None = None, use_gpu_resident: bool = False):
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
        self.working_device = self.graph_device if use_gpu_resident else torch.device('cpu')
        self.last_window_flights = None
        
        # Cache edge indices on working device
        self.edge_index_cache = {}
        for etype in graph.edge_types:
            self.edge_index_cache[etype] = graph[etype].edge_index.to(self.working_device)
        
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
            if s == 'flight':
                ptr, col = _build_csr(ei[0], ei[1], self.num_flights)
                self.csr_src.setdefault(d, {})['ptr'] = ptr
                self.csr_src[d]['col'] = col
            if d == 'flight':
                ptr, col = _build_csr(ei[1], ei[0], self.num_flights)
                self.csr_dst.setdefault(s, {})['ptr'] = ptr
                self.csr_dst[s]['col'] = col

        # Node refcounts for incremental updates
        self.node_refcounts = {}
        for ntype in graph.node_types:
            if ntype == 'flight':
                continue
            x = getattr(graph[ntype], 'x', None)
            if x is not None:
                n_nodes = x.size(0)
                self.node_refcounts[ntype] = torch.zeros(n_nodes, dtype=torch.int32, device=self.working_device)
        
    def build_subgraph(self, learn_indices, pred_indices, device=None, window_time_range=None):
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
        window_flights_tensor = torch.arange(window_start, window_end, dtype=torch.long, device=self.working_device)
        
        # Build node selections using CSR neighbor gathering
        node_selections = {'flight': window_flights_tensor}
        
        # Vectorized neighbor gathering for all flights in window
        for dst_type, csr in self.csr_src.items():
            ptr, col = csr['ptr'], csr['col']
            # Gather neighbors for range [window_start, window_end)
            start_ptr = ptr[window_start]
            end_ptr = ptr[window_end]
            if end_ptr > start_ptr:
                neighbors = torch.unique(col[start_ptr:end_ptr])
                node_selections[dst_type] = neighbors
        
        for src_type, csr in self.csr_dst.items():
            ptr, col = csr['ptr'], csr['col']
            start_ptr = ptr[window_start]
            end_ptr = ptr[window_end]
            if end_ptr > start_ptr:
                neighbors = torch.unique(col[start_ptr:end_ptr])
                if src_type in node_selections:
                    node_selections[src_type] = torch.unique(torch.cat([node_selections[src_type], neighbors]))
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
        local_pred_mask = torch.zeros(subgraph_flight_count, dtype=torch.bool, device=self.graph_device)
        pred_start_local = len(learn_indices)
        pred_end_local = min(pred_start_local + len(pred_indices), subgraph_flight_count)
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


class OptimizedWindowSubgraphBuilder:
    """
    Highly optimized sliding window subgraph builder for sorted flight data.
    
    Features:
    - GPU-resident graph support (zero-copy subgraph building)
    - Vectorized CSR neighbor gathering using range queries
    - Direct edge assembly without PyG's subgraph method
    - Minimal memory overhead (no masks, no per-unit caching)
    """
    
    def __init__(self, graph, unit=None, learn_window=None, pred_window=None, window_stride=None, use_gpu_resident=False):
        """
        Initialize the optimized builder.
        
        Args:
            graph: Full HeteroData graph with sorted flights
            unit: Time unit size in minutes (unused, kept for compatibility)
            learn_window: learn window length in units (unused, kept for compatibility)
            pred_window: pred window length in units (unused, kept for compatibility)
            window_stride: stride in units (unused, kept for compatibility)
            use_gpu_resident: Keep graph on GPU and build subgraphs in GPU memory
        """
        self.graph = graph
        self.graph_device = graph["flight"].x.device
        self.use_gpu_resident = use_gpu_resident
        self.working_device = self.graph_device if use_gpu_resident else torch.device('cpu')
        
        print(f"Initializing OptimizedWindowSubgraphBuilder (GPU-resident: {use_gpu_resident}, device: {self.working_device})")
        
        # Cache edge indices and attributes on working device
        self.edge_index_cache = {}
        self.edge_attrs = {}
        for etype in graph.edge_types:
            ei = graph[etype].edge_index
            self.edge_index_cache[etype] = ei.to(self.working_device)
            # Cache edge attributes
            self.edge_attrs[etype] = {}
            for key, val in graph[etype].items():
                if key != 'edge_index' and torch.is_tensor(val):
                    self.edge_attrs[etype][key] = val.to(self.working_device)
        
        # Cache node features on working device
        self.node_features = {}
        for ntype in graph.node_types:
            self.node_features[ntype] = {}
            for key, val in graph[ntype].items():
                if torch.is_tensor(val):
                    self.node_features[ntype][key] = val.to(self.working_device)
        
        self.num_flights = graph["flight"].x.size(0)
        
        # Build CSR adjacency on working device
        self.csr_src = {}
        self.csr_dst = {}
        
        def _build_csr(rows, cols, n_rows):
            order = torch.argsort(rows, stable=True)
            rows_sorted = rows[order]
            cols_sorted = cols[order]
            counts = torch.bincount(rows_sorted, minlength=n_rows)
            ptr = torch.zeros(n_rows + 1, dtype=torch.long, device=self.working_device)
            torch.cumsum(counts, dim=0, out=ptr[1:])
            return ptr, cols_sorted, order
        
        for et in graph.edge_types:
            s, _, d = et
            ei = self.edge_index_cache[et]
            if s == 'flight':
                ptr, col, order = _build_csr(ei[0], ei[1], self.num_flights)
                self.csr_src.setdefault(d, {})
                self.csr_src[d]['ptr'] = ptr
                self.csr_src[d]['col'] = col
                self.csr_src[d]['edge_order'] = order
                self.csr_src[d]['edge_type'] = et
            if d == 'flight':
                ptr, col, order = _build_csr(ei[1], ei[0], self.num_flights)
                self.csr_dst.setdefault(s, {})
                self.csr_dst[s]['ptr'] = ptr
                self.csr_dst[s]['col'] = col
                self.csr_dst[s]['edge_order'] = order
                self.csr_dst[s]['edge_type'] = et
        
        print(f"  CSR adjacency built: {len(self.csr_src)} src types, {len(self.csr_dst)} dst types")
    
    def build_subgraph(self, learn_indices, pred_indices, device=None, window_time_range=None):
        """
        Build subgraph using direct CSR range queries.
        
        Since flights are sorted, window flights form a contiguous range.
        
        Args:
            learn_indices: numpy array (int32) of flight indices in learning window
            pred_indices: numpy array (int32) of flight indices in prediction window
            device: Optional device to move subgraph to
            window_time_range: Optional (start_unit, end_unit) for metadata
            
        Returns:
            subgraph: Induced HeteroData subgraph
            local_pred_mask: Boolean tensor marking pred flights
        """
        # Window is contiguous range [learn_start, pred_end)
        window_start = int(learn_indices[0])
        window_end = int(pred_indices[-1]) + 1
        window_size = window_end - window_start
        
        # Build node selections using CSR range queries
        node_selections = {}
        node_selections['flight'] = torch.arange(window_start, window_end, dtype=torch.long, device=self.working_device)
        
        # Vectorized neighbor gathering using CSR pointers
        for dst_type, csr in self.csr_src.items():
            ptr = csr['ptr']
            col = csr['col']
            start_ptr = ptr[window_start]
            end_ptr = ptr[window_end]
            if end_ptr > start_ptr:
                node_selections[dst_type] = torch.unique(col[start_ptr:end_ptr])
        
        for src_type, csr in self.csr_dst.items():
            ptr = csr['ptr']
            col = csr['col']
            start_ptr = ptr[window_start]
            end_ptr = ptr[window_end]
            if end_ptr > start_ptr:
                neighbors = torch.unique(col[start_ptr:end_ptr])
                if src_type in node_selections:
                    node_selections[src_type] = torch.unique(torch.cat([node_selections[src_type], neighbors]))
                else:
                    node_selections[src_type] = neighbors
        
        # Build subgraph directly
        subgraph = self._build_subgraph_direct(node_selections)
        
        # Compute pred mask based on actual subgraph size
        subgraph_flight_count = subgraph["flight"].x.size(0)
        local_pred_mask = torch.zeros(subgraph_flight_count, dtype=torch.bool, device=self.working_device)
        pred_start_local = len(learn_indices)
        pred_end_local = min(pred_start_local + len(pred_indices), subgraph_flight_count)
        local_pred_mask[pred_start_local:pred_end_local] = True
        
        # Move to target device if needed
        if device is not None and device != self.working_device:
            subgraph = subgraph.to(device)
            local_pred_mask = local_pred_mask.to(device)
        
        return subgraph, local_pred_mask
    
    def _build_subgraph_direct(self, node_selections):
        """Build subgraph manually without PyG's subgraph method."""
        subgraph = HeteroData()
        
        # Build node mappings (global -> local)
        node_maps = {}
        for ntype, global_ids in node_selections.items():
            # Get sample tensor to determine size
            sample_feat = next(iter(self.node_features[ntype].values()))
            n_global = sample_feat.size(0)
            
            # Create mapping
            local_map = torch.full((n_global,), -1, dtype=torch.long, device=self.working_device)
            local_map[global_ids] = torch.arange(global_ids.numel(), device=self.working_device)
            
            node_maps[ntype] = {
                'global_ids': global_ids,
                'local_map': local_map
            }
            
            # Copy node features
            for attr_name, attr_val in self.node_features[ntype].items():
                subgraph[ntype][attr_name] = attr_val[global_ids]
        
        # Build edges
        for etype in self.graph.edge_types:
            src_type, rel, dst_type = etype
            if src_type not in node_maps or dst_type not in node_maps:
                continue
            
            edge_index = self.edge_index_cache[etype]
            src_map = node_maps[src_type]['local_map']
            dst_map = node_maps[dst_type]['local_map']
            
            # Remap edges: both src and dst must be in subgraph
            src_local = src_map[edge_index[0]]
            dst_local = dst_map[edge_index[1]]
            mask = (src_local >= 0) & (dst_local >= 0)
            
            if mask.any():
                subgraph[etype].edge_index = torch.stack([src_local[mask], dst_local[mask]], dim=0)
                # Copy edge attributes
                for attr_name, attr_val in self.edge_attrs[etype].items():
                    if attr_val.size(0) == edge_index.size(1):
                        subgraph[etype][attr_name] = attr_val[mask]
        
        return subgraph
