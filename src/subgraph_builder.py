"""
Sliding window subgraph builders for heterogeneous graphs.

Contains:
- WindowSubgraphBuilder: CSR-based incremental builder with GPU-resident support
- OptimizedWindowSubgraphBuilder: Direct edge assembly without PyG's subgraph method
"""

import torch
import numpy as np
from torch_geometric.data import HeteroData


class WindowSubgraphBuilder:
    """
    Iterative builder for sliding window subgraphs that reuses computation.
    
    This builds subgraphs containing flights in a time window plus all edges
    between them and all connected nodes. Uses an incremental approach where
    consecutive windows reuse overlapping data to minimize rebuilding.
    """
    
    def __init__(self, graph, unit: int | None = None, learn_window: int | None = None, pred_window: int | None = None, window_stride: int | None = None, use_gpu_resident: bool = False):
        """
        Initialize the builder with the full graph.
        
        Args:
            graph: Full HeteroData graph
            unit: Time unit size in minutes (optional, enables per-unit caching)
            learn_window: learn window length in units (optional)
            pred_window: pred window length in units (optional)
            window_stride: stride in units (optional)
            use_gpu_resident: Keep graph on GPU and build subgraphs directly in GPU memory (default: False)
        """
        self.graph = graph
        self.graph_device = graph["flight"].x.device
        self.use_gpu_resident = use_gpu_resident
        self.working_device = self.graph_device if use_gpu_resident else torch.device('cpu')
        self.last_window_flights = None
        self.last_subgraph_data = None
        # Cache edge indices on working device (CPU or GPU if gpu_resident)
        self.edge_index_cpu = {}
        for etype in graph.edge_types:
            self.edge_index_cpu[etype] = graph[etype].edge_index.to(self.working_device)
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
                    (min_u + offset): torch.tensor(arr, dtype=torch.long, device=self.working_device) if len(arr) > 0 else torch.tensor([], dtype=torch.long, device=self.working_device)
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
            ptr = torch.empty(n_rows + 1, dtype=torch.long, device=self.working_device)
            ptr[0] = 0
            torch.cumsum(counts, dim=0, out=ptr[1:])
            return ptr.to(self.working_device), cols_sorted.to(self.working_device)
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
            self.node_refcounts[ntype] = torch.zeros(n_nodes, dtype=torch.int32, device=self.working_device)
        
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
            window_time_range: Optional (start_unit, end_unit) for incremental path
            
        Returns:
            subgraph: Induced HeteroData subgraph
            local_pred_mask: Boolean tensor marking pred flights in subgraph ordering
        """
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
                window_flights_tensor = torch.tensor([], dtype=torch.long, device=self.working_device)
        else:
            window_flights = np.unique(np.concatenate([learn_indices, pred_indices]))
            window_flights_tensor = torch.from_numpy(window_flights).long().to(self.working_device)
            window_flights_tensor = torch.unique(window_flights_tensor, sorted=True)

        # Rolling update using CSR and refcounts if we have per-unit windows
        node_selections = {}
        node_selections['flight'] = window_flights_tensor

        if window_time_range is not None and self.flights_by_unit is not None:
            # Determine added/removed flights vs last window
            if self.last_window_flights is None:
                added = window_flights_tensor
                removed = torch.tensor([], dtype=torch.long, device=self.working_device)
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
                        self.node_refcounts[dst_type].index_add_(0, ncat, torch.full((ncat.numel(),), delta, dtype=torch.int32, device=self.working_device))
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
                        self.node_refcounts[src_type].index_add_(0, ncat, torch.full((ncat.numel(),), delta, dtype=torch.int32, device=self.working_device))

            _update_counts(added, +1)
            _update_counts(removed, -1)

            # Build node selections from positive refcounts
            for ntype, counts in self.node_refcounts.items():
                sel = torch.nonzero(counts > 0, as_tuple=False).view(-1)
                node_selections[ntype] = sel
        else:
            # Fallback: compute neighbors fresh via masking (still efficient using flight_mask)
            flight_mask = torch.zeros(self.num_flights, dtype=torch.bool, device=self.working_device)
            flight_mask[window_flights_tensor] = True
            for edge_type in self.graph.edge_types:
                src_type, rel_name, dst_type = edge_type
                edge_index = self.edge_index_cpu[edge_type]
                if src_type == 'flight':
                    ei0, ei1 = edge_index[0], edge_index[1]
                    connected = torch.unique(ei1[flight_mask[ei0]])
                    if connected.numel() > 0:
                        node_selections[dst_type] = torch.unique(torch.cat([
                            node_selections.get(dst_type, torch.tensor([], dtype=torch.long, device=self.working_device)), connected
                        ]))
                if dst_type == 'flight':
                    ei0, ei1 = edge_index[0], edge_index[1]
                    connected = torch.unique(ei0[flight_mask[ei1]])
                    if connected.numel() > 0:
                        node_selections[src_type] = torch.unique(torch.cat([
                            node_selections.get(src_type, torch.tensor([], dtype=torch.long, device=self.working_device)), connected
                        ]))
        
        # Build induced subgraph using PyG's subgraph method
        # CRITICAL: PyG's subgraph expects indices on same device as graph
        # Move node_selections to graph's device, ensure sorted for stable ordering
        node_selections_for_pyg = {}
        for ntype in node_selections.keys():
            sel = node_selections[ntype].to(self.graph_device).long()
            node_selections_for_pyg[ntype] = torch.unique(sel, sorted=True)
        
        subgraph = self.graph.subgraph(node_selections_for_pyg)
        
        # Compute local pred_mask
        pred_indices_tensor = torch.from_numpy(pred_indices).long().to(self.working_device)
        pred_indices_tensor = torch.unique(pred_indices_tensor, sorted=True)
        # Mask corresponds to the order of node_selections['flight'] which we controlled
        local_pred_mask = torch.isin(node_selections_for_pyg['flight'], pred_indices_tensor.to(self.graph_device))
        
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


class OptimizedWindowSubgraphBuilder:
    """
    Highly optimized sliding window subgraph builder with:
    - GPU-resident graph support (zero-copy subgraph building)
    - Vectorized CSR neighbor gathering
    - Direct edge assembly without PyG's subgraph method
    - Refcount-based rolling updates for O(delta) performance
    """
    
    def __init__(self, graph, unit=None, learn_window=None, pred_window=None, window_stride=None, use_gpu_resident=False):
        """
        Initialize the optimized builder.
        
        Args:
            graph: Full HeteroData graph
            unit: Time unit size in minutes (enables per-unit caching)
            learn_window: learn window length in units
            pred_window: pred window length in units
            window_stride: stride in units
            use_gpu_resident: Keep graph on GPU and build subgraphs directly in GPU memory
        """
        self.graph = graph
        self.graph_device = graph["flight"].x.device
        self.use_gpu_resident = use_gpu_resident
        self.working_device = self.graph_device if use_gpu_resident else torch.device('cpu')
        self.last_window_flights = None
        self.last_subgraph_data = None
        
        print(f"Initializing OptimizedWindowSubgraphBuilder (GPU-resident: {use_gpu_resident}, device: {self.working_device})")
        
        # Cache edge indices and attributes on working device
        self.edge_index_cache = {}
        self.edge_attrs = {}
        for etype in graph.edge_types:
            ei = graph[etype].edge_index
            self.edge_index_cache[etype] = ei.to(self.working_device)
            # Cache all edge attributes
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
        
        # Per-unit flight cache
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
                    (min_u + offset): torch.tensor(arr, dtype=torch.long, device=self.working_device) if len(arr) > 0 else torch.tensor([], dtype=torch.long, device=self.working_device)
                    for offset, arr in enumerate(buckets)
                }
                self.unit_min = min_u
                print(f"  Per-unit cache built: {len(self.flights_by_unit)} units")
            except Exception as e:
                print(f"  Per-unit cache failed: {e}")
                self.flights_by_unit = None
        
        # Build CSR adjacency on working device
        self.csr_src = {}
        self.csr_dst = {}
        def _build_csr(rows, cols, n_rows):
            order = torch.argsort(rows, stable=True)
            rows_sorted = rows[order]
            cols_sorted = cols[order]
            counts = torch.bincount(rows_sorted, minlength=n_rows)
            ptr = torch.empty(n_rows + 1, dtype=torch.long, device=self.working_device)
            ptr[0] = 0
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
        
        # Node refcounts for rolling updates
        self.node_refcounts = {}
        for ntype in graph.node_types:
            if ntype == 'flight':
                continue
            n_nodes = None
            x = self.node_features[ntype].get('x')
            if x is not None:
                n_nodes = x.size(0)
            if n_nodes is not None:
                self.node_refcounts[ntype] = torch.zeros(n_nodes, dtype=torch.int32, device=self.working_device)
        
        print(f"  Refcount tracking for {len(self.node_refcounts)} node types")
    
    def _vectorized_gather_neighbors(self, flight_indices):
        """Vectorized CSR neighbor gathering."""
        neighbors = {}
        
        # Gather from csr_src (flight -> dst_type)
        for dst_type, csr in self.csr_src.items():
            ptr, col = csr['ptr'], csr['col']
            starts = ptr[flight_indices]
            ends = ptr[flight_indices + 1]
            total = (ends - starts).sum().item()
            if total > 0:
                indices = []
                for s, e in zip(starts.tolist(), ends.tolist()):
                    if e > s:
                        indices.append(torch.arange(s, e, device=self.working_device))
                if indices:
                    flat_idx = torch.cat(indices)
                    neighbors[dst_type] = col[flat_idx].unique()
        
        # Gather from csr_dst (src_type -> flight)
        for src_type, csr in self.csr_dst.items():
            ptr, col = csr['ptr'], csr['col']
            starts = ptr[flight_indices]
            ends = ptr[flight_indices + 1]
            total = (ends - starts).sum().item()
            if total > 0:
                indices = []
                for s, e in zip(starts.tolist(), ends.tolist()):
                    if e > s:
                        indices.append(torch.arange(s, e, device=self.working_device))
                if indices:
                    flat_idx = torch.cat(indices)
                    neigh = col[flat_idx].unique()
                    if src_type in neighbors:
                        neighbors[src_type] = torch.unique(torch.cat([neighbors[src_type], neigh]))
                    else:
                        neighbors[src_type] = neigh
        
        return neighbors
    
    def _build_subgraph_direct(self, node_selections, flight_nodes):
        """Build subgraph manually without PyG's subgraph method."""
        subgraph = HeteroData()
        
        # Build node mappings (global -> local)
        node_maps = {}
        for ntype, global_ids in node_selections.items():
            global_ids_sorted = torch.unique(global_ids, sorted=True)
            # Get a sample tensor to determine size
            sample_feat = next(iter(self.node_features[ntype].values()))
            n_global = sample_feat.size(0)
            
            local_map = torch.full((n_global,), -1, dtype=torch.long, device=self.working_device)
            local_map[global_ids_sorted] = torch.arange(global_ids_sorted.numel(), device=self.working_device)
            
            node_maps[ntype] = {
                'global_ids': global_ids_sorted,
                'local_map': local_map
            }
            
            # Copy node features
            for attr_name, attr_val in self.node_features[ntype].items():
                subgraph[ntype][attr_name] = attr_val[global_ids_sorted]
        
        # Build edges
        for etype in self.graph.edge_types:
            src_type, rel, dst_type = etype
            if src_type not in node_maps or dst_type not in node_maps:
                continue
            
            edge_index = self.edge_index_cache[etype]
            src_map = node_maps[src_type]['local_map']
            dst_map = node_maps[dst_type]['local_map']
            
            # Filter: both src and dst in subgraph
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
    
    def build_subgraph(self, learn_indices, pred_indices, device=None, window_time_range=None):
        """Build subgraph with all optimizations."""
        import numpy as np
        
        # Get window flights
        if window_time_range is not None and self.flights_by_unit is not None:
            start_u, end_u = window_time_range
            flights_list = []
            for u in range(start_u, end_u + 1):
                bucket = self.flights_by_unit.get(u)
                if bucket is not None and bucket.numel() > 0:
                    flights_list.append(bucket)
            if flights_list:
                window_flights_tensor = torch.unique(torch.cat(flights_list), sorted=True)
            else:
                window_flights_tensor = torch.tensor([], dtype=torch.long, device=self.working_device)
        else:
            window_flights = np.unique(np.concatenate([learn_indices, pred_indices]))
            window_flights_tensor = torch.from_numpy(window_flights).long().to(self.working_device)
            window_flights_tensor = torch.unique(window_flights_tensor, sorted=True)
        
        # Build node selections
        node_selections = {}
        node_selections['flight'] = window_flights_tensor
        
        if window_time_range is not None and self.flights_by_unit is not None and self.last_window_flights is not None:
            # Rolling update
            added = window_flights_tensor[~torch.isin(window_flights_tensor, self.last_window_flights)]
            removed = self.last_window_flights[~torch.isin(self.last_window_flights, window_flights_tensor)]
            
            def _update_counts(flights, delta):
                if flights.numel() == 0:
                    return
                neigh = self._vectorized_gather_neighbors(flights)
                for ntype, nodes in neigh.items():
                    if ntype in self.node_refcounts:
                        self.node_refcounts[ntype].index_add_(0, nodes, torch.full((nodes.numel(),), delta, dtype=torch.int32, device=self.working_device))
            
            _update_counts(added, +1)
            _update_counts(removed, -1)
            
            # Select from refcounts
            for ntype, counts in self.node_refcounts.items():
                sel = torch.nonzero(counts > 0, as_tuple=False).view(-1)
                node_selections[ntype] = sel
        else:
            # Fresh computation
            flight_mask = torch.zeros(self.num_flights, dtype=torch.bool, device=self.working_device)
            flight_mask[window_flights_tensor] = True
            for etype in self.graph.edge_types:
                src_type, _, dst_type = etype
                ei = self.edge_index_cache[etype]
                if src_type == 'flight':
                    connected = torch.unique(ei[1][flight_mask[ei[0]]])
                    if connected.numel() > 0:
                        node_selections[dst_type] = torch.unique(torch.cat([
                            node_selections.get(dst_type, torch.tensor([], dtype=torch.long, device=self.working_device)), connected
                        ]))
                if dst_type == 'flight':
                    connected = torch.unique(ei[0][flight_mask[ei[1]]])
                    if connected.numel() > 0:
                        node_selections[src_type] = torch.unique(torch.cat([
                            node_selections.get(src_type, torch.tensor([], dtype=torch.long, device=self.working_device)), connected
                        ]))
        
        # Build subgraph directly
        subgraph = self._build_subgraph_direct(node_selections, window_flights_tensor)
        
        # Compute pred mask
        pred_indices_tensor = torch.from_numpy(pred_indices).long().to(self.working_device)
        pred_indices_tensor = torch.unique(pred_indices_tensor, sorted=True)
        local_pred_mask = torch.isin(node_selections['flight'], pred_indices_tensor)
        
        # Move to target device if needed
        if device is not None and device != self.working_device:
            subgraph = subgraph.to(device)
            local_pred_mask = local_pred_mask.to(device)
        
        # Store for next iteration
        self.last_window_flights = window_flights_tensor
        self.last_subgraph_data = node_selections
        
        return subgraph, local_pred_mask
