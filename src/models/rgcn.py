import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class RGCN(nn.Module):
    """
    Relational Graph Convolutional Network (RGCN) for heterogeneous graphs.
    Uses PyG's RGCNConv which explicitly models different relation types.
    """

    def __init__(
        self,
        metadata,
        in_channels_dict,
        hidden_channels=64,
        out_channels=1,
        num_layers=2,
        dropout=0.2,
        num_bases=None,
    ):
        super().__init__()
        self.node_types, self.edge_types = metadata
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        # If num_bases not specified, use min(#edge_types, 3) for basis decomposition
        if num_bases is None:
            num_bases = min(len(self.edge_types), 3)

        # Input encoders: project variable-size node features to hidden_channels
        self.encoders = nn.ModuleDict(
            {
                node_type: nn.Linear(in_channels_dict[node_type], hidden_channels)
                for node_type in self.node_types
            }
        )

        # RGCN layers: each layer explicitly handles relation types
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv = RGCNConv(
                hidden_channels,
                hidden_channels,
                num_relations=len(self.edge_types),
                num_bases=num_bases,
            )
            self.convs.append(conv)

        # Output projection (predict on flight nodes only)
        self.readout = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass through RGCN.

        Args:
            x_dict: dict of node features by type
            edge_index_dict: dict of edge indices by relation type

        Returns:
            predictions for flight nodes
        """
        # Encode node features to common hidden dimension
        x_dict = {
            node_type: F.relu(self.encoders[node_type](x))
            for node_type, x in x_dict.items()
            if node_type in self.encoders
        }

        if not x_dict:
            raise ValueError("RGCN received no node features to process.")

        # Prefer model parameter device; fall back to first input tensor device, then CPU
        first_param = next(iter(self.parameters()), None)
        if first_param is not None:
            device = first_param.device
        else:
            # x_dict has at least one entry here
            device = (
                next(iter(x_dict.values())).device if x_dict else torch.device("cpu")
            )
        dtype = next(iter(x_dict.values())).dtype

        # Compute node offsets and total node count
        node_offsets = {}
        offset = 0
        for node_type in self.node_types:
            node_offsets[node_type] = offset
            if node_type in x_dict:
                offset += x_dict[node_type].size(0)
        total_nodes = offset

        # Build the concatenated node feature tensor (always needed, even when reusing edge cache)
        x = torch.empty(total_nodes, self.hidden_channels, device=device, dtype=dtype)
        x.zero_()
        for node_type in self.node_types:
            if node_type not in x_dict:
                continue
            start = node_offsets[node_type]
            end = start + x_dict[node_type].size(0)
            x[start:end] = x_dict[node_type]

        # Build edge_index and edge_type for this forward
        total_edges = 0
        for src_type, rel_name, dst_type in self.edge_types:
            key = (src_type, rel_name, dst_type)
            if key in edge_index_dict:
                total_edges += edge_index_dict[key].size(1)

        if total_edges == 0:
            edge_index = torch.zeros((2, 0), device=device, dtype=torch.long)
            edge_type = torch.zeros(0, device=device, dtype=torch.long)
        else:
            edge_index = torch.empty((2, total_edges), device=device, dtype=torch.long)
            edge_type = torch.empty((total_edges,), device=device, dtype=torch.long)
            pos = 0
            for rel_idx, (src_type, rel_name, dst_type) in enumerate(self.edge_types):
                key = (src_type, rel_name, dst_type)
                if key not in edge_index_dict:
                    continue
                e = edge_index_dict[key].to(device).clone()
                src_offset = node_offsets[src_type]
                dst_offset = node_offsets[dst_type]
                e[0].add_(src_offset)
                e[1].add_(dst_offset)
                n = e.size(1)
                edge_index[:, pos : pos + n] = e
                edge_type[pos : pos + n] = rel_idx
                pos += n

        # Pass through RGCN layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Extract flight node embeddings and predict
        flight_feats = x_dict.get("flight")
        if flight_feats is None:
            raise KeyError("Flight node type is required for RGCN forward pass.")
        flight_start = node_offsets["flight"]
        flight_end = flight_start + flight_feats.size(0)
        flight_x = x[flight_start:flight_end]

        return self.readout(flight_x).squeeze(-1)
