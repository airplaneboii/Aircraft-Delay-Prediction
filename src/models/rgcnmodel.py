import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F


class RGCN(nn.Module):
    """
    Relational Graph Convolutional Network (RGCN) for heterogeneous graphs.
    Uses PyG's RGCNConv which explicitly models different relation types.
    """
    def __init__(self, metadata, in_channels_dict, hidden_channels=64, out_channels=1, num_layers=2, dropout=0.2, num_bases=None):
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
        self.encoders = nn.ModuleDict({
            node_type: nn.Linear(in_channels_dict[node_type], hidden_channels)
            for node_type in self.node_types
        })

        # RGCN layers: each layer explicitly handles relation types
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv = RGCNConv(
                hidden_channels,
                hidden_channels,
                num_relations=len(self.edge_types),
                num_bases=num_bases
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
        }

        # Convert x_dict to a single tensor (concatenate all node types)
        # and build a mapping for relation indices
        node_type_to_idx = {nt: i for i, nt in enumerate(self.node_types)}
        node_offsets = {}
        offset = 0
        for node_type in self.node_types:
            node_offsets[node_type] = offset
            offset += x_dict[node_type].size(0)
        
        total_nodes = offset
        x = torch.zeros(total_nodes, self.hidden_channels, device=x_dict[self.node_types[0]].device, dtype=x_dict[self.node_types[0]].dtype)
        
        # Fill x tensor with node features from each type
        for node_type in self.node_types:
            start = node_offsets[node_type]
            end = start + x_dict[node_type].size(0)
            x[start:end] = x_dict[node_type]

        # Convert edge_index_dict to a single edge_index and relation type indices
        edge_index_list = []
        relation_type_list = []
        for rel_idx, (src_type, rel_name, dst_type) in enumerate(self.edge_types):
            if (src_type, rel_name, dst_type) in edge_index_dict:
                edge_idx = edge_index_dict[(src_type, rel_name, dst_type)]
                # Offset node indices by their cumulative position
                src_offset = node_offsets[src_type]
                dst_offset = node_offsets[dst_type]
                offset_edge_idx = edge_idx.clone()
                offset_edge_idx[0] += src_offset
                offset_edge_idx[1] += dst_offset
                edge_index_list.append(offset_edge_idx)
                relation_type_list.extend([rel_idx] * edge_idx.size(1))
        
        if edge_index_list:
            edge_index = torch.cat(edge_index_list, dim=1)
            edge_type = torch.tensor(relation_type_list, device=x.device, dtype=torch.long)
        else:
            # Empty graph fallback
            edge_index = torch.zeros((2, 0), device=x.device, dtype=torch.long)
            edge_type = torch.zeros(0, device=x.device, dtype=torch.long)

        # Pass through RGCN layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Extract flight node embeddings and predict
        flight_start = node_offsets["flight"]
        flight_end = flight_start + x_dict["flight"].size(0)
        flight_x = x[flight_start:flight_end]
        
        return self.readout(flight_x).squeeze(-1)