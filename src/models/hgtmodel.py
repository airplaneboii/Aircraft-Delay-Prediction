import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv
import torch.nn.functional as F
from typing import Dict

class HGT(nn.Module):
    def __init__(self, metadata, in_channels_dict, hidden_channels=64, out_channels=1, num_layers=2, num_heads=2, dropout=0.2):
        super().__init__()
        self.metadata = metadata
        self.node_types, self.edge_types = metadata

        # Convert all node types to same size embeddings
        self.in_proj = nn.ModuleDict()
        for ntype in self.node_types:
            in_dim = in_channels_dict[ntype]
            self.in_proj[ntype] = nn.Sequential(
                nn.Linear(in_dim, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        # HGT layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HGTConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    metadata=metadata,
                    heads=num_heads,
                )
            )

        self.dropout = nn.Dropout(dropout)

        # Final MLP head for flight nodes (regression):
        self.flight_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x_dict, edge_index_dict):
        # Project each node type to hidden space
        x_dict = {ntype: self.in_proj[ntype](x) for ntype, x in x_dict.items()}

        # HGT propagation
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            x_dict = {k: self.dropout(v) for k, v in x_dict.items()}

        flight_out = self.flight_head(x_dict["flight"]).squeeze(-1)
        return flight_out