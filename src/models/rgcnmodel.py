import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, RGCNConv, HeteroConv
import torch.nn.functional as F


# NOT FINAL IMPLEMENTATION
# need to adapt to graph structure and data, then test and debug



class RGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=1, num_relations=1, num_bases=None, num_layers=2, dropout=0.2):
        """

        Args:
            in_channels: Number of input features per node
            hidden_channels: Hidden embedding dimension
            out_channels: Output dimension (1 for regression, >=2 for classification)
            num_relations: Number of edge types
            num_bases: Number of bases for RGCNConv (optional)
            num_layers: Number of RGCN layers
            dropout: Dropout probability
        """
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # First layer
        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations, num_bases=num_bases))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=num_bases))

        # Output layer
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_type=None):
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge index [2, num_edges]
            edge_type: Tensor of edge types [num_edges] (optional, default all zeros)
        """
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)

        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_type))
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.lin(x)
        return out