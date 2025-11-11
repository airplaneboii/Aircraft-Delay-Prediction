import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, RGCNConv, HeteroConv, SAGEConv
import torch.nn.functional as F


# NOT FINAL IMPLEMENTATION
# need to adapt to graph structure and data, then test and debug
# currently not a true rgcn, just a GNN for heterogeneous graphs

class RGCN(nn.Module):
    def __init__(self, metadata, in_channels_dict, hidden_channels=64, out_channels=1, num_layers=2, dropout=0.2):

        super().__init__()
        self.node_types, self.edge_types = metadata
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        #Encode to a common hidden dim
        self.encoders = nn.ModuleDict({
            nodeType: nn.Linear(in_channels_dict[nodeType], hidden_channels)
            for nodeType in self.node_types
        })

        # Message passing layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroConv(
                {
                    edgeType: SAGEConv(hidden_channels, hidden_channels)
                    for edgeType in self.edge_types
                },
                aggr="sum"
            )
            self.convs.append(conv)


        self.readout = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            nodeType: F.relu(self.encoders[nodeType](x))
            for nodeType, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {nodeType: F.relu(x) for nodeType, x in x_dict.items()}
            x_dict = {nodeType: F.dropout(x, p=self.dropout, training=self.training) for nodeType, x in x_dict.items()}
        
        return self.readout(x_dict["flight"])
