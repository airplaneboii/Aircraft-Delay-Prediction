import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, RGCNConv, HeteroConv
import torch.nn.functional as F


# NOT FINAL IMPLEMENTATION
# might not work, need to adapt to graph structure and data, then test and debug
class RGCN(nn.Module):
    def __init__(self, metadata, input_dims, hidden_dim=64, num_layers=2, num_classes=2, num_bases=4, dropout=0.2):
        """   
        Args:
            metadata: (node_types, edge_types) from HeteroData.metadata()
            hidden_dim: embedding size
            num_layers: number of GNN layers
            num_classes: output classes (2 for delayed/not delayed)
        """
        super().__init__()
        
        self.node_types, self.edge_types = metadata
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout


        # Node-type specific input encoders
        self.node_encoders = nn.ModuleDict({
        node_type: nn.Linear(input_dims[node_type], hidden_dim)
        for node_type in self.node_types
        })

        # Relational GCN layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            convs = {
                edge_type: RGCNConv((-1, -1), hidden_dim, num_relations=1, num_bases=num_bases) #Maybe we use fastRGCNConv for faster computing?(uses more memory)
                for edge_type in self.edge_types
            }
            self.layers.append(HeteroConv(convs, aggr='sum'))

        # Classification heads (for delay and cancellation)
        self.classifier_delay = nn.Linear(hidden_dim, num_classes)
        self.classifier_cancel = nn.Linear(hidden_dim, num_classes)

        def forward(self, x_dict, edge_index_dict):

            # Encode inputs per node type
            x_dict = {k: F.relu(self.node_encoders[k](v)) for k, v in x_dict.items()}

            # Message passing
            for layer in self.layers:
                new_x_dict = layer(x_dict, edge_index_dict)
                x_dict = {
                    k: F.dropout(F.relu(v), p=self.dropout, training=self.training) + x_dict[k]
                    for k, v in new_x_dict.items()
                }

            # Prepare outputs
            out = {
                'embeddings': x_dict,
                'delay_outputs': {},
                'cancel_outputs': {}
            }

            # Classification per node type
            for node_type, z in x_dict.items():
                out['delay_outputs'][node_type] = self.classifier_delay(z)
                out['cancel_outputs'][node_type] = self.classifier_cancel(z)

            return out
