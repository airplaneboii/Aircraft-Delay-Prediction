import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GraphConv, Linear

###############################################################
# GENERIC HETEROGENEOUS GRAPH NEURAL NETWORK (DUMMY)
###############################################################
class DummyModel(nn.Module):
    def __init__(
            self,
            metadata,
            in_channels_dict: dict,
            hidden_channels: int,
            out_channels: int
            ) -> None:
        super().__init__()
        self.node_types, self.edge_types = metadata
        self.hidden_channels = hidden_channels

        # Per-node-type input projection
        self.node_proj = nn.ModuleDict({
            ntype: Linear(in_channels_dict.get(ntype, hidden_channels), hidden_channels)
            for ntype in self.node_types
        })

        # Use GraphConv which supports bipartite message passing
        self.conv = HeteroConv({
            etype: GraphConv(hidden_channels, hidden_channels)
            for etype in self.edge_types
        }, aggr='mean')

        # Output projection for target node type (look for 'flight' node or use first)
        # Training expects predictions for 'flight' nodes
        if 'flight' in self.node_types:
            self.target_node_type = 'flight'
        else:
            self.target_node_type = self.node_types[0] if self.node_types else None
        self.out_proj = Linear(hidden_channels, out_channels)

        from src.utils import get_logger
        logger = get_logger()
        logger.info("DummyModel: %d node types, %d edge types", len(self.node_types), len(self.edge_types))
        logger.info("DummyModel: Outputting predictions for '%s' nodes", self.target_node_type)

    def forward(self, x_dict, edge_index_dict):
        # Project each node type to hidden dimension (skip unknowns defensively)
        x_dict = {
            node_type: self.node_proj[node_type](x)
            for node_type, x in x_dict.items()
            if node_type in self.node_proj
        }

        # Apply heterogeneous convolution
        x_dict = self.conv(x_dict, edge_index_dict)

        # ReLU activation
        x_dict = {node_type: x.relu() for node_type, x in x_dict.items()}

        if self.target_node_type is None or self.target_node_type not in x_dict:
            # If no target, return zeros matching first available node type
            if len(x_dict) == 0:
                return torch.zeros(0, device=next(self.parameters()).device)
            first_ntype = next(iter(x_dict))
            out = torch.zeros(x_dict[first_ntype].size(0), self.out_proj.out_features, device=x_dict[first_ntype].device)
            return out.squeeze(-1)

        return self.out_proj(x_dict[self.target_node_type]).squeeze(-1)
