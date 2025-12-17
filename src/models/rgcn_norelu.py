import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F


class RGCNNoReLU(nn.Module):
    def __init__(
        self,
        metadata,
        in_channels_dict,
        hidden_channels=256,
        out_channels=1,
        num_layers=4,
        dropout=0.2,
        num_bases=None,
    ):
        super().__init__()
        self.node_types, self.edge_types = metadata
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        if num_bases is None:
            num_bases = min(len(self.edge_types), 3)

        # Input encoders (NO ReLU)
        self.encoders = nn.ModuleDict({
            node_type: nn.Linear(in_channels_dict[node_type], hidden_channels)
            for node_type in self.node_types
        })

        self.convs = nn.ModuleList([
            RGCNConv(
                hidden_channels,
                hidden_channels,
                num_relations=len(self.edge_types),
                num_bases=num_bases
            )
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_layers)
        ])

        self.readout = nn.Linear(hidden_channels, out_channels)
        nn.init.zeros_(self.readout.bias)

    def forward(self, x_dict, edge_index_dict):
        # Encode (no ReLU)
        x_dict = {
            k: self.encoders[k](v)
            for k, v in x_dict.items()
            if k in self.encoders
        }

        # Offsets
        node_offsets = {}
        offset = 0
        for nt in self.node_types:
            node_offsets[nt] = offset
            if nt in x_dict:
                offset += x_dict[nt].size(0)

        total_nodes = offset
        x = torch.zeros(
            total_nodes,
            self.hidden_channels,
            device=next(self.parameters()).device,
            dtype=next(iter(x_dict.values())).dtype
        )

        for nt, start in node_offsets.items():
            if nt in x_dict:
                x[start:start + x_dict[nt].size(0)] = x_dict[nt]

        # Build edges
        edge_index, edge_type = [], []
        for rel_idx, et in enumerate(self.edge_types):
            if et in edge_index_dict:
                ei = edge_index_dict[et].clone()
                ei[0] += node_offsets[et[0]]
                ei[1] += node_offsets[et[2]]
                edge_index.append(ei)
                edge_type.append(
                    torch.full((ei.size(1),), rel_idx, device=ei.device)
                )

        if edge_index:
            edge_index = torch.cat(edge_index, dim=1)
            edge_type = torch.cat(edge_type)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
            edge_type = torch.empty((0,), dtype=torch.long, device=x.device)

        # RGCN layers with residuals
        for i, conv in enumerate(self.convs):
            x_res = x
            x = conv(x, edge_index, edge_type)
            x = self.norms[i](x)
            x = F.leaky_relu(x, 0.1)
            x = x + x_res
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Flight nodes
        fs = node_offsets["flight"]
        fe = fs + x_dict["flight"].size(0)
        return self.readout(x[fs:fe]).squeeze(-1)

