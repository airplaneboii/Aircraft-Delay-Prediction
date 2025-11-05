import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

####################################
# TODO: IMPLEMENT THE ACTUAL MODEL #
####################################

###############################################################
# TODO: DELETE THIS FILE AFTER IMPLEMENTING THE ACTUAL MODEL! #
###############################################################
class DummyModel(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int
            ) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        print("NOTE: This is only a dummy model. Implement the actual model!")

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x.squeeze(-1)
