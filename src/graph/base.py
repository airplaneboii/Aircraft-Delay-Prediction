import torch
from torch_geometric.data import HeteroData

###########################################################################
# TODO: DELETE THIS FILE AFTER IMPLEMENTING THE ACTUAL GRAPH CONSTRUCTION #
###########################################################################

class BaseGraph:
    def __init__(self, df, args):
        self.df = df
        self.args = args

    def build(self):
        print("NOTE: This is only a basic graph. Implement the actual graph!")
        data = HeteroData()

        # Nodes
        airports = sorted(set(self.df["ORIGIN"]).union(set(self.df["DEST"])))
        aircrafts = sorted(self.df["TAIL_NUM"].unique())

        airport_map = {a: i for i, a in enumerate(airports)}
        aircraft_map = {a: i for i, a in enumerate(aircrafts)}

        data["airport"].x = torch.eye(len(airports))
        data["aircraft"].x = torch.eye(len(aircrafts))

        # Edges
        src = [airport_map[o] for o in self.df["ORIGIN"]]
        dst = [airport_map[d] for d in self.df["DEST"]]
        data["airport", "flies_to", "airport"].edge_index = torch.tensor([src, dst], dtype=torch.long)

        a_src = [aircraft_map[t] for t in self.df["TAIL_NUM"]]
        a_dst = [airport_map[o] for o in self.df["ORIGIN"]]
        data["aircraft", "operates_from", "airport"].edge_index = torch.tensor([a_src, a_dst], dtype=torch.long)

        # Labels
        airport_labels = torch.zeros(len(airports), dtype=torch.float)
        counts = torch.zeros(len(airports), dtype=torch.float)


        for i, row in self.df.iterrows():
            idx = airport_map[row["ORIGIN"]]
            airport_labels[idx] += row["y"]
            counts[idx] += 1

        counts[counts == 0] = 1
        airport_labels = airport_labels / counts

        data["airport"].y = airport_labels.unsqueeze(1)
        return data
