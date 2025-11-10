import torch
from torch_geometric.data import HeteroData

class BaseGraph:
    def __init__(self, df, args):
        self.df = df
        self.args = args

    def build(self):
        data = HeteroData()

        # Nodes (Airports, Aircrafts, Airlines, Causes, Flights)
        airports = sorted(set(self.df["ORIGIN"]).union(set(self.df["DEST"])))
        aircrafts = sorted(self.df["TAIL_NUM"].unique())
        airlines = sorted(self.df["OP_UNIQUE_CARRIER"].unique())
        causes = ["CARRIER", "WEATHER", "NAS", "SECURITY", "LATE_AIRCRAFT"]

        airport_map = {a: i for i, a in enumerate(airports)}
        aircraft_map = {a: i for i, a in enumerate(aircrafts)}
        airline_map = {a: i for i, a in enumerate(airlines)}
        cause_map = {c: i for i, c in enumerate(causes)}

        #TODO: add real features
        data["airport"].x = torch.randn(len(airports), 16)
        data["aircraft"].x = torch.randn(len(aircrafts), 16)
        data["airline"].x = torch.randn(len(airlines), 16)
        data["cause"].x = torch.randn(len(causes), 8)

        num_flights = len(self.df) #Every line in data is a different flight
        data["flight"].x = torch.randn(num_flights, 32) 

        # Edges

        flight_index = list(range(num_flights))
        origin = [airport_map[o] for o in self.df["ORIGIN"]]
        dest = [airport_map[d] for d in self.df["DEST"]]
        airline = [airline_map[a] for a in self.df["OP_UNIQUE_CARRIER"]]
        aircraft = [aircraft_map[t] for t in self.df["TAIL_NUM"]]

        # Edge 1: flight originates from airport
        data["flight", "originates_from", "airport"].edge_index = torch.tensor([flight_index, origin], dtype=torch.long)

        # Edge 2: flight arrives at airport
        data["flight", "arrives_at", "airport"].edge_index = torch.tensor([flight_index, dest], dtype=torch.long)

        # Edge 3: flight operated by airline
        data["flight", "operated_by", "airline"].edge_index = torch.tensor([flight_index, airline], dtype=torch.long)

        # Edge 4: flight performed by aircraft
        data["flight", "performed_by", "aircraft"].edge_index = torch.tensor([flight_index, aircraft], dtype=torch.long)

        # Edge 5: flight 1 performed by aircraft that later performs flight 2 (temporal link)
        next_src = []
        next_dst = []
        df_sorted = self.df.sort_values(["TAIL_NUM", "FL_DATE", "CRS_DEP_TIME"]).reset_index(drop=True)
        for tail, group in df_sorted.groupby("TAIL_NUM"):
            idx_list = group.index.tolist()
            # link consecutive flights
            for i in range(len(idx_list) - 1):
                next_src.append(idx_list[i])
                next_dst.append(idx_list[i + 1])

        data["flight", "next_same_aircraft", "flight"].edge_index = torch.tensor([next_src, next_dst], dtype=torch.long)

        # Edge 6: flight delayed because of cause
        delay_causes = {
            "CARRIER_DELAY": "CARRIER",
            "WEATHER_DELAY": "WEATHER",
            "NAS_DELAY": "NAS",
            "SECURITY_DELAY": "SECURITY",
            "LATE_AIRCRAFT_DELAY": "LATE_AIRCRAFT"
        }
        src = []
        dst = []

        df_reset = self.df.reset_index(drop=True)

        for i, row in df_reset.iterrows():
            for delay_col, cause in delay_causes.items():
                if row[delay_col] > 0:
                    src.append(i)
                    dst.append(cause_map[cause])

        data["flight", "delayed_because_of", "cause"].edge_index = torch.tensor([src, dst], dtype=torch.long)

        # Edge 7: flight cancelled because of cause
        cancel_causes = {
            "A": "CARRIER",
            "B": "WEATHER", 
            "C": "NAS",
            "D": "SECURITY"
        }

        c_src = []
        c_dst = []
        for i, row in df_reset.iterrows():
            if row["CANCELLED"] == 1 and row["CANCELLATION_CODE"] in cancel_causes:
                cause = cancel_causes[row["CANCELLATION_CODE"]]
                c_src.append(i)
                c_dst.append(cause_map[cause])
        data["flight", "cancelled_because_of", "cause"].edge_index = torch.tensor([c_src, c_dst], dtype=torch.long)

        #Probably not needed anymore
        # data["airport", "flies_to", "airport"].edge_index = torch.tensor([origin, dest], dtype=torch.long)
        # data["aircraft", "operates_from", "airport"].edge_index = torch.tensor([aircraft, origin], dtype=torch.long)


        #label for predicting arrival delays
        #might have to fix so that cancelled flights arent seen as having a delay of 0
        data["flight"].y = torch.tensor(self.df["ARR_DELAY"].fillna(0).values, dtype=torch.float).unsqueeze(1)
        return data
