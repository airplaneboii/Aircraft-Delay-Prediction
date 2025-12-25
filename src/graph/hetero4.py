import torch
import numpy as np
from torch_geometric.data import HeteroData
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from src.utils import hhmm_to_minutes

class HeteroGraph4:
    """
    HeteroGraph4: Optimized for sliding-window temporal training with single graph + masks.
    
    Key differences from HeteroGraph3:
    - Builds ONE graph from entire dataset
    - Stores snapshot IDs and timestamps on flight nodes for temporal masking
    - Includes ARR_DELAY in flight features (masked during val/test via separate feature)
    - Designed for efficient windowing without rebuilding graphs
    """

    def __init__(self, df, args, train_index, val_index, test_index, norm_stats=None):
        self.df = df
        self.args = args
        self.train_index = train_index
        self.val_index = val_index
        self.test_index = test_index
        self.norm_stats = norm_stats
        self.classification = args.prediction_type == "classification"

    
    def build(self):
        data = HeteroData()

        # Fit statistics on training rows only to avoid leakage
        train_df = self.df.iloc[self.train_index]

        # Nodes (Airports, Aircrafts, Airlines, Flights)
        airports = sorted(set(self.df["ORIGIN_AIRPORT_ID"]).union(set(self.df["DEST_AIRPORT_ID"])))
        aircrafts = sorted(self.df["TAIL_NUM"].unique())
        airlines = sorted(self.df["OP_CARRIER_AIRLINE_ID"].unique())

        airport_map = {a: i for i, a in enumerate(airports)}
        aircraft_map = {a: i for i, a in enumerate(aircrafts)}
        airline_map = {a: i for i, a in enumerate(airlines)}

        # Ensure FL_DATE exists
        if "FL_DATE" not in self.df.columns or not pd.api.types.is_datetime64_any_dtype(self.df["FL_DATE"]):
            self.df['FL_DATE'] = pd.to_datetime(self.df[['YEAR', 'MONTH', 'DAY_OF_MONTH']].rename(columns={'YEAR': 'year', 'MONTH': 'month', 'DAY_OF_MONTH': 'day'}))

        ######### AIRPORT FEATURES ############
        
        # Use training data for statistics
        origin_counts = train_df.groupby("ORIGIN_AIRPORT_ID").size()
        dest_counts = train_df.groupby("DEST_AIRPORT_ID").size()
        
        airport_origin_counts = pd.Series({a: origin_counts.get(a, 0) for a in airports})
        airport_dest_counts = pd.Series({a: dest_counts.get(a, 0) for a in airports})
        
        # Normalize counts
        max_origin = airport_origin_counts.max() if airport_origin_counts.max() > 0 else 1
        max_dest = airport_dest_counts.max() if airport_dest_counts.max() > 0 else 1
        airport_origin_norm = airport_origin_counts / max_origin
        airport_dest_norm = airport_dest_counts / max_dest

        # Categorical features from lookup data
        airport_info = self.df[["ORIGIN_AIRPORT_ID", "ORIGIN_CITY_MARKET_ID", "ORIGIN_WAC"]].drop_duplicates(subset=["ORIGIN_AIRPORT_ID"])
        airport_info_dict = {row["ORIGIN_AIRPORT_ID"]: row for _, row in airport_info.iterrows()}

        dest_info = self.df[["DEST_AIRPORT_ID", "DEST_CITY_MARKET_ID", "DEST_WAC"]].drop_duplicates(subset=["DEST_AIRPORT_ID"])
        dest_info_dict = {row["DEST_AIRPORT_ID"]: row for _, row in dest_info.iterrows()}
        airport_info_dict.update(dest_info_dict)

        all_city_markets = sorted(set(self.df["ORIGIN_CITY_MARKET_ID"]).union(set(self.df["DEST_CITY_MARKET_ID"])))
        all_wac = sorted(set(self.df["ORIGIN_WAC"]).union(set(self.df["DEST_WAC"])))

        city_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        city_encoder.fit(np.array(all_city_markets).reshape(-1, 1))
        
        wac_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        wac_encoder.fit(np.array(all_wac).reshape(-1, 1))

        airport_features = []
        for a in airports:
            info = airport_info_dict.get(a, {})
            city_id = info.get("ORIGIN_CITY_MARKET_ID", info.get("DEST_CITY_MARKET_ID", all_city_markets[0]))
            wac_id = info.get("ORIGIN_WAC", info.get("DEST_WAC", all_wac[0]))
            
            city_onehot = city_encoder.transform([[city_id]])[0]
            wac_onehot = wac_encoder.transform([[wac_id]])[0]
            
            feat = np.concatenate([[airport_origin_norm[a], airport_dest_norm[a]], city_onehot, wac_onehot])
            airport_features.append(feat)

        data["airport"].x = torch.tensor(np.array(airport_features), dtype=torch.float32)

        ######### AIRCRAFT FEATURES ############
        
        aircraft_counts = train_df.groupby("TAIL_NUM").size()
        aircraft_norm_counts = {a: aircraft_counts.get(a, 0) / aircraft_counts.max() if aircraft_counts.max() > 0 else 0 for a in aircrafts}
        
        aircraft_features = []
        for ac in aircrafts:
            feat = [aircraft_norm_counts[ac], 0, 0, 0]  # Placeholder features
            aircraft_features.append(feat)
        
        data["aircraft"].x = torch.tensor(np.array(aircraft_features), dtype=torch.float32)

        ######### AIRLINE FEATURES ############
        
        airline_counts = train_df.groupby("OP_CARRIER_AIRLINE_ID").size()
        airline_norm_counts = {a: airline_counts.get(a, 0) / airline_counts.max() if airline_counts.max() > 0 else 0 for a in airlines}
        
        airline_features = []
        for al in airlines:
            feat = [airline_norm_counts[al], 0, 0, 0, 0]  # Placeholder features
            airline_features.append(feat)
        
        data["airline"].x = torch.tensor(np.array(airline_features), dtype=torch.float32)

        ######### FLIGHT FEATURES (WITH ARR_DELAY AND TIMESTAMPS) ############
        
        # Convert departure time to minutes
        self.df["dep_minutes"] = self.df["CRS_DEP_TIME"].apply(hhmm_to_minutes)
        self.df["arr_minutes"] = self.df["CRS_ARR_TIME"].apply(hhmm_to_minutes)
        
        # Create timestamps
        self.df["dep_timestamp"] = self.df["FL_DATE"] + pd.to_timedelta(self.df["dep_minutes"], unit="m")
        self.df["arr_timestamp"] = self.df["FL_DATE"] + pd.to_timedelta(self.df["arr_minutes"], unit="m")
        
        # Normalize timestamps to [0, 1] range for neural network
        min_timestamp = self.df["dep_timestamp"].min()
        max_timestamp = self.df["dep_timestamp"].max()
        timestamp_range = (max_timestamp - min_timestamp).total_seconds()
        
        if timestamp_range > 0:
            self.df["dep_timestamp_norm"] = ((self.df["dep_timestamp"] - min_timestamp).dt.total_seconds() / timestamp_range).astype(np.float32)
        else:
            self.df["dep_timestamp_norm"] = 0.0
        
        # Flight features including ARR_DELAY (will be masked during val/test)
        flight_features = []
        train_index_set = set(self.train_index)
        for idx, row in self.df.iterrows():
            feat = [
                row["dep_minutes"] / 1440.0,  # Normalized to [0, 1] (24h = 1440min)
                row["arr_minutes"] / 1440.0,
                row["CRS_ELAPSED_TIME"] / 600.0 if pd.notna(row["CRS_ELAPSED_TIME"]) else 0,  # Normalized
                row["DISTANCE"] / 5000.0 if pd.notna(row["DISTANCE"]) else 0,  # Normalized
                row["dep_timestamp_norm"],  # Temporal position
                row["DAY_OF_WEEK"] / 7.0,
                row["MONTH"] / 12.0,
                row["DAY_OF_MONTH"] / 31.0,
                # ARR_DELAY feature - this will be masked during val/test
                row["ARR_DELAY"] / 100.0 if pd.notna(row["ARR_DELAY"]) and idx in train_index_set else 0.0,
                1.0 if idx in train_index_set else 0.0,  # is_training mask
            ]
            flight_features.append(feat)
        
        data["flight"].x = torch.tensor(np.array(flight_features), dtype=torch.float32)
        
        # Store original timestamps for temporal analysis
        data["flight"].timestamp = torch.tensor(self.df["dep_timestamp_norm"].values, dtype=torch.float32)
        
        # Store original indices for mask operations
        data["flight"].original_index = torch.arange(len(self.df), dtype=torch.long)

        ######### EDGES ############
        
        # Precompute flight indices for vectorized edge construction
        flight_indices = np.arange(len(self.df), dtype=np.int64)

        # Origin edges (vectorized)
        origin_nodes = self.df["ORIGIN_AIRPORT_ID"].map(airport_map).to_numpy()
        origin_edges = np.stack([flight_indices, origin_nodes], axis=0)
        data["flight", "originates_from", "airport"].edge_index = torch.tensor(origin_edges, dtype=torch.long)
        data["airport", "has_departure", "flight"].edge_index = torch.tensor(origin_edges[[1,0], :], dtype=torch.long)
        
        # Destination edges (vectorized)
        dest_nodes = self.df["DEST_AIRPORT_ID"].map(airport_map).to_numpy()
        dest_edges = np.stack([flight_indices, dest_nodes], axis=0)
        data["flight", "arrives_at", "airport"].edge_index = torch.tensor(dest_edges, dtype=torch.long)
        data["airport", "has_arrival", "flight"].edge_index = torch.tensor(dest_edges[[1,0], :], dtype=torch.long)
        
        # Airline edges (vectorized)
        airline_nodes = self.df["OP_CARRIER_AIRLINE_ID"].map(airline_map).to_numpy()
        airline_edges = np.stack([flight_indices, airline_nodes], axis=0)
        data["flight", "operated_by", "airline"].edge_index = torch.tensor(airline_edges, dtype=torch.long)
        data["airline", "operates", "flight"].edge_index = torch.tensor(airline_edges[[1,0], :], dtype=torch.long)
        
        # Aircraft edges (vectorized)
        aircraft_nodes = self.df["TAIL_NUM"].map(aircraft_map).to_numpy()
        aircraft_edges = np.stack([flight_indices, aircraft_nodes], axis=0)
        data["flight", "performed_by", "aircraft"].edge_index = torch.tensor(aircraft_edges, dtype=torch.long)
        data["aircraft", "performs", "flight"].edge_index = torch.tensor(aircraft_edges[[1,0], :], dtype=torch.long)

        ######### TEMPORAL EDGES ############
        
        # Reuse current ordering (already time-sorted by loader) for temporal edge construction
        def _next_edges(group_col: str):
            edges = []
            for _, grp in self.df.groupby(group_col, sort=False):
                idx = grp.index.to_numpy()
                if idx.size > 1:
                    edges.append(np.column_stack([idx[:-1], idx[1:]]))
            return edges

        # Next same aircraft edges
        aircraft_next = _next_edges("TAIL_NUM")
        if aircraft_next:
            next_aircraft_edges = torch.tensor(np.concatenate(aircraft_next, axis=0), dtype=torch.long).t().contiguous()
            data["flight", "next_same_aircraft", "flight"].edge_index = next_aircraft_edges

        # Next same origin airport edges
        origin_next = _next_edges("ORIGIN_AIRPORT_ID")
        if origin_next:
            next_origin_edges = torch.tensor(np.concatenate(origin_next, axis=0), dtype=torch.long).t().contiguous()
            data["flight", "next_same_origin", "flight"].edge_index = next_origin_edges

        # Next same airline edges
        airline_next = _next_edges("OP_CARRIER_AIRLINE_ID")
        if airline_next:
            next_airline_edges = torch.tensor(np.concatenate(airline_next, axis=0), dtype=torch.long).t().contiguous()
            data["flight", "next_same_airline", "flight"].edge_index = next_airline_edges

        ######### LABELS AND MASKS ############
        
        if self.classification:
            border = getattr(self.args, "border", 0.45)
            labels = (self.df["ARR_DELAY"] >= border * 60).astype(int).values
            data["flight"].y = torch.tensor(labels, dtype=torch.long).unsqueeze(1)
        else:
            labels = self.df["ARR_DELAY"].values
            data["flight"].y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        
        # Store masks for train/val/test splits
        train_mask = torch.zeros(len(self.df), dtype=torch.bool)
        train_mask[self.train_index] = True
        data["flight"].train_mask = train_mask
        
        val_mask = torch.zeros(len(self.df), dtype=torch.bool)
        val_mask[self.val_index] = True
        data["flight"].val_mask = val_mask
        
        test_mask = torch.zeros(len(self.df), dtype=torch.bool)
        test_mask[self.test_index] = True
        data["flight"].test_mask = test_mask

        return data
