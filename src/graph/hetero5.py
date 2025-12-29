import torch
import numpy as np
from torch_geometric.data import HeteroData
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from src.utils import hhmm_to_minutes, normalize_with_idx

class HeteroGraph5:
    """
    HeteroGraph5: Graph builder with rich embeddings for heterogeneous GNN training.
    
    Features:
    - Richer aircraft embeddings (statistical features)
    - Richer airline embeddings (statistical features)
    - Cyclical time embeddings for flights (sin/cos for time of day, day of week, month)
    - Optimized for sliding-window temporal training
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

        # Fit encoders once
        city_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        city_encoder.fit(np.array(all_city_markets).reshape(-1, 1))
        wac_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        wac_encoder.fit(np.array(all_wac).reshape(-1, 1))

        # Build arrays of city/wac ids for all airports (vectorized -> single transform call)
        city_ids = []
        wac_ids = []
        for a in airports:
            info = airport_info_dict.get(a, {})
            city_id = info.get("ORIGIN_CITY_MARKET_ID", info.get("DEST_CITY_MARKET_ID", all_city_markets[0]))
            wac_id = info.get("ORIGIN_WAC", info.get("DEST_WAC", all_wac[0]))
            city_ids.append(city_id)
            wac_ids.append(wac_id)
        city_arr = city_encoder.transform(np.array(city_ids).reshape(-1, 1)).astype(np.float32)
        wac_arr = wac_encoder.transform(np.array(wac_ids).reshape(-1, 1)).astype(np.float32)

        # Base numeric features (normalized counts)
        base_counts = np.stack([airport_origin_norm.values.astype(np.float32), airport_dest_norm.values.astype(np.float32)], axis=1)

        airport_features_arr = np.concatenate([base_counts, city_arr, wac_arr], axis=1)
        data["airport"].x = torch.from_numpy(airport_features_arr).float()

        # free temporaries
        del city_arr, wac_arr, city_ids, wac_ids, airport_features_arr, base_counts
        import gc; gc.collect()

        ######### AIRCRAFT FEATURES (RICHER - from hetero3) ############
        
        aircraft_grp = train_df.groupby("TAIL_NUM").agg({
            "DEP_DELAY": "mean",              # Average departure delay for this aircraft
            "DISTANCE": "mean",               # Average flight distance for this aircraft
            "CRS_ELAPSED_TIME": "mean",       # Average block time (make/model differences)
            "TAIL_NUM": "count"               # Number of flights for this aircraft
        }).rename(columns={"TAIL_NUM": "num_of_flights"})
        
        aircraft_features = []
        for tail in aircrafts:
            row = aircraft_grp.loc[tail] if tail in aircraft_grp.index else None
            avg_dep_delay = row["DEP_DELAY"] if row is not None else 0
            avg_distance = row["DISTANCE"] if row is not None else 0
            crs_elapsed_time = row["CRS_ELAPSED_TIME"] if row is not None else 0
            num_of_flights = row["num_of_flights"] if row is not None else 0
            
            feat = [avg_dep_delay, avg_distance, crs_elapsed_time, num_of_flights]
            aircraft_features.append(feat)
        
        # Normalize using train aircraft
        aircraft_arr = np.array(aircraft_features, dtype=np.float32)
        train_aircraft = sorted(train_df["TAIL_NUM"].unique())
        train_aircraft_idx = np.array([aircraft_map[a] for a in train_aircraft if a in aircraft_map], dtype=int)
        fit_idx = train_aircraft_idx if len(train_aircraft_idx) > 0 else np.arange(len(aircraft_arr))
        aircraft_arr, _, _ = normalize_with_idx(aircraft_arr, fit_idx)
        
        data["aircraft"].x = torch.tensor(aircraft_arr, dtype=torch.float32)

        ######### AIRLINE FEATURES (RICHER - from hetero3) ############
        
        airline_grp = train_df.groupby("OP_CARRIER_AIRLINE_ID").agg({
            "DEP_DELAY": "mean",                  # Avg departure delay for this airline
            "DIVERTED": "mean",                   # Diversion rate
            "DISTANCE": "mean",                   # Avg distance flown
            "TAXI_OUT": "mean",                   # Avg taxi-out time (congestion proxy)
            "OP_CARRIER_AIRLINE_ID": "count"      # Number of flights for this airline
        }).rename(columns={"OP_CARRIER_AIRLINE_ID": "num_flights"})
        
        airline_features = []
        for carrier in airlines:
            row = airline_grp.loc[carrier] if carrier in airline_grp.index else None
            avg_dep_delay = row["DEP_DELAY"] if row is not None else 0
            diverted_rate = row["DIVERTED"] if row is not None else 0
            avg_distance = row["DISTANCE"] if row is not None else 0
            avg_taxi_out = row["TAXI_OUT"] if row is not None else 0
            num_flights = row["num_flights"] if row is not None else 0
            
            feat = [avg_dep_delay, diverted_rate, avg_distance, avg_taxi_out, num_flights]
            airline_features.append(feat)
        
        # Normalize using train airlines
        airline_arr = np.array(airline_features, dtype=np.float32)
        train_airlines = sorted(train_df["OP_CARRIER_AIRLINE_ID"].unique())
        train_airline_idx = np.array([airline_map[a] for a in train_airlines if a in airline_map], dtype=int)
        fit_idx = train_airline_idx if len(train_airline_idx) > 0 else np.arange(len(airline_arr))
        airline_arr, _, _ = normalize_with_idx(airline_arr, fit_idx)
        
        data["airline"].x = torch.tensor(airline_arr, dtype=torch.float32)

        ######### FLIGHT FEATURES (WITH CYCLICAL TIME + ARR_DELAY + TIMESTAMPS) ############
        
        # Convert departure time to minutes
        self.df["dep_minutes"] = self.df["CRS_DEP_TIME"].apply(hhmm_to_minutes)
        self.df["arr_minutes"] = self.df["CRS_ARR_TIME"].apply(hhmm_to_minutes)

        # Create timestamps
        self.df["dep_timestamp"] = self.df["FL_DATE"] + pd.to_timedelta(self.df["dep_minutes"], unit="m")
        self.df["arr_timestamp"] = self.df["FL_DATE"] + pd.to_timedelta(self.df["arr_minutes"], unit="m")

        # Normalize timestamps to [0, 1] range for neural network (float32)
        min_timestamp = self.df["dep_timestamp"].min()
        max_timestamp = self.df["dep_timestamp"].max()
        timestamp_range = (max_timestamp - min_timestamp).total_seconds()

        if timestamp_range > 0:
            self.df["dep_timestamp_norm"] = ((self.df["dep_timestamp"] - min_timestamp).dt.total_seconds() / timestamp_range).astype(np.float32)
        else:
            self.df["dep_timestamp_norm"] = np.float32(0.0)

        # Also store absolute minutes since the first timestamp for downstream windowing
        self.df["dep_timestamp_minutes"] = ((self.df["dep_timestamp"] - min_timestamp).dt.total_seconds() / 60.0).astype(np.float32)

        # Cyclical time embeddings (vectorized, float32)
        dep_time_sin = np.sin(2 * np.pi * self.df["dep_minutes"] / (24 * 60)).astype(np.float32)
        dep_time_cos = np.cos(2 * np.pi * self.df["dep_minutes"] / (24 * 60)).astype(np.float32)
        arr_time_sin = np.sin(2 * np.pi * self.df["arr_minutes"] / (24 * 60)).astype(np.float32)
        arr_time_cos = np.cos(2 * np.pi * self.df["arr_minutes"] / (24 * 60)).astype(np.float32)

        day_sin = np.sin(2 * np.pi * self.df["DAY_OF_WEEK"] / 7).astype(np.float32)
        day_cos = np.cos(2 * np.pi * self.df["DAY_OF_WEEK"] / 7).astype(np.float32)

        month_sin = np.sin(2 * np.pi * (self.df["MONTH"] - 1) / 12).astype(np.float32)
        month_cos = np.cos(2 * np.pi * (self.df["MONTH"] - 1) / 12).astype(np.float32)

        # Vectorized flight features (fast, float32)
        n_flights = len(self.df)
        train_mask_np = np.zeros(n_flights, dtype=bool)
        train_mask_np[self.train_index] = True

        dep_minutes_arr = self.df["dep_minutes"].to_numpy(dtype=np.float32)
        crs_elapsed = self.df["CRS_ELAPSED_TIME"].fillna(0.0).to_numpy(dtype=np.float32) / 600.0
        distance_norm = self.df["DISTANCE"].fillna(0.0).to_numpy(dtype=np.float32) / 5000.0
        dep_ts_norm = self.df["dep_timestamp_norm"].to_numpy(dtype=np.float32)
        arr_delay_vals = self.df["ARR_DELAY"].fillna(0.0).to_numpy(dtype=np.float32)

        arr_delay_col = np.where(train_mask_np, arr_delay_vals, 0.0).astype(np.float32)
        is_train_col = train_mask_np.astype(np.float32)

        flight_features_arr = np.column_stack([
            dep_time_sin,
            dep_time_cos,
            arr_time_sin,
            arr_time_cos,
            day_sin,
            day_cos,
            month_sin,
            month_cos,
            crs_elapsed,
            distance_norm,
            dep_ts_norm,
            arr_delay_col,
            is_train_col,
        ]).astype(np.float32)

        data["flight"].x = torch.from_numpy(flight_features_arr).float()

        # Store feature name -> column index mapping for robustness
        feat_dim = data["flight"].x.size(1)
        data["flight"].feat_index = {
            "arr_delay": feat_dim - 2,
            "is_train": feat_dim - 1,
        }

        # Store normalized and absolute minute timestamps on flight nodes (share memory via from_numpy)
        data["flight"].timestamp = torch.from_numpy(dep_ts_norm).float()
        data["flight"].timestamp_min = torch.from_numpy(self.df["dep_timestamp_minutes"].to_numpy(dtype=np.float32)).float()

        # Store original indices for mask operations
        data["flight"].original_index = torch.arange(len(self.df), dtype=torch.long)

        # Free large temporaries
        del dep_time_sin, dep_time_cos, arr_time_sin, arr_time_cos, day_sin, day_cos, month_sin, month_cos
        del crs_elapsed, distance_norm, dep_ts_norm, arr_delay_vals, arr_delay_col, is_train_col, flight_features_arr
        import gc; gc.collect()

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
        
        # Labels: store both regression and classification targets for backward compatibility
        # Regression target (ARR_DELAY) as float32
        arr_delay_vals = self.df["ARR_DELAY"].fillna(0.0).astype(float).values
        data["flight"].y_reg = torch.tensor(arr_delay_vals, dtype=torch.float32).unsqueeze(1)

        # Classification target: prefer ARR_DEL15 column if present, else compute from ARR_DELAY
        if "ARR_DEL15" in self.df.columns:
            cls_vals = self.df["ARR_DEL15"].astype(int).values
        else:
            border = getattr(self.args, "border", 0.45)
            # legacy behavior used border * 60 threshold; keep that for compatibility
            cls_vals = (self.df["ARR_DELAY"].fillna(0.0) >= border * 60).astype(int).values
        data["flight"].y_cls = torch.tensor(cls_vals, dtype=torch.float32).unsqueeze(1)

        # Backwards-compatible `y` attribute: keep previous single-target `y` for older code
        
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
