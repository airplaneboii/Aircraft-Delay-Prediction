import torch
import numpy as np
from torch_geometric.data import HeteroData
import pandas as pd
import logging
import time

class NotVeryHetero:
    """
    NotVeryHetero: Heterogeneous graph with a single FLIGHT node type.
    
    Core idea: Remove airport/aircraft/airline nodes entirely.
    Compensate with:
    1. Enriched flight features that include airport/aircraft/airline statistics
    2. Multiple flight-to-flight edge types capturing different temporal relationships
    3. Edge attributes encoding temporal gaps and relationship types
    
    Edge types:
    - next_aircraft: consecutive flights by same aircraft (reliability signal)
    - next_origin: consecutive departures from same origin (congestion signal)
    - next_route: consecutive flights on same route (weather/ATC signal)
    - temporal_window: flights departing within same hour (system-wide signal)
    """

    def normalize_features(self, x):
        x = np.asarray(x, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True) + 1e-6
        return (x - mean) / std

    def __init__(self, df, args, train_index, val_index, test_index, norm_stats=None):
        self.df = df
        self.args = args
        self.train_index = train_index
        self.val_index = val_index
        self.test_index = test_index
        self.norm_stats = norm_stats
        self.classification = args.prediction_type == "classification"

    def hhmm_to_minutes(self, hhmm):
        if pd.isna(hhmm):
            return 0
        hhmm = int(hhmm)
        hours = hhmm // 100
        minutes = hhmm % 100
        return hours * 60 + minutes

    def build(self):
        data = HeteroData()
        logger = logging.getLogger(__name__)
        start_total = time.time()

        logger.debug("[NotVeryHetero] Starting graph build...")
        start = time.time()
        
        self.df['FL_DATE'] = pd.to_datetime(
            self.df[['YEAR', 'MONTH', 'DAY_OF_MONTH']].rename(
                columns={'YEAR': 'year', 'MONTH': 'month', 'DAY_OF_MONTH': 'day'}
            )
        )
        self.df["DEP_TIME_MINUTES"] = self.df["CRS_DEP_TIME"].apply(self.hhmm_to_minutes)
        logger.debug(f"[NotVeryHetero] Date/time preprocessing: {time.time() - start:.2f}s")

        # ===== EFFICIENT STATISTICS PRECOMPUTATION =====
        start = time.time()
        logger.debug("[NotVeryHetero] Computing origin airport statistics...")
        origin_agg = self.df.groupby("ORIGIN_AIRPORT_ID").agg({
            "DEP_DELAY": ["mean", "std", "median"],
            "TAXI_OUT": "mean",
        }).fillna(0)
        origin_agg.columns = ['_'.join(col).strip() for col in origin_agg.columns.values]
        
        origin_counts = self.df.groupby("ORIGIN_AIRPORT_ID").size()
        logger.debug(f"[NotVeryHetero] Origin airport stats: {time.time() - start:.2f}s ({len(origin_counts)} airports)")

        start = time.time()
        logger.debug("[NotVeryHetero] Computing destination airport statistics...")
        dest_agg = self.df.groupby("DEST_AIRPORT_ID").agg({
            "ARR_DELAY": ["mean", "std", "median"],
        }).fillna(0)
        dest_agg.columns = ['_'.join(col).strip() for col in dest_agg.columns.values]
        dest_counts = self.df.groupby("DEST_AIRPORT_ID").size()
        logger.debug(f"[NotVeryHetero] Destination airport stats: {time.time() - start:.2f}s")

        start = time.time()
        logger.debug("[NotVeryHetero] Computing aircraft statistics...")
        aircraft_agg = self.df.groupby("TAIL_NUM").agg({
            "DEP_DELAY": ["mean", "std"],
        }).fillna(0)
        aircraft_agg.columns = ['_'.join(col).strip() for col in aircraft_agg.columns.values]
        logger.debug(f"[NotVeryHetero] Aircraft stats: {time.time() - start:.2f}s ({len(aircraft_agg)} aircraft)")

        start = time.time()
        logger.debug("[NotVeryHetero] Computing airline statistics...")
        airline_agg = self.df.groupby("OP_CARRIER_AIRLINE_ID").agg({
            "DEP_DELAY": ["mean", "std"],
            "DIVERTED": "mean",
        }).fillna(0)
        airline_agg.columns = ['_'.join(col).strip() for col in airline_agg.columns.values]
        logger.debug(f"[NotVeryHetero] Airline stats: {time.time() - start:.2f}s ({len(airline_agg)} airlines)")

        # ===== VECTORIZED FEATURE ENGINEERING =====
        start = time.time()
        logger.debug("[NotVeryHetero] Extracting base flight features...")
        flight_features = self.df[[
            "CRS_DEP_TIME", "CRS_ARR_TIME", "DAY_OF_WEEK", "MONTH", "DISTANCE", "CRS_ELAPSED_TIME",
            "ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "TAIL_NUM", "OP_CARRIER_AIRLINE_ID"
        ]].copy()

        # Convert times to minutes (vectorized)
        dep_minutes = flight_features["CRS_DEP_TIME"].values.astype(int)
        dep_hours = dep_minutes // 100
        dep_mins = dep_minutes % 100
        dep_minutes = dep_hours * 60 + dep_mins

        arr_minutes = flight_features["CRS_ARR_TIME"].values.astype(int)
        arr_hours = arr_minutes // 100
        arr_mins = arr_minutes % 100
        arr_minutes = arr_hours * 60 + arr_mins
        logger.debug(f"[NotVeryHetero] Time conversion: {time.time() - start:.2f}s")

        # Cyclical encoding (vectorized)
        start = time.time()
        logger.debug("[NotVeryHetero] Computing cyclical encodings...")
        features_dict = {
            "distance": flight_features["DISTANCE"].values,
            "elapsed_time": flight_features["CRS_ELAPSED_TIME"].values,
            "dep_time_sin": np.sin(2 * np.pi * dep_minutes / (24 * 60)),
            "dep_time_cos": np.cos(2 * np.pi * dep_minutes / (24 * 60)),
            "arr_time_sin": np.sin(2 * np.pi * arr_minutes / (24 * 60)),
            "arr_time_cos": np.cos(2 * np.pi * arr_minutes / (24 * 60)),
            "day_sin": np.sin(2 * np.pi * flight_features["DAY_OF_WEEK"].values / 7),
            "day_cos": np.cos(2 * np.pi * flight_features["DAY_OF_WEEK"].values / 7),
            "month_sin": np.sin(2 * np.pi * (flight_features["MONTH"].values - 1) / 12),
            "month_cos": np.cos(2 * np.pi * (flight_features["MONTH"].values - 1) / 12),
        }
        logger.debug(f"[NotVeryHetero] Cyclical encodings: {time.time() - start:.2f}s")

        # Add aggregated statistics (single vectorized lookup instead of multiple maps)
        start = time.time()
        logger.debug("[NotVeryHetero] Adding aggregated airport/aircraft/airline statistics to features...")
        origin_ids = flight_features["ORIGIN_AIRPORT_ID"].values
        dest_ids = flight_features["DEST_AIRPORT_ID"].values
        tail_nums = flight_features["TAIL_NUM"].values
        airline_ids = flight_features["OP_CARRIER_AIRLINE_ID"].values

        features_dict["origin_dep_delay_mean"] = origin_agg.loc[origin_ids, "DEP_DELAY_mean"].values
        features_dict["origin_dep_delay_std"] = origin_agg.loc[origin_ids, "DEP_DELAY_std"].values
        features_dict["origin_dep_delay_median"] = origin_agg.loc[origin_ids, "DEP_DELAY_median"].values
        features_dict["origin_taxi_out"] = origin_agg.loc[origin_ids, "TAXI_OUT_mean"].values
        features_dict["origin_num_deps"] = np.log1p(origin_counts.loc[origin_ids].values)

        features_dict["dest_arr_delay_mean"] = dest_agg.loc[dest_ids, "ARR_DELAY_mean"].values
        features_dict["dest_arr_delay_std"] = dest_agg.loc[dest_ids, "ARR_DELAY_std"].values
        features_dict["dest_num_arrs"] = np.log1p(dest_counts.loc[dest_ids].values)

        features_dict["aircraft_dep_delay_mean"] = aircraft_agg.loc[tail_nums, "DEP_DELAY_mean"].values
        features_dict["aircraft_dep_delay_std"] = aircraft_agg.loc[tail_nums, "DEP_DELAY_std"].values
        features_dict["aircraft_num_flights"] = np.log1p(self.df.groupby("TAIL_NUM").size().loc[tail_nums].values)

        features_dict["airline_dep_delay_mean"] = airline_agg.loc[airline_ids, "DEP_DELAY_mean"].values
        features_dict["airline_dep_delay_std"] = airline_agg.loc[airline_ids, "DEP_DELAY_std"].values
        features_dict["airline_diverted_rate"] = airline_agg.loc[airline_ids, "DIVERTED_mean"].values
        logger.debug(f"[NotVeryHetero] Aggregated statistics lookup: {time.time() - start:.2f}s")

        # Stack all features
        start = time.time()
        logger.debug("[NotVeryHetero] Stacking and normalizing features...")
        feature_matrix = np.column_stack([features_dict[k] for k in features_dict.keys()])
        feature_matrix = self.normalize_features(feature_matrix)
        flight_arr = feature_matrix
        logger.debug(f"[NotVeryHetero] Feature normalization and stacking: {time.time() - start:.2f}s (shape: {flight_arr.shape})")

        data["flight"].x = torch.tensor(flight_arr, dtype=torch.float)

        # ===== BUILD FLIGHT-TO-FLIGHT EDGES (OPTIMIZED) =====
        start = time.time()
        logger.debug("[NotVeryHetero] Sorting dataframe for edge creation...")
        num_flights = len(self.df)
        
        # Pre-sort dataframe once for temporal edges
        df_sorted = self.df.sort_values(["FL_DATE", "DEP_TIME_MINUTES"]).reset_index(drop=True)
        logger.debug(f"[NotVeryHetero] Dataframe sorted: {time.time() - start:.2f}s")
        
        all_edges = []

        # Edge type 1: next_aircraft (consecutive by tail number)
        start = time.time()
        logger.debug("[NotVeryHetero] Creating next_aircraft edges...")
        # Create aircraft index on sorted data
        aircraft_groups = df_sorted.groupby("TAIL_NUM", sort=False).apply(lambda g: g.index.tolist())
        aircraft_src, aircraft_dst = [], []
        for idx_list in aircraft_groups:
            for i in range(len(idx_list) - 1):
                aircraft_src.append(idx_list[i])
                aircraft_dst.append(idx_list[i + 1])
        
        if aircraft_src:
            data["flight", "next_aircraft", "flight"].edge_index = torch.tensor([aircraft_src, aircraft_dst], dtype=torch.long)
            all_edges.extend([(s, d, 0) for s, d in zip(aircraft_src, aircraft_dst)])
        logger.debug(f"[NotVeryHetero] next_aircraft edges: {time.time() - start:.2f}s ({len(aircraft_src)} edges)")

        # Edge type 2: next_origin (consecutive departures from same origin)
        start = time.time()
        logger.debug("[NotVeryHetero] Creating next_origin edges...")
        origin_groups = df_sorted.groupby("ORIGIN_AIRPORT_ID", sort=False).apply(lambda g: g.index.tolist())
        origin_src, origin_dst = [], []
        for idx_list in origin_groups:
            for i in range(len(idx_list) - 1):
                origin_src.append(idx_list[i])
                origin_dst.append(idx_list[i + 1])
        
        if origin_src:
            data["flight", "next_origin", "flight"].edge_index = torch.tensor([origin_src, origin_dst], dtype=torch.long)
            all_edges.extend([(s, d, 1) for s, d in zip(origin_src, origin_dst)])
        logger.debug(f"[NotVeryHetero] next_origin edges: {time.time() - start:.2f}s ({len(origin_src)} edges)")

        # Edge type 3: next_route (same origin-destination pair, within 720 minutes/12 hours)
        start = time.time()
        logger.debug("[NotVeryHetero] Creating next_route edges...")
        # Create route on sorted data
        route_col = df_sorted["ORIGIN_AIRPORT_ID"].astype(str) + "_" + df_sorted["DEST_AIRPORT_ID"].astype(str)
        route_groups = df_sorted.groupby(route_col, sort=False).apply(lambda g: g.index.tolist())
        
        route_src, route_dst = [], []
        for idx_list in route_groups:
            for i in range(len(idx_list) - 1):
                time_diff = (df_sorted.loc[idx_list[i+1], "DEP_TIME_MINUTES"] - 
                           df_sorted.loc[idx_list[i], "DEP_TIME_MINUTES"]) % (24 * 60)
                if 0 < time_diff <= 720:  # Within 12 hours
                    route_src.append(idx_list[i])
                    route_dst.append(idx_list[i + 1])
        
        if route_src:
            data["flight", "next_route", "flight"].edge_index = torch.tensor([route_src, route_dst], dtype=torch.long)
            all_edges.extend([(s, d, 2) for s, d in zip(route_src, route_dst)])
        logger.debug(f"[NotVeryHetero] next_route edges: {time.time() - start:.2f}s ({len(route_src)} edges)")

        # Edge type 4: temporal_window (flights departing within same hour)
        start = time.time()
        logger.debug("[NotVeryHetero] Creating temporal_window edges...")
        temporal_src, temporal_dst = [], []
        date_groups = df_sorted.groupby("FL_DATE", sort=False).apply(lambda g: g.index.tolist())
        
        for idx_list in date_groups:
            for i in range(len(idx_list) - 1):
                time_diff = (df_sorted.loc[idx_list[i+1], "DEP_TIME_MINUTES"] - 
                           df_sorted.loc[idx_list[i], "DEP_TIME_MINUTES"])
                if 0 < time_diff < 60:  # Within 1 hour
                    temporal_src.append(idx_list[i])
                    temporal_dst.append(idx_list[i + 1])
        
        if temporal_src:
            data["flight", "next_temporal", "flight"].edge_index = torch.tensor([temporal_src, temporal_dst], dtype=torch.long)
            all_edges.extend([(s, d, 3) for s, d in zip(temporal_src, temporal_dst)])
        logger.debug(f"[NotVeryHetero] temporal_window edges: {time.time() - start:.2f}s ({len(temporal_src)} edges)")

        # Note: HeteroData stores typed edges natively; no unified index needed
        logger.debug(f"[NotVeryHetero] Edge types stored: {len(data.edge_types)} relation types")

        # ===== MASKS AND LABELS =====
        start = time.time()
        logger.debug("[NotVeryHetero] Creating masks and labels...")
        train_mask = torch.zeros(num_flights, dtype=torch.bool)
        val_mask = torch.zeros(num_flights, dtype=torch.bool)
        test_mask = torch.zeros(num_flights, dtype=torch.bool)

        train_mask[self.train_index] = True
        val_mask[self.val_index] = True
        test_mask[self.test_index] = True

        data["flight"].train_mask = train_mask
        data["flight"].val_mask = val_mask
        data["flight"].test_mask = test_mask

        # Keep negative delays as they're informative
        if self.classification:
            data["flight"].y = torch.tensor(self.df["ARR_DEL15"].fillna(0).values, dtype=torch.float).unsqueeze(1)
        else:
            data["flight"].y = torch.tensor(self.df["ARR_DELAY"].fillna(0).values, dtype=torch.float).unsqueeze(1)
        logger.debug(f"[NotVeryHetero] Masks and labels: {time.time() - start:.2f}s")

        try:
            data.norm_stats = self.norm_stats
            data.train_index = self.train_index
            data.val_index = self.val_index
            data.test_index = self.test_index
        except Exception:
            pass

        total_time = time.time() - start_total
        logger.debug(f"[NotVeryHetero] TOTAL GRAPH BUILD TIME: {total_time:.2f}s")
        logger.debug(f"[NotVeryHetero] Graph stats: {num_flights} flights, {len(all_edges) if all_edges else 0} edges, {flight_arr.shape[1]} features")

        return data
