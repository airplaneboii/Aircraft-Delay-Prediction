import torch
import numpy as np
from torch_geometric.data import HeteroData
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from src.utils import normalize_with_idx, hhmm_to_minutes

class Hetero2Nodes:
    def __init__(self, df, args, train_index, val_index, test_index, norm_stats=None):
        self.df = df.reset_index(drop=True)
        self.args = args
        self.train_index = np.asarray(train_index, dtype=int)
        self.val_index   = np.asarray(val_index, dtype=int)
        self.test_index  = np.asarray(test_index, dtype=int)
        self.norm_stats = norm_stats  # optional externally-provided stats


    def _time_to_cyclic(self, minutes_arr: np.ndarray):
        # minutes in [0, 1439], represent time-of-day cyclically
        minutes_arr = minutes_arr.astype(np.float32)
        angle = 2.0 * np.pi * (minutes_arr / 1440.0)
        return np.stack([np.sin(angle), np.cos(angle)], axis=1).astype(np.float32)

    def build(self):
        data = HeteroData()
        train_df = self.df.iloc[self.train_index]
        self.df['FL_DATE'] = pd.to_datetime(self.df[['YEAR', 'MONTH', 'DAY_OF_MONTH']].rename(columns={'YEAR': 'year', 'MONTH': 'month', 'DAY_OF_MONTH': 'day'}))


        airports = sorted(set(self.df["ORIGIN_AIRPORT_ID"]).union(set(self.df["DEST_AIRPORT_ID"])))
        airport_map = {a: i for i, a in enumerate(airports)}
        num_airports = len(airports)

        ######### AIRPORT FEATURES ############

        # features based on origin airport 
        origin_features = train_df.groupby("ORIGIN_AIRPORT_ID").agg({
            "DEP_DELAY" : "mean",   # Average departure delay from this airport
            "TAXI_OUT" : "mean",    # Average taxi out time
            "DISTANCE" : "mean",    # Average distance of flights from this airport
            "ORIGIN_AIRPORT_ID" : "count"      # Number of departures
        }).rename(columns={"ORIGIN_AIRPORT_ID": "num_of_departures"})  #renaming for clarity

        # features based on destination airport
        dest_features = train_df.groupby("DEST_AIRPORT_ID").agg({
            "DISTANCE" : "mean",     # Average distance of flights to this airport
            "DEST_AIRPORT_ID" : "count"         # Number of arrivals
        }).rename(columns={"DEST_AIRPORT_ID": "num_of_arrivals"})  #renaming for clarity

        # one-hot encoding categorical features (WACs - World Area Codes)

        # get all unique WACs
        all_wacs = sorted(train_df["ORIGIN_WAC"].dropna().unique())
        
        # create one-hot encoder that will produce a one-hot vector for each WAC
        wac_encoder = OneHotEncoder(categories=[all_wacs], sparse_output=False, handle_unknown='ignore')
        wac_encoder.fit(np.array(all_wacs).reshape(-1, 1))  # fit encoder on all unique WACs
        airport_wac_map = (
            train_df.drop_duplicates("ORIGIN_AIRPORT_ID").set_index("ORIGIN_AIRPORT_ID")["ORIGIN_WAC"].to_dict()
        )

        #Build feature vectors for each airport
        airport_features = []
        airport_wac = []
        for airport in airports:
            
            # rows of aggregated stats for departures and arrivals from/to this airport
            dep_row = origin_features.loc[airport] if airport in origin_features.index else None
            arr_row = dest_features.loc[airport] if airport in dest_features.index else None
            
            avg_dep_delay = dep_row["DEP_DELAY"] if dep_row is not None else 0      # Average departure delay
            avg_taxi_out = dep_row["TAXI_OUT"] if dep_row is not None else 0        # Average taxi out time
            num_of_departures = dep_row["num_of_departures"] if dep_row is not None else 0  # Number of departures
            num_of_arrivals = arr_row["num_of_arrivals"] if arr_row is not None else 0      # Number of arrivals
            arrivals_avg_distance = arr_row["DISTANCE"] if arr_row is not None else 0      # Average distance of arrivals
            departures_avg_distance = dep_row["DISTANCE"] if dep_row is not None else 0  # Average distance of departures

            # One-hot encode WAC
            wac = airport_wac_map.get(airport, None)
            if wac is not None:
                wac_onehot = wac_encoder.transform([[wac]])[0]
            else:
                wac_onehot = np.zeros(len(all_wacs))
            
            # use log scaling for count features to make them less skewed
            num_of_departures = np.log1p(num_of_departures)
            num_of_arrivals   = np.log1p(num_of_arrivals)

            
            # Combine all features
            features = [
                avg_dep_delay,
                avg_taxi_out,
                num_of_departures,
                num_of_arrivals,
                arrivals_avg_distance,
                departures_avg_distance
            ]

            airport_features.append(features)
            airport_wac.append(wac_onehot)

        # Normalize airport features using train airports
        airport_arr = np.array(airport_features, dtype=np.float32)
        train_airports = sorted(set(train_df["ORIGIN_AIRPORT_ID"]).union(set(train_df["DEST_AIRPORT_ID"])))
        train_airport_idx = np.array([airport_map[a] for a in train_airports if a in airport_map], dtype=int)
        fit_idx = train_airport_idx if len(train_airport_idx) > 0 else np.arange(len(airport_arr))
        airport_arr, _, _ = normalize_with_idx(airport_arr, fit_idx)
        airport_wac = np.vstack(airport_wac).astype(np.float32)
        airport_arr = np.concatenate([airport_arr, airport_wac], axis=1)
        data["airport"].x = torch.tensor(airport_arr, dtype=torch.float)

        # ==============================================================================
        # 2. FLIGHT NODES 
        # ==============================================================================
        num_flights = len(self.df)
        data["flight"].num_nodes = num_flights
        self.df["DEP_TIME_MINUTES"] = self.df["CRS_DEP_TIME"].apply(hhmm_to_minutes)
        
        # Since each flight is a row in the dataframe, we can directly use the columns as features
        flight_features = self.df[[
            "CRS_DEP_TIME",
            "CRS_ARR_TIME",
            "DAY_OF_WEEK",
            "MONTH",
            "DISTANCE",
            "CRS_ELAPSED_TIME",
        ]].copy()

        # Convert CRS_DEP_TIME and CRS_ARR_TIME from HHMM to minutes since midnight
        flight_features["CRS_DEP_TIME"] = flight_features["CRS_DEP_TIME"].apply(hhmm_to_minutes)
        flight_features["CRS_ARR_TIME"] = flight_features["CRS_ARR_TIME"].apply(hhmm_to_minutes)

        # encode time features as cyclical features
        flight_features["dep_time_sin"] = np.sin(2 * np.pi * flight_features["CRS_DEP_TIME"] / (24 * 60))
        flight_features["dep_time_cos"] = np.cos(2 * np.pi * flight_features["CRS_DEP_TIME"] / (24 * 60))
        flight_features["arr_time_sin"] = np.sin(2 * np.pi * flight_features["CRS_ARR_TIME"] / (24 * 60))
        flight_features["arr_time_cos"] = np.cos(2 * np.pi * flight_features["CRS_ARR_TIME"] / (24 * 60))

        # encoding day of week
        flight_features['day_sin'] = np.sin(2 * np.pi * flight_features["DAY_OF_WEEK"] / 7)
        flight_features['day_cos'] = np.cos(2 * np.pi * flight_features["DAY_OF_WEEK"] / 7)

        # encoding month
        flight_features['month_sin'] = np.sin(2 * np.pi * (flight_features["MONTH"] - 1) / 12)
        flight_features['month_cos'] = np.cos(2 * np.pi * (flight_features["MONTH"] - 1) / 12)

        # drop columns that we don't need anymore
        flight_features = flight_features.drop(columns=[
            "CRS_DEP_TIME",
            "CRS_ARR_TIME",
            "DAY_OF_WEEK",
            "MONTH"
        ])
        # Normalize flight features using train flights
        flight_arr_raw = flight_features.values.astype(np.float32)
        fit_idx = np.asarray(self.train_index, dtype=int)
        flight_arr, _, _ = normalize_with_idx(flight_arr_raw, fit_idx)
        data["flight"].x = torch.tensor(flight_arr, dtype=torch.float)

        # ==============================================================================
        # 3. EDGES (Bipartite: Flight <-> Airport)
        # ==============================================================================
        origin = np.array([airport_map[o] for o in self.df["ORIGIN_AIRPORT_ID"]], dtype=np.int64)
        dest   = np.array([airport_map[d] for d in self.df["DEST_AIRPORT_ID"]], dtype=np.int64)
        flight_ids = np.arange(num_flights, dtype=np.int64)

        # Edge: Flight -> Origin Airport
        data["flight", "originates_from", "airport"].edge_index = torch.tensor(
            np.vstack([flight_ids, origin]), dtype=torch.long
        )
        # Reverse: Origin Airport -> Flight (Crucial for message passing to flight)
        data["airport", "has_departure", "flight"].edge_index = torch.tensor(
            np.vstack([origin, flight_ids]), dtype=torch.long
        )

        # Edge: Flight -> Dest Airport
        data["flight", "arrives_at", "airport"].edge_index = torch.tensor(
            np.vstack([flight_ids, dest]), dtype=torch.long
        )
        # Reverse: Dest Airport -> Flight
        data["airport", "has_arrival", "flight"].edge_index = torch.tensor(
            np.vstack([dest, flight_ids]), dtype=torch.long
        )

        # temporal edge (flight that later does another flight)
        # Edge 5: flight 1 performed by aircraft that later performs flight 2 (temporal link)
        next_src, next_dst = [], []

        temp_df = self.df[["TAIL_NUM", "FL_DATE", "DEP_TIME_MINUTES"]].copy()
        temp_df["node_idx"] = np.arange(len(temp_df), dtype=np.int64)

        df_sorted = temp_df.sort_values(["TAIL_NUM", "FL_DATE", "DEP_TIME_MINUTES"])

        for _, group in df_sorted.groupby("TAIL_NUM", sort=False):
            ids = group["node_idx"].to_numpy()
            if len(ids) >= 2:
                next_src.extend(ids[:-1].tolist())
                next_dst.extend(ids[1:].tolist())

        data["flight", "next_same_aircraft", "flight"].edge_index = torch.tensor(
            [next_src, next_dst], dtype=torch.long
        )
        data["flight", "prev_same_aircraft", "flight"].edge_index = torch.tensor(
            [next_dst, next_src], dtype=torch.long
        )

        # ======================================================================
        # 4. LABELS & MASKS (On Flight Nodes)
        # ======================================================================
        arr_delay_vals = self.df["ARR_DELAY"].fillna(0.0).astype(float).to_numpy()
        data["flight"].y_reg = torch.tensor(arr_delay_vals, dtype=torch.float32).unsqueeze(1)

        if "ARR_DEL15" in self.df.columns:
            cls_vals = self.df["ARR_DEL15"].astype(int).to_numpy()
        else:
            border = getattr(self.args, "border", 0.45)
            cls_vals = (self.df["ARR_DELAY"].fillna(0.0) >= border * 60).astype(int).to_numpy()
        data["flight"].y_cls = torch.tensor(cls_vals, dtype=torch.float32).unsqueeze(1)


        train_mask = torch.zeros(num_flights, dtype=torch.bool)
        val_mask   = torch.zeros(num_flights, dtype=torch.bool)
        test_mask  = torch.zeros(num_flights, dtype=torch.bool)

        train_mask[self.train_index] = True
        val_mask[self.val_index]     = True
        test_mask[self.test_index]   = True

        data["flight"].train_mask = train_mask
        data["flight"].val_mask   = val_mask
        data["flight"].test_mask  = test_mask

        # Metadata for saving
        data.norm_stats = self.norm_stats
        data.train_index = self.train_index
        data.val_index = self.val_index
        data.test_index = self.test_index
        return data