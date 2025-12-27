import torch
import numpy as np
from torch_geometric.data import HeteroData
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from src.utils import normalize_with_idx, hhmm_to_minutes

class HeteroGraph2:

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

        self.df['FL_DATE'] = pd.to_datetime(self.df[['YEAR', 'MONTH', 'DAY_OF_MONTH']].rename(columns={'YEAR': 'year', 'MONTH': 'month', 'DAY_OF_MONTH': 'day'}))


        ######### AIRPORT FEATURES ############


        # features based on origin airport (fit on train only)
        origin_features = train_df.groupby("ORIGIN_AIRPORT_ID").agg({
            "DEP_DELAY" : "mean",   # Average departure delay from this airport
            "TAXI_OUT" : "mean",    # Average taxi out time
            "DISTANCE" : "mean",    # Average distance of flights from this airport
            "ORIGIN_AIRPORT_ID" : "count"      # Number of departures
        }).rename(columns={"ORIGIN_AIRPORT_ID": "num_of_departures"})  #renaming for clarity

        # features based on destination airport (fit on train only)
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




        ####### AIRCRAFT FEATURES #############


        aircraft_grp = train_df.groupby("TAIL_NUM").agg({
            "DEP_DELAY": "mean",              # Average departure delay for this aircraft
            "DISTANCE": "mean",               # Average flight distance for this aircraft
            "CRS_ELAPSED_TIME": "mean",    # Average block time (makes / model differences)
            "TAIL_NUM": "count"               # Number of flights for this aircraft
        }).rename(columns={"TAIL_NUM": "num_of_flights"}) #renaming for clarity
        
        # Build feature vectors for each aircraft
        aircraft_features = []

        for tail in aircrafts:

            # Extract aggregated statistics for this aircraft
            row = aircraft_grp.loc[tail] if tail in aircraft_grp.index else None

            avg_dep_delay = row["DEP_DELAY"] if row is not None else 0
            avg_distance = row["DISTANCE"] if row is not None else 0
            crs_elapsed_time = row["CRS_ELAPSED_TIME"] if row is not None else 0
            num_of_flights = row["num_of_flights"] if row is not None else 0

            # Combine into feature vector
            features = [
                avg_dep_delay,     
                avg_distance,      
                crs_elapsed_time, 
                num_of_flights        
            ]

            aircraft_features.append(features)
        # Normalize aircraft features using train aircraft
        aircraft_arr = np.array(aircraft_features, dtype=np.float32)
        train_aircraft = sorted(train_df["TAIL_NUM"].unique())
        train_aircraft_idx = np.array([aircraft_map[a] for a in train_aircraft if a in aircraft_map], dtype=int)
        fit_idx = train_aircraft_idx if len(train_aircraft_idx) > 0 else np.arange(len(aircraft_arr))
        aircraft_arr, _, _ = normalize_with_idx(aircraft_arr, fit_idx)

        
        ######## AIRLINE FEATURES #############


        airline_grp = train_df.groupby("OP_CARRIER_AIRLINE_ID").agg({
            "DEP_DELAY": "mean",          # Avg departure delay for this airline
            "DIVERTED": "mean",           # Diversion rate
            "DISTANCE": "mean",           # Avg distance flown
            "TAXI_OUT": "mean",           # Avg taxi-out time (congestion proxy)
            "OP_CARRIER_AIRLINE_ID": "count"  # Number of flights for this airline
        }).rename(columns={"OP_CARRIER_AIRLINE_ID": "num_flights"})
        # Now: num_flights = total flights this airline operated

        # build feature vectors for each airline
        airline_features = []

        for carrier in airlines:

            # Extract airline stats from precomputed groupby
            row = airline_grp.loc[carrier] if carrier in airline_grp.index else None

            avg_dep_delay = row["DEP_DELAY"] if row is not None else 0
            diverted_rate = row["DIVERTED"] if row is not None else 0
            avg_distance = row["DISTANCE"] if row is not None else 0
            avg_taxi_out = row["TAXI_OUT"] if row is not None else 0
            num_flights = row["num_flights"] if row is not None else 0

            # Combine all features into one vector
            features = [
                avg_dep_delay,
                diverted_rate,
                avg_distance,
                avg_taxi_out,
                num_flights
            ]

            airline_features.append(features)
        # Normalize airline features using train airlines
        airline_arr = np.array(airline_features, dtype=np.float32)
        train_airlines = sorted(train_df["OP_CARRIER_AIRLINE_ID"].unique())
        train_airline_idx = np.array([airline_map[a] for a in train_airlines if a in airline_map], dtype=int)
        fit_idx = train_airline_idx if len(train_airline_idx) > 0 else np.arange(len(airline_arr))
        airline_arr, _, _ = normalize_with_idx(airline_arr, fit_idx)


        ###### FLIGHT FEATURES ##########
        
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

        # Assign node features
        data["airport"].x = torch.tensor(airport_arr, dtype=torch.float)
        data["aircraft"].x = torch.tensor(aircraft_arr, dtype=torch.float)
        data["airline"].x = torch.tensor(airline_arr, dtype=torch.float)

        num_flights = len(self.df) #Every line in data is a different flight
        data["flight"].x = torch.tensor(flight_arr, dtype=torch.float)
        
        # Add timestamps for windowing support
        # Create departure timestamps
        self.df["dep_minutes_calc"] = self.df["CRS_DEP_TIME"].apply(hhmm_to_minutes)
        self.df["dep_timestamp"] = self.df["FL_DATE"] + pd.to_timedelta(self.df["dep_minutes_calc"], unit="m")
        
        # Store absolute minutes since first timestamp for windowing
        min_timestamp = self.df["dep_timestamp"].min()
        max_timestamp = self.df["dep_timestamp"].max()
        timestamp_range = (max_timestamp - min_timestamp).total_seconds()
        
        if timestamp_range > 0:
            self.df["dep_timestamp_norm"] = ((self.df["dep_timestamp"] - min_timestamp).dt.total_seconds() / timestamp_range).astype(np.float32)
        else:
            self.df["dep_timestamp_norm"] = 0.0
        
        self.df["dep_timestamp_minutes"] = ((self.df["dep_timestamp"] - min_timestamp).dt.total_seconds() / 60.0).astype(np.float32)
        
        data["flight"].timestamp = torch.tensor(self.df["dep_timestamp_norm"].values, dtype=torch.float32)
        data["flight"].timestamp_min = torch.tensor(self.df["dep_timestamp_minutes"].values, dtype=torch.float32)

        # Edges

        flight_index = list(range(num_flights))
        origin = [airport_map[o] for o in self.df["ORIGIN_AIRPORT_ID"]]
        dest = [airport_map[d] for d in self.df["DEST_AIRPORT_ID"]]
        airline = [airline_map[a] for a in self.df["OP_CARRIER_AIRLINE_ID"]]
        aircraft = [aircraft_map[t] for t in self.df["TAIL_NUM"]]

        # Edge 1: flight originates from airport
        data["flight", "originates_from", "airport"].edge_index = torch.tensor([flight_index, origin], dtype=torch.long)
        # Reverse edge: airport has departing flights
        data["airport", "has_departure", "flight"].edge_index = torch.tensor([origin, flight_index], dtype=torch.long)

        # Edge 2: flight arrives at airport
        data["flight", "arrives_at", "airport"].edge_index = torch.tensor([flight_index, dest], dtype=torch.long)
        # Reverse edge: airport has arriving flights
        data["airport", "has_arrival", "flight"].edge_index = torch.tensor([dest, flight_index], dtype=torch.long)

        # Edge 3: flight operated by airline
        data["flight", "operated_by", "airline"].edge_index = torch.tensor([flight_index, airline], dtype=torch.long)
        # Reverse edge: airline operates flights
        data["airline", "operates", "flight"].edge_index = torch.tensor([airline, flight_index], dtype=torch.long)

        # Edge 4: flight performed by aircraft
        data["flight", "performed_by", "aircraft"].edge_index = torch.tensor([flight_index, aircraft], dtype=torch.long)
        # Reverse edge: aircraft performs flights
        data["aircraft", "performs", "flight"].edge_index = torch.tensor([aircraft, flight_index], dtype=torch.long)

        # Edge 5: flight 1 performed by aircraft that later performs flight 2 (temporal link)
        next_src = []
        next_dst = []
        self.df["DEP_TIME_MINUTES"] = self.df["CRS_DEP_TIME"].apply(hhmm_to_minutes)
        df_sorted = self.df.sort_values(["TAIL_NUM", "FL_DATE", "DEP_TIME_MINUTES"]).reset_index(drop=True)
        for tail, group in df_sorted.groupby("TAIL_NUM"):
            idx_list = group.index.tolist()
            # link consecutive flights
            for i in range(len(idx_list) - 1):
                next_src.append(idx_list[i])
                next_dst.append(idx_list[i + 1])

        #data["flight", "next_same_aircraft", "flight"].edge_index = torch.tensor([next_src, next_dst], dtype=torch.long)

        # Edge 6: flight delayed because of cause
        # removed so that no label leakage occurs, as causes are only known after delay happens

        # Edge 7: flight cancelled because of cause
        # currently removed cancelled flights from dataset, might change later to
        # support cancellation prediction as well

        # dataset split masks
        train_mask = torch.zeros(num_flights, dtype=torch.bool)
        val_mask = torch.zeros(num_flights, dtype=torch.bool)
        test_mask = torch.zeros(num_flights, dtype=torch.bool)

        train_mask[self.train_index] = True
        val_mask[self.val_index] = True
        test_mask[self.test_index] = True

        data["flight"].train_mask = train_mask
        data["flight"].val_mask = val_mask
        data["flight"].test_mask = test_mask

        # Labels: store both regression and classification targets for backward compatibility
        arr_delay_vals = self.df["ARR_DELAY"].fillna(0.0).astype(float).values
        data["flight"].y_reg = torch.tensor(arr_delay_vals, dtype=torch.float32).unsqueeze(1)

        if "ARR_DEL15" in self.df.columns:
            cls_vals = self.df["ARR_DEL15"].astype(int).values
        else:
            border = getattr(self.args, "border", 0.45)
            cls_vals = (self.df["ARR_DELAY"].fillna(0.0) >= border * 60).astype(int).values
        data["flight"].y_cls = torch.tensor(cls_vals, dtype=torch.float32).unsqueeze(1)


        # Attach norm_stats and split indexes so the graph is self-contained when saved
        try:
            data.norm_stats = self.norm_stats
        except Exception:
            pass
        try:
            data.train_index = self.train_index
            data.val_index = self.val_index
            data.test_index = self.test_index
        except Exception:
            pass

        return data
