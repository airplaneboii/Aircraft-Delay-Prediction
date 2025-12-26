import torch
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData
from sklearn.preprocessing import OneHotEncoder
from src.utils import normalize_with_idx, hhmm_to_minutes

class HomoGraph:
    def __init__(self, df, args, train_index, val_index, test_index, norm_stats=None):
        self.df = df.reset_index(drop=True)
        self.args = args
        self.train_index = np.asarray(train_index, dtype=int)
        self.val_index   = np.asarray(val_index, dtype=int)
        self.test_index  = np.asarray(test_index, dtype=int)
        self.norm_stats = norm_stats  # optional externally-provided stats
        self.classification = args.prediction_type == "classification"

    def _time_to_cyclic(self, minutes_arr: np.ndarray):
        # minutes in [0, 1439], represent time-of-day cyclically
        minutes_arr = minutes_arr.astype(np.float32)
        angle = 2.0 * np.pi * (minutes_arr / 1440.0)
        return np.stack([np.sin(angle), np.cos(angle)], axis=1).astype(np.float32)

    def build(self):
        data = HeteroData()

        # airport nodes
        airports = sorted(set(self.df["ORIGIN_AIRPORT_ID"]).union(set(self.df["DEST_AIRPORT_ID"])))
        airport_map = {a: i for i, a in enumerate(airports)}
        data["airport"].num_nodes = len(airports)

        # Ensure a date column exists if you need it elsewhere (not used in features below)
        if {"YEAR", "MONTH", "DAY_OF_MONTH"}.issubset(self.df.columns):
            self.df["FL_DATE"] = pd.to_datetime(
                self.df[["YEAR", "MONTH", "DAY_OF_MONTH"]].rename(
                    columns={"YEAR": "year", "MONTH": "month", "DAY_OF_MONTH": "day"}
                ),
                errors="coerce",
            )

        train_df = self.df.iloc[self.train_index].copy()

        # Aggregated airport stats computed ONLY from training flights
        origin_features = train_df.groupby("ORIGIN_AIRPORT_ID").agg(
            DEP_DELAY=("DEP_DELAY", "mean"),
            TAXI_OUT=("TAXI_OUT", "mean"),
            ORIG_DIST=("DISTANCE", "mean"),
            num_of_departures=("ORIGIN_AIRPORT_ID", "count"),
        )

        dest_features = train_df.groupby("DEST_AIRPORT_ID").agg(
            DEST_DIST=("DISTANCE", "mean"),
            num_of_arrivals=("DEST_AIRPORT_ID", "count"),
        )

        # One-hot encoding for WAC (fit on training WACs)
        all_wacs = sorted(train_df["ORIGIN_WAC"].dropna().unique().tolist())
        wac_encoder = OneHotEncoder(categories=[all_wacs], sparse_output=False, handle_unknown="ignore")
        if len(all_wacs) > 0:
            wac_encoder.fit(np.array(all_wacs).reshape(-1, 1))

        # Map airport -> WAC from training (prefer origin WAC; fallback to dest WAC if you have it)
        airport_wac_map = (
            train_df.drop_duplicates("ORIGIN_AIRPORT_ID")
                    .set_index("ORIGIN_AIRPORT_ID")["ORIGIN_WAC"]
                    .to_dict()
        )

        # Build raw airport feature matrix (un-normalized)
        raw_feats = []
        raw_wac = []

        for a in airports:
            dep = origin_features.loc[a] if a in origin_features.index else None
            arr = dest_features.loc[a] if a in dest_features.index else None

            avg_dep_delay = float(dep["DEP_DELAY"]) if dep is not None and pd.notna(dep["DEP_DELAY"]) else 0.0
            avg_taxi_out  = float(dep["TAXI_OUT"])  if dep is not None and pd.notna(dep["TAXI_OUT"])  else 0.0

            num_dep = float(dep["num_of_departures"]) if dep is not None else 0.0
            num_arr = float(arr["num_of_arrivals"])   if arr is not None else 0.0

            dep_dist = float(dep["ORIG_DIST"]) if dep is not None and pd.notna(dep["ORIG_DIST"]) else 0.0
            arr_dist = float(arr["DEST_DIST"]) if arr is not None and pd.notna(arr["DEST_DIST"]) else 0.0

            # log scaling for count features
            num_dep = np.log1p(num_dep)
            num_arr = np.log1p(num_arr)

            raw_feats.append([avg_dep_delay, avg_taxi_out, num_dep, num_arr, arr_dist, dep_dist])

            wac = airport_wac_map.get(a, None)
            if wac is not None and len(all_wacs) > 0:
                w = wac_encoder.transform([[wac]])[0].astype(np.float32)
            else:
                w = np.zeros((len(all_wacs),), dtype=np.float32)
            raw_wac.append(w)

        raw_feats = np.asarray(raw_feats, dtype=np.float32)
        raw_wac = np.vstack(raw_wac).astype(np.float32) if len(all_wacs) > 0 else np.zeros((len(airports), 0), dtype=np.float32)

        # Fit/apply normalization using ONLY training airports (airports appearing in train_df)
        train_airports = sorted(set(train_df["ORIGIN_AIRPORT_ID"]).union(set(train_df["DEST_AIRPORT_ID"])))
        train_airport_idx = np.array([airport_map[a] for a in train_airports if a in airport_map], dtype=int)

        fit_idx = train_airport_idx if len(train_airport_idx) > 0 else np.arange(len(airports))
        if self.norm_stats is None:
            norm_feats, mu_airport, std_airport = normalize_with_idx(raw_feats, fit_idx)
            self.norm_stats = {"mean": mu_airport, "std": std_airport}
        else:
            norm_feats = (raw_feats - self.norm_stats["mean"]) / self.norm_stats["std"]

        # Final airport node features = normalized numeric + one-hot WAC
        airport_x = np.concatenate([norm_feats, raw_wac], axis=1).astype(np.float32)
        data["airport"].x = torch.tensor(airport_x, dtype=torch.float)

        # ---------- Edges: flights (airport -> airport), one per row ----------
        num_flights = len(self.df)
        origin = np.array([airport_map[o] for o in self.df["ORIGIN_AIRPORT_ID"]], dtype=np.int64)
        dest   = np.array([airport_map[d] for d in self.df["DEST_AIRPORT_ID"]], dtype=np.int64)

        edge_index = torch.tensor(np.vstack([origin, dest]), dtype=torch.long)

        edge_store = data["airport", "flight_to", "airport"]
        edge_store.edge_index = edge_index

        # Edge attributes
        ef = self.df[["CRS_DEP_TIME", "CRS_ARR_TIME", "DAY_OF_WEEK"]].copy()
        dep_min = ef["CRS_DEP_TIME"].apply(hhmm_to_minutes).to_numpy(dtype=np.float32)
        arr_min = ef["CRS_ARR_TIME"].apply(hhmm_to_minutes).to_numpy(dtype=np.float32)
        dow     = ef["DAY_OF_WEEK"].fillna(0).to_numpy(dtype=np.float32).reshape(-1, 1)

        dep_cyc = self._time_to_cyclic(dep_min)  # (N,2)
        arr_cyc = self._time_to_cyclic(arr_min)  # (N,2)

        edge_attr = np.concatenate([dep_cyc, arr_cyc, dow], axis=1).astype(np.float32)
        edge_store.edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # ---------- Edge labels and split masks (edge-level regression/classification) ----------
        arr_delay_vals = self.df["ARR_DELAY"].fillna(0.0).to_numpy(dtype=np.float32)
        edge_store.edge_label_reg = torch.tensor(arr_delay_vals).unsqueeze(1)

        if "ARR_DEL15" in self.df.columns:
            cls_vals = self.df["ARR_DEL15"].fillna(0).astype(int).to_numpy()
        else:
            border = getattr(self.args, "border", 0.45)
            cls_vals = (self.df["ARR_DELAY"].fillna(0.0) >= border * 60).astype(int).to_numpy()
        edge_store.edge_label_cls = torch.tensor(cls_vals, dtype=torch.float32).unsqueeze(1)


        train_mask = torch.zeros(num_flights, dtype=torch.bool)
        val_mask   = torch.zeros(num_flights, dtype=torch.bool)
        test_mask  = torch.zeros(num_flights, dtype=torch.bool)

        train_mask[self.train_index] = True
        val_mask[self.val_index]     = True
        test_mask[self.test_index]   = True

        edge_store.train_mask = train_mask
        edge_store.val_mask   = val_mask
        edge_store.test_mask  = test_mask

        # Make graph self-contained when saved
        data.norm_stats = self.norm_stats
        data.train_index = self.train_index
        data.val_index = self.val_index
        data.test_index = self.test_index

        return data

