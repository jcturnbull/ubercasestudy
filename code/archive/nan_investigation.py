# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 17:42:32 2025

@author: epicx
"""

import pandas as pd

# 1. Load aggregated hourly (local NY, tz-naive index)
lga_df = pd.read_parquet(
    r"C:\Users\epicx\Projects\ubercasestudy\data\processed\weather_lga_hourly_agg_2025.parquet"
)

# 2. Identify NaN hours excluding wind
wind_cols = [c for c in lga_df.columns if "wind" in c.lower()]
nan_hours = lga_df[lga_df.drop(columns=wind_cols).isna().any(axis=1)]
nan_times_local = nan_hours.index  # tz-naive, but *NY local*

# 3. Convert those NaN times (local NY) → UTC
nan_times_utc = (
    nan_times_local
    .tz_localize("America/New_York")   # interpret as NY local
    .tz_convert("UTC")                 # convert to UTC
)

tmin, tmax = nan_times_utc.min(), nan_times_utc.max()

# 4. Load RAW ISD in UTC
raw_fp = r"C:\Users\epicx\Projects\ubercasestudy\data\raw\weather\72505394728_2025-01-01_2025-08-27.csv"
raw = pd.read_csv(raw_fp, parse_dates=["DATE"])
raw["DATE"] = raw["DATE"].dt.tz_localize("UTC")  # ensure tz-aware UTC

# 5. Slice raw to that UTC window
raw_slice = raw[(raw["DATE"] >= tmin) & (raw["DATE"] <= tmax)]

# 6. Keep only rows whose UTC hour matches one of the NaN hours (also in UTC)
nan_hours_utc_floor = nan_times_utc.floor("H")

raw_nan = raw_slice[
    raw_slice["DATE"].dt.floor("H").isin(nan_hours_utc_floor)
]

# Inspect
print(len(raw_nan))
print(raw_nan["DATE"].dt.floor("H").value_counts().sort_index())
