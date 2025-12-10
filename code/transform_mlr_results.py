# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 20:55:12 2025

@author: epicx
"""

import numpy as np
import pandas as pd

df = pd.read_excel(r"C:\Users\epicx\Projects\ubercasestudy\reports\revenue_weather_models_mlr_results.xlsx")

# (optional) create short model names like "E0-fare_per_mile_resid"
# You can just hard-code a mapping once you know which rows are which.
model_map = {
    # full model_label string : "E0-fare_per_mile_resid" etc.
    # "avg_base_passenger_fare_resid ~ rain_flag_lag0 + rain_flag_lag1": "E0-avg_base",
}
df["model_short"] = df["model_label"].map(model_map).fillna(df["model_label"])

records = []
for _, row in df.iterrows():
    m = row["model_short"]

    # global stats
    records.append({"model": m, "metric": "R2",      "coef": row["r2"],     "sig": "", "p": np.nan})
    records.append({"model": m, "metric": "adj_R2",  "coef": row["r2_adj"], "sig": "", "p": np.nan})
    records.append({"model": m, "metric": "nObs",    "coef": row["n_obs"],  "sig": "", "p": np.nan})

    # per-variable stats
    for i in range(1, 9):
        var = row.get(f"var_{i}")
        if isinstance(var, str) and var:   # skip NaNs
            records.append({
                "model":  m,
                "metric": var,
                "coef":   row.get(f"coef_{i}"),
                "sig":    row.get(f"sig_{i}"),
                "p":      row.get(f"p_{i}"),
            })

long_df = pd.DataFrame(records)

def format_p(x):
    if pd.isna(x):
        return ""
    if x < 0.001:
        return "< 0.001"
    return f"{x:.3f}"

long_df["p_fmt"] = long_df["p"].apply(format_p)

coef_wide = long_df.pivot_table(index="metric", columns="model", values="coef", aggfunc="first")
sig_wide  = long_df.pivot_table(index="metric", columns="model", values="sig",  aggfunc="first")
p_wide    = long_df.pivot_table(index="metric", columns="model", values="p_fmt",aggfunc="first")

# combine into a single table with a 2-level column index: (model, stat)
wide = pd.concat(
    {"coef": coef_wide, "sig": sig_wide, "p": p_wide},
    axis=1
).swaplevel(axis=1).sort_index(axis=1)   # columns: model -> (coef, sig, p)

row_order = [
    "R2", "adj_R2", "nObs",
    "avg_trip_miles",
    "avg_trip_time_min",
    "demand_resid",
    "driver_pay_pct_of_base_fare",
    "rain_flag_lag0",
    "heavy_rain_flag_lag0",
    "precip_1h_mm_total",
    "wind_chill_f",
]

wide = wide.reindex(row_order)
