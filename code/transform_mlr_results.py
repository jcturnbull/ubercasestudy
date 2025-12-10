# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 20:55:12 2025

@author: epicx
"""

import numpy as np
import pandas as pd

# === 1. Load raw MLR output ===
df = pd.read_excel(
    r"C:\Users\epicx\Projects\ubercasestudy\reports\revenue_weather_models_mlr_results.xlsx"
)

# (optional) mapping from long model names → clean-short names
model_map = {
    # "avg_base_passenger_fare_resid ~ ...": "E0-base",
}
df["model_short"] = df["model_label"].map(model_map).fillna(df["model_label"])

# === 2. Build long-form table ===
records = []

for _, row in df.iterrows():
    m = row["model_short"]

    records.append({"model": m, "metric": "R2",     "coef": row["r2"],     "sig": "", "p": np.nan})
    records.append({"model": m, "metric": "adj_R2", "coef": row["r2_adj"], "sig": "", "p": np.nan})
    records.append({"model": m, "metric": "nObs",   "coef": row["n_obs"],  "sig": "", "p": np.nan})

    for i in range(1, 9):
        var = row.get(f"var_{i}")
        if isinstance(var, str) and var:
            records.append({
                "model": m,
                "metric": var,
                "coef": row.get(f"coef_{i}"),
                "sig":  row.get(f"sig_{i}"),
                "p":    row.get(f"p_{i}"),
            })

long_df = pd.DataFrame(records)

# === 3. Format p-values ===
def format_p(x):
    if pd.isna(x):
        return ""
    if x < 0.001:
        return "< 0.001"
    return f"{x:.3f}"

long_df["p_fmt"] = long_df["p"].apply(format_p)

# === 4. Pivot to wide ===
coef_wide = long_df.pivot_table(index="metric", columns="model", values="coef",    aggfunc="first")
sig_wide  = long_df.pivot_table(index="metric", columns="model", values="sig",     aggfunc="first")
p_wide    = long_df.pivot_table(index="metric", columns="model", values="p_fmt",   aggfunc="first")

coef_wide = coef_wide.round(5)

coef_wide.columns = pd.MultiIndex.from_product([coef_wide.columns, ["coef"]])
sig_wide.columns  = pd.MultiIndex.from_product([sig_wide.columns,  ["sig"]])
p_wide.columns    = pd.MultiIndex.from_product([p_wide.columns,    ["p"]])

wide = pd.concat([coef_wide, sig_wide, p_wide], axis=1)
wide = wide.sort_index(axis=1, level=0)

# === 5. Order rows ===
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

# === 6. FIX: convert "model.stat" → MultiIndex (model, stat) ===
def split_col(col):
    # col is a tuple: (model_name, "coef") because of MultiIndex
    # so just return as-is
    return col

# Already correct structure: (model, stat)
# but ensure index name is clean
wide.index.name = None

# === 7. Export ===
output_path = r"C:\Users\epicx\Projects\ubercasestudy\reports\mlr_model_comparison.xlsx"

wide.to_excel(
    output_path,
    sheet_name="comparison",
    merge_cells=False,
    freeze_panes=(2, 1)   # freeze top 2 header rows + first column
)

print("Export complete:", output_path)
