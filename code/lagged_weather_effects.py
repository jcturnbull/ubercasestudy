# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 22:30:33 2025

@author: epicx
"""

import pandas as pd
import statsmodels.api as sm
from baseline_weather_models import (
    fit_baseline_demand_model,
    add_weather_differentials,
)


# -----------------------------------------------------------
# Helper to create binary rain lag variables
# -----------------------------------------------------------
def create_lagged_flags(df, flag_col="rain_flag", max_lag=3):
    out = df.copy()
    for lag in range(max_lag + 1):
        out[f"{flag_col}_lag{lag}"] = out[flag_col].shift(lag)
    return out


# -----------------------------------------------------------
# Fit multivariate OLS with arbitrary regressors
# -----------------------------------------------------------
def run_multivariate_ols(df, y_col, x_cols, label):
    sub = df[[y_col] + x_cols].dropna()
    X = sm.add_constant(sub[x_cols])
    y = sub[y_col]
    model = sm.OLS(y, X).fit()

    print("=" * 90)
    print(f"{label}")
    print(f"Observations: {len(sub)}")
    print(model.summary().tables[1])  # coefficients only
    print("=" * 90)
    return model, sub


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
if __name__ == "__main__":

    # 1. Load merged weather + TLC dataset
    PATH = r"C:\Users\epicx\Projects\ubercasestudy\data\processed\fhvhv_lga_hourly_with_weather_2025.parquet"
    df = pd.read_parquet(PATH)

    # 2. Residualize demand using existing baseline model
    df_resid, base_model = fit_baseline_demand_model(
        df,
        dep_var="request_count",
        time_col="datetime_hour",
        include_month_dummies=True,
    )

    # 3. Add weather differentials, rain flags, heavy rain flags
    df_weather = add_weather_differentials(df_resid)

    # 4. Create lagged versions (0 = now, 1, 2, 3 hours ago)
    df_weather = create_lagged_flags(df_weather, flag_col="rain_flag", max_lag=3)
    df_weather = create_lagged_flags(df_weather, flag_col="heavy_rain_flag", max_lag=3)

    # 5. Model A: Demand_t ~ rain_flag_t + rain_flag_(t-1) + rain_flag_(t-2) + rain_flag_(t-3)
    lag_cols_rain = ["rain_flag_lag0", "rain_flag_lag1", "rain_flag_lag2", "rain_flag_lag3"]

    rain_model, rain_data = run_multivariate_ols(
        df_weather,
        y_col="demand_resid",
        x_cols=lag_cols_rain,
        label="Lagged Model: demand_resid ~ rain_flag (0–3 hour lags)",
    )

    # 6. Model B: same structure for heavy rain
    lag_cols_heavy = ["heavy_rain_flag_lag0", "heavy_rain_flag_lag1", "heavy_rain_flag_lag2", "heavy_rain_flag_lag3"]

    heavy_model, heavy_data = run_multivariate_ols(
        df_weather,
        y_col="demand_resid",
        x_cols=lag_cols_heavy,
        label="Lagged Model: demand_resid ~ heavy_rain_flag (0–3 hour lags)",
    )

    # -----------------------------------------------------------
    # OPTIONAL: Combined Model C
    # Heavy rain flag + precipitation amount (mm)
    # Question: Given that heavy rain occurred, does *more* rain matter?
    # -----------------------------------------------------------

    df_weather["precip_lag0"] = df_weather["precip_1h_mm_total"]
    df_weather["precip_lag1"] = df_weather["precip_1h_mm_total"].shift(1)
    df_weather["precip_lag2"] = df_weather["precip_1h_mm_total"].shift(2)
    df_weather["precip_lag3"] = df_weather["precip_1h_mm_total"].shift(3)

    # Multivariate: heavy_rain + precip amount
    multi_cols = ["heavy_rain_flag_lag0", "precip_lag0"]

    combo_model, combo_data = run_multivariate_ols(
        df_weather,
        y_col="demand_resid",
        x_cols=multi_cols,
        label="Combined Model: demand_resid ~ heavy_rain_flag + precip_amount",
    )
    
