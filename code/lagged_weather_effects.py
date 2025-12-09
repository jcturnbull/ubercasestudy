# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 22:30:33 2025

Lagged rain/heavy-rain models on residual demand, plus combined heavy rain + precip.

Uses only lag0 and lag1 as agreed (no lag2/lag3 in final models). 

@author: epicx
"""

import os
import pandas as pd
import statsmodels.api as sm

# import custom utilities
from model_utils import coeff_table, sig_code, add_to_summary, clean_sheet_name
from baseline_weather_models import (
    fit_baseline_demand_model,
    add_weather_differentials,
)

LAGGED_XLSX = (
    r"C:\Users\epicx\Projects\ubercasestudy\reports\lagged_weather_effects_results.xlsx"
)


# -----------------------------------------------------------
# Helper to create binary rain lag variables
# -----------------------------------------------------------
def create_lagged_flags(df, flag_col="rain_flag", max_lag=1):
    """
    Create lag0..lagN columns for a binary flag (by default 0 and 1 only).
    """
    out = df.copy()
    for lag in range(max_lag + 1):
        out[f"{flag_col}_lag{lag}"] = out[flag_col].shift(lag)
    return out


# -----------------------------------------------------------
# Fit multivariate OLS with arbitrary regressors
# -----------------------------------------------------------
def run_multivariate_ols(df, y_col, x_cols, label):
    sub = df[[y_col] + x_cols].dropna()
    if sub.empty:
        print(f"[{label}] No data available (all NA).")
        return None, None

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

    summary_rows = []
    model_dict = {}

    # 1. Load merged weather + TLC dataset
    PATH = (
        r"C:\Users\epicx\Projects\ubercasestudy\data\processed\fhvhv_lga_hourly_with_weather_2025.parquet"
    )
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

    # 4. Create lagged versions (0 = now, 1 hour ago); no 2/3 lags per your call
    df_weather = create_lagged_flags(df_weather, flag_col="rain_flag", max_lag=3)
    df_weather = create_lagged_flags(df_weather, flag_col="heavy_rain_flag", max_lag=3)

    # 5. Model A: demand_t ~ rain_flag_t + rain_flag_(t-1)
    lag_cols_rain = ["rain_flag_lag0", "rain_flag_lag1"]

    rain_model, rain_data = run_multivariate_ols(
        df_weather,
        y_col="demand_resid",
        x_cols=lag_cols_rain,
        label="Lagged Model: demand_resid ~ rain_flag (0–1 hour lags)",
    )
    if rain_model is not None:
        add_to_summary(
            summary_rows,
            rain_model,
            dep_var="demand_resid",
            label="demand_resid ~ rain_flag_lag0 + rain_flag_lag1",
        )
        model_dict["demand_resid ~ rain_flag_lag0+1"] = rain_model

    # 6. Model B: same structure for heavy rain
    lag_cols_heavy = ["heavy_rain_flag_lag0", "heavy_rain_flag_lag1"]

    heavy_model, heavy_data = run_multivariate_ols(
        df_weather,
        y_col="demand_resid",
        x_cols=lag_cols_heavy,
        label="Lagged Model: demand_resid ~ heavy_rain_flag (0–1 hour lags)",
    )
    if heavy_model is not None:
        add_to_summary(
            summary_rows,
            heavy_model,
            dep_var="demand_resid",
            label="demand_resid ~ heavy_rain_flag_lag0 + heavy_rain_flag_lag1",
        )
        model_dict["demand_resid ~ heavy_rain_flag_lag0+1"] = heavy_model

    # -----------------------------------------------------------
    # 7. Combined Model C: heavy_rain_flag_lag0 + precip amount
    # -----------------------------------------------------------

    df_weather["precip_lag0"] = df_weather["precip_1h_mm_total"]
    df_weather["precip_lag1"] = df_weather["precip_1h_mm_total"].shift(1)

    multi_cols = ["heavy_rain_flag_lag0", "precip_lag0"]

    combo_model, combo_data = run_multivariate_ols(
        df_weather,
        y_col="demand_resid",
        x_cols=multi_cols,
        label="Combined Model: demand_resid ~ heavy_rain_flag_lag0 + precip_lag0",
    )
    if combo_model is not None:
        add_to_summary(
            summary_rows,
            combo_model,
            dep_var="demand_resid",
            label="demand_resid ~ heavy_rain_flag_lag0 + precip_lag0",
        )
        model_dict["demand_resid ~ heavy_rain_flag_lag0 + precip_lag0"] = combo_model

    # -----------------------------------------------------------
    # 8. Write Excel summary + per-model sheets
    # -----------------------------------------------------------
    summary_df = pd.DataFrame(summary_rows)

    out_dir = os.path.dirname(LAGGED_XLSX)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with pd.ExcelWriter(LAGGED_XLSX, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        for label, model in model_dict.items():
            sheet = clean_sheet_name(label)
            coef_df = coeff_table(model, drop_const=False)
            coef_df.to_excel(writer, sheet_name=sheet, index=False)

    print(f"\nLagged models written to: {LAGGED_XLSX}")
