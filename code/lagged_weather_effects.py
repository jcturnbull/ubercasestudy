# -*- coding: utf-8 -*-
"""
Lagged rain / heavy-rain / precipitation models on residual demand.

Generates:
  - Iterative cumulative lag models:
      demand_resid ~ X_lag0
      demand_resid ~ X_lag0 + X_lag1
      ...
      demand_resid ~ X_lag0 + X_lag1 + X_lag2 + X_lag3
  - Single-lag-only models:
      demand_resid ~ X_lag0
      demand_resid ~ X_lag1
      demand_resid ~ X_lag2
      demand_resid ~ X_lag3

for:
  - rain_flag
  - heavy_rain_flag
  - precip_1h_mm_total (via precip_lag0..3)

Also supports arbitrary "interesting" combinations of regressors.

@author: epicx
"""

import os
import pandas as pd
import statsmodels.api as sm

from model_utils import coeff_table, sig_code, add_to_summary, clean_sheet_name
from baseline_weather_models import (
    fit_baseline_demand_model,
    add_weather_differentials,
)

LAGGED_XLSX = (
    r"C:\Users\epicx\Projects\ubercasestudy\reports\lagged_weather_effects_results.xlsx"
)

# Master path for merged TLC + weather hourly data (2025)
PATH = (
    r"C:\Users\epicx\Projects\ubercasestudy\data\processed\fhvhv_lga_hourly_with_weather_2025.parquet"
)


# -----------------------------------------------------------
# Helper to create binary rain / heavy-rain lag variables
# -----------------------------------------------------------
def create_lagged_flags(df, flag_col: str, max_lag: int = 3) -> pd.DataFrame:
    """
    Create lag0..lagN columns for a binary flag (0/1) such as rain_flag or heavy_rain_flag.

    Example:
        flag_col = "rain_flag", max_lag = 3
        -> rain_flag_lag0, rain_flag_lag1, rain_flag_lag2, rain_flag_lag3
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
    print(label)
    print(f"Observations: {len(sub)}")
    print(model.summary().tables[1])  # coefficients table only
    print("=" * 90)
    return model, sub


# -----------------------------------------------------------
# Iterative cumulative lag models
# -----------------------------------------------------------
def run_iterative_lag_models(df, base_col: str, max_lag: int, label_prefix: str,
                             summary_rows: list = None, model_dict: dict = None):
    """
    Runs cumulative lag models:

        demand_resid ~ base_col_lag0
        demand_resid ~ base_col_lag0 + base_col_lag1
        ...
        demand_resid ~ base_col_lag0 + ... + base_col_lag{max_lag}

    base_col is the un-suffixed name, e.g.:
        "rain_flag"         -> uses rain_flag_lag0..3
        "heavy_rain_flag"   -> uses heavy_rain_flag_lag0..3
        "precip_lag"        -> uses precip_lag0..3
    """
    for L in range(max_lag + 1):
        lag_cols = [f"{base_col}_lag{i}" for i in range(L + 1)]
        label = f"{label_prefix} ~ " + "+".join(lag_cols)

        model, data = run_multivariate_ols(
            df,
            y_col="demand_resid",
            x_cols=lag_cols,
            label=label,
        )

        if (summary_rows is not None) and (model_dict is not None) and (model is not None):
            add_to_summary(
                summary_rows,
                model,
                dep_var="demand_resid",
                label=label,
            )
            model_dict[label] = model


# -----------------------------------------------------------
# Single-lag-only models for flags
# -----------------------------------------------------------
def run_single_lag_models(df, base_col: str, max_lag: int, label_prefix: str,
                          summary_rows: list = None, model_dict: dict = None):
    """
    For a given base_col (e.g., 'rain_flag'), run:

        demand_resid ~ base_col_lag0
        demand_resid ~ base_col_lag1
        ...
        demand_resid ~ base_col_lag{max_lag}
    """
    for lag in range(max_lag + 1):
        col = f"{base_col}_lag{lag}"
        label = f"{label_prefix} ~ {col}"

        model, data = run_multivariate_ols(
            df,
            y_col="demand_resid",
            x_cols=[col],
            label=label,
        )

        if (summary_rows is not None) and (model_dict is not None) and (model is not None):
            add_to_summary(
                summary_rows,
                model,
                dep_var="demand_resid",
                label=label,
            )
            model_dict[label] = model


# -----------------------------------------------------------
# Single-lag-only models for precipitation (continuous)
# -----------------------------------------------------------
def run_single_precip_lag_models(df, max_lag: int, label_prefix: str,
                                 summary_rows: list = None, model_dict: dict = None):
    """
    Runs:

        demand_resid ~ precip_lag0
        demand_resid ~ precip_lag1
        ...
        demand_resid ~ precip_lag{max_lag}
    """
    for lag in range(max_lag + 1):
        col = f"precip_lag{lag}"
        label = f"{label_prefix} ~ {col}"

        model, data = run_multivariate_ols(
            df,
            y_col="demand_resid",
            x_cols=[col],
            label=label,
        )

        if (summary_rows is not None) and (model_dict is not None) and (model is not None):
            add_to_summary(
                summary_rows,
                model,
                dep_var="demand_resid",
                label=label,
            )
            model_dict[label] = model


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
if __name__ == "__main__":

    summary_rows = []
    model_dict = {}

    # 1. Load merged weather + TLC dataset
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

    # 4. Create lagged versions (0 = now, up to 3 hours ago)
    df_weather = create_lagged_flags(df_weather, flag_col="rain_flag", max_lag=3)
    df_weather = create_lagged_flags(df_weather, flag_col="heavy_rain_flag", max_lag=3)

    # 5. Precipitation lags (continuous)
    df_weather["precip_lag0"] = df_weather["precip_1h_mm_total"]
    df_weather["precip_lag1"] = df_weather["precip_1h_mm_total"].shift(1)
    df_weather["precip_lag2"] = df_weather["precip_1h_mm_total"].shift(2)
    df_weather["precip_lag3"] = df_weather["precip_1h_mm_total"].shift(3)

    # ======================================================================
    # 6. ITERATIVE CUMULATIVE LAG MODELS
    # ======================================================================

    # 6A. Rain flag: cumulative lags (0 → 3)
    run_iterative_lag_models(
        df_weather,
        base_col="rain_flag",
        max_lag=3,
        label_prefix="demand_resid",
        summary_rows=summary_rows,
        model_dict=model_dict,
    )

    # 6B. Heavy rain flag: cumulative lags (0 → 3)
    run_iterative_lag_models(
        df_weather,
        base_col="heavy_rain_flag",
        max_lag=3,
        label_prefix="demand_resid",
        summary_rows=summary_rows,
        model_dict=model_dict,
    )

    # 6C. Precipitation amount: cumulative lags (0 → 3)
    #     Uses precip_lag0..3
    run_iterative_lag_models(
        df_weather,
        base_col="precip",
        max_lag=3,
        label_prefix="demand_resid",
        summary_rows=summary_rows,
        model_dict=model_dict,
    )

    # ======================================================================
    # 7. SINGLE-LAG-ONLY MODELS
    # ======================================================================

    # 7A. Rain flag single-lag models
    run_single_lag_models(
        df_weather,
        base_col="rain_flag",
        max_lag=3,
        label_prefix="demand_resid",
        summary_rows=summary_rows,
        model_dict=model_dict,
    )

    # 7B. Heavy rain flag single-lag models
    run_single_lag_models(
        df_weather,
        base_col="heavy_rain_flag",
        max_lag=3,
        label_prefix="demand_resid",
        summary_rows=summary_rows,
        model_dict=model_dict,
    )

    # 7C. Precipitation amount single-lag models
    run_single_precip_lag_models(
        df_weather,
        max_lag=3,
        label_prefix="demand_resid",
        summary_rows=summary_rows,
        model_dict=model_dict,
    )

    # ======================================================================
    # 8. INTERESTING CUSTOM COMBINATIONS (EDITABLE)
    # ======================================================================

    # Add any custom regressor combinations you find interesting here.
    # Example:
    #   ["heavy_rain_flag_lag0", "precip_lag0"]
    #   ["rain_flag_lag0", "precip_lag1"]
    #   ["heavy_rain_flag_lag1", "precip_lag0", "precip_lag1"]
    '''interesting_combos = [
        # 1. wind chill diff + rain_flag_lag0 + rain_flag_lag1
        ["wind_chill_diff", "rain_flag_lag0", "rain_flag_lag1"],

        # 2. rain_flag_lag0 + rain_flag_lag1 + heavy_rain_flag_lag0 + heavy_rain_flag_lag1
        ["rain_flag_lag0", "rain_flag_lag1", "heavy_rain_flag_lag0", "heavy_rain_flag_lag1"],

        # 3. wind chill (wind_chill_f) + rain_flag_lag0 + rain_flag_lag1
        ["wind_chill_f", "rain_flag_lag0", "rain_flag_lag1"],

        # 4. wind chill + wind chill diff
        ["wind_chill_f", "wind_chill_diff"],

        # 5. wind chill + wind chill diff + precip_lag0 + precip_lag1 + precip_lag2
        ["wind_chill_f", "wind_chill_diff", "precip_lag0", "precip_lag1", "precip_lag2"],

        # 6. rain_flag_lag1 + precip_lag0
        ["rain_flag_lag1", "precip_lag0"],

        # 7. heavy_rain_flag_lag0 + precip_lag0
        ["heavy_rain_flag_lag0", "precip_lag0"],

        # 8. wind chill + rain_flag_lag1 + precip_lag0
        ["wind_chill_f", "rain_flag_lag1", "precip_lag0"],

        # 9. wind chill diff + rain_flag_lag1 + precip_lag0
        ["wind_chill_diff", "rain_flag_lag1", "precip_lag0"],
    ]'''
    interesting_combos = [
        # 1. wind chill + precip_lag0 + rain_flag_lag0
        ["wind_chill_f", "precip_lag0", "rain_flag_lag0"],

        # 2. wind chill + precip_lag0 + rain_flag_lag1
        ["wind_chill_f", "precip_lag0", "rain_flag_lag1"],

        # 3. wind chill + precip_lag1 + rain_flag_lag0
        ["wind_chill_f", "precip_lag1", "rain_flag_lag0"],
    ]

    for combo in interesting_combos:
        label = "demand_resid ~ " + "+".join(combo)
        model, data = run_multivariate_ols(
            df_weather,
            y_col="demand_resid",
            x_cols=combo,
            label=label,
        )
        if model is not None:
            add_to_summary(summary_rows, model, dep_var="demand_resid", label=label)
            model_dict[label] = model

    # ======================================================================
    # 9. WRITE EXCEL SUMMARY + PER-MODEL SHEETS
    # ======================================================================

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
