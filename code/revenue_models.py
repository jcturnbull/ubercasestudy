# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 09:53:03 2025

Revenue / driver pay / margin vs weather models,
writing all results to a single Excel workbook.

@author: epicx
"""

import os
import pandas as pd
import statsmodels.api as sm
import patsy

# import custom utilities + baseline/lagged helpers
from baseline_weather_models import (
    fit_baseline_demand_model,
    add_weather_differentials,
)
from lagged_weather_effects import create_lagged_flags
from model_utils import coeff_table, sig_code, add_to_summary, clean_sheet_name


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
DATA_PATH = (
    r"C:\Users\epicx\Projects\ubercasestudy\data\processed\fhvhv_lga_hourly_with_weather_2025.parquet"
)
OUTPUT_XLSX_SLR = r"C:\Users\epicx\Projects\ubercasestudy\reports\revenue_weather_models_slr_results.xlsx"
OUTPUT_XLSX_MLR = r"C:\Users\epicx\Projects\ubercasestudy\reports\revenue_weather_models_mlr_results.xlsx"



# -------------------------------------------------------------------
# Helpers for model running
# -------------------------------------------------------------------
def run_ols(df: pd.DataFrame, y_col: str, x_cols, label: str):
    """
    Run OLS: y_col ~ x_cols on non-null rows.
    Returns (model, sub_df).
    """
    if isinstance(x_cols, str):
        x_cols = [x_cols]

    cols = [y_col] + list(x_cols)
    sub = df[cols].dropna()
    if sub.empty:
        print(f"[{label}] No data available (all NA) for {y_col} vs {x_cols}.")
        return None, None

    X = sm.add_constant(sub[x_cols])
    y = sub[y_col]

    model = sm.OLS(y, X).fit()
    print("=" * 90)
    print(label)
    print(f"Observations: {len(sub)}")
    print(model.summary().tables[1])
    print("=" * 90)
    return model, sub


# -------------------------------------------------------------------
# Metric construction helpers
# -------------------------------------------------------------------
def add_revenue_and_trip_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - fare_per_mile, fare_per_min
      - driverpay_per_mile, driverpay_per_min
      - margin_per_mile
      - avg_trip_miles, avg_trip_time_min
    Uses existing aggregate columns in df.
    """
    out = df.copy()

    miles = out["trip_miles_sum"]
    time_min = out["trip_time_sum"] / 60.0

    out["fare_per_mile"] = out["base_passenger_fare_sum"] / miles.replace(0, pd.NA)
    out["fare_per_min"] = out["base_passenger_fare_sum"] / time_min.replace(
        0, pd.NA
    )

    out["driverpay_per_mile"] = out["driver_pay_sum"] / miles.replace(0, pd.NA)
    out["driverpay_per_min"] = out["driver_pay_sum"] / time_min.replace(0, pd.NA)

    out["margin_per_mile"] = out["fare_per_mile"] - out["driverpay_per_mile"]

    out["avg_trip_miles"] = out["trip_miles_mean"]
    out["avg_trip_time_min"] = out["trip_time_mean"] / 60.0

    return out

def add_residuals_for_metrics(df: pd.DataFrame,
                              metrics: list[str],
                              time_col: str = "datetime_hour") -> tuple[pd.DataFrame, dict]:
    """
    For each metric in `metrics`, fit:
        metric ~ C(hour_of_week) + C(month)
    using existing hour_of_week and month columns in df, and add:
        metric_fitted, metric_resid
    Returns (df_with_resids, models_dict).
    """
    out = df.copy()
    models = {}

    for y_col in metrics:
        # Drop NA rows for this metric
        sub = out[[y_col, "hour_of_week", "month", time_col]].dropna()
        if sub.empty:
            print(f"[Residualization] No data for {y_col}, skipping.")
            continue

        formula = f"{y_col} ~ C(hour_of_week) + C(month)"
        y, X = patsy.dmatrices(formula, data=sub, return_type="dataframe")

        model = sm.OLS(y, X).fit()
        models[y_col] = model

        # model.fittedvalues and model.resid are already 1D with the right index
        fitted = pd.Series(model.fittedvalues, index=y.index)
        resid = pd.Series(model.resid, index=y.index)

        out.loc[fitted.index, f"{y_col}_fitted"] = fitted
        out.loc[resid.index, f"{y_col}_resid"] = resid

        print(f"[Residualization] Fitted {y_col} ~ C(hour_of_week) + C(month)")

    return out, models

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":

    # ensure output directory exists (once)
    out_dir = os.path.dirname(OUTPUT_XLSX_SLR)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Containers for SLR (single regressor) models
    summary_rows_slr = []
    model_dict_slr = {}

    # Containers for MLR (multi-regressor) models
    summary_rows_mlr = []
    model_dict_mlr = {}

    # 1. Load canonical dataset
    df = pd.read_parquet(DATA_PATH)

    # 2. Baseline residualization (imported from baseline_weather_models)
    df_resid, base_model = fit_baseline_demand_model(
        df,
        dep_var="request_count",
        time_col="datetime_hour",
        include_month_dummies=True,
    )

    # 3. Weather differentials and flags
    df_weather = add_weather_differentials(df_resid)

    # 4. Add rain and heavy-rain lag0/lag1 flags (reuse from lagged_weather_effects)
    df_weather = create_lagged_flags(df_weather, flag_col="rain_flag", max_lag=1)
    df_weather = create_lagged_flags(df_weather, flag_col="heavy_rain_flag", max_lag=1)
    
    df_weather["precip_lag0"] = df_weather["precip_1h_mm_total"]
    df_weather["precip_lag1"] = df_weather["precip_1h_mm_total"].shift(1)

    # ensure lag0 names exist explicitly (create_lagged_flags handles this, but explicit is fine)
    df_weather["rain_flag_lag0"] = df_weather["rain_flag"]
    df_weather["heavy_rain_flag_lag0"] = df_weather["heavy_rain_flag"]

    # 5. Add revenue, pay, margin, and trip metrics
    df_metrics = add_revenue_and_trip_metrics(df_weather)
    
    # ----------------------------------------------------------------
    # 6. Define modeling targets and regressors
    # ----------------------------------------------------------------
    metrics_to_resid = [
        "avg_base_passenger_fare",
        "avg_driver_pay",
        "fare_per_mile",
        "fare_per_min",
        "driverpay_per_mile",
        "driverpay_per_min",
        "margin_per_mile",
        "driver_pay_pct_of_base_fare",
        "avg_trip_miles",
        "avg_trip_time_min",
        "avg_speed_mph",
        "median_speed_mph",
    ]   

    df_metrics_resid, resid_models = add_residuals_for_metrics(
        df_metrics,
        metrics=metrics_to_resid,
        time_col="datetime_hour",
    )

    base_dep_vars = [
        "avg_base_passenger_fare",
        "avg_driver_pay",
        "fare_per_mile",
        "fare_per_min",
        "driverpay_per_mile",
        "driverpay_per_min",
        "margin_per_mile",
        "driver_pay_pct_of_base_fare",
        "avg_trip_miles",
        "avg_trip_time_min",
        "avg_speed_mph",
        "median_speed_mph",
    ]

    dep_vars = [f"{m}_resid" for m in base_dep_vars]

    single_regressors = [
        "rain_flag_lag0",
        "rain_flag_lag1",
        "heavy_rain_flag_lag0",
        "heavy_rain_flag_lag1",
        "wind_chill_diff",
        "heat_index_diff",
        "wind_chill_f",
        "heat_index_f",
        "precip_1h_mm_total",
        "precip_lag1"
    ]

    # ----------------------------------------------------------------
    # 7. Univariate models: each dep_var ~ single weather regressor
    # ----------------------------------------------------------------
    for y in dep_vars:
        for x in single_regressors:
            label = f"{y} ~ {x}"
            model, used = run_ols(df_metrics_resid, y_col=y, x_cols=x, label=label)
            if model is None:
                continue
            # SLR containers
            add_to_summary(summary_rows_slr, model, dep_var=y, label=label)
            model_dict_slr[label] = model

    # ----------------------------------------------------------------
    # 8. Simple multivariate: rain_flag_lag0 + rain_flag_lag1
    # ----------------------------------------------------------------
    combo_specs = [
        ( "fare_per_mile_resid", ["rain_flag_lag0", "rain_flag_lag1"] ),
        ( "driverpay_per_mile_resid", ["rain_flag_lag0", "rain_flag_lag1"] ),
        ( "margin_per_mile_resid", ["rain_flag_lag0", "rain_flag_lag1"] ),
    ]

    for y, xs in combo_specs:
        label = f"{y} ~ rain_flag_lag0 + rain_flag_lag1"
        model, used = run_ols(df_metrics_resid, y_col=y, x_cols=xs, label=label)
        if model is None:
            continue
        # MLR containers
        add_to_summary(summary_rows_mlr, model, dep_var=y, label=label)
        model_dict_mlr[label] = model

    # ----------------------------------------------------------------
    # 9. Write Excel: Summary + per-model sheets
    # ----------------------------------------------------------------
    # --- SLR workbook ---
    summary_df_slr = pd.DataFrame(summary_rows_slr)
    
    with pd.ExcelWriter(OUTPUT_XLSX_SLR, engine="xlsxwriter") as writer:
        summary_df_slr.to_excel(writer, sheet_name="Summary", index=False)
        for label, model in model_dict_slr.items():
            sheet = clean_sheet_name(label)
            coef_df = coeff_table(model, drop_const=False)
            coef_df.to_excel(writer, sheet_name=sheet, index=False)

    print(f"\nSLR revenue-weather model results written to: {OUTPUT_XLSX_SLR}")
    
    # --- MLR workbook ---
    summary_df_mlr = pd.DataFrame(summary_rows_mlr)
    
    with pd.ExcelWriter(OUTPUT_XLSX_MLR, engine="xlsxwriter") as writer:
        summary_df_mlr.to_excel(writer, sheet_name="Summary", index=False)
        for label, model in model_dict_mlr.items():
            sheet = clean_sheet_name(label)
            coef_df = coeff_table(model, drop_const=False)
            coef_df.to_excel(writer, sheet_name=sheet, index=False)

    print(f"MLR revenue-weather model results written to: {OUTPUT_XLSX_MLR}")

