# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 22:44:54 2025

@author: epicx

# -*- coding: utf-8 -*-

Weather monetization models

This script:
  1. Loads the canonical merged LGA hourly dataset.
  2. Residualizes demand (demand_resid) using hour_of_week + month FEs.
  3. Adds weather differentials, rain / heavy-rain flags, and lag0 precip.
  4. Adds revenue/trip metrics, residualizes key economics metrics.
  5. Runs:
       (A) demand_resid ~ wind_chill_f + heavy_rain_flag_lag0 + precip_lag0
       (B) avg_base_passenger_fare_resid  (E0(f), E1(f))
       (C) margin_per_mile_resid          (E0(m), E1(m))
  6. Reconstructs full-level predictions by adding residual predictions
     back to their seasonal baselines:
       demand_hat_weather
       avg_base_passenger_fare_hat_E0 / _E1
       margin_per_mile_hat_E0 / _E1

You can then filter to a given date (e.g. 2025-03-20) and compare:
  - actual vs predicted vs historical baselines (to be added separately).
"""

import os
import pandas as pd
import statsmodels.api as sm

from baseline_weather_models import (
    fit_baseline_demand_model,
    add_weather_differentials,
)
from lagged_weather_effects import create_lagged_flags
from revenue_models import (
    add_revenue_and_trip_metrics,
    add_residuals_for_metrics,
    run_ols,
)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

DATA_PATH = (
    r"C:\Users\epicx\Projects\ubercasestudy\data\processed\fhvhv_lga_hourly_with_weather_2025.parquet"
)


# -----------------------------------------------------------------------------
# Build master modeling DataFrame
# -----------------------------------------------------------------------------

def build_master_df(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load merged TLC + weather data and construct all residualized metrics
    needed for the weather monetization models.

    Steps:
      1. Baseline demand residualization -> demand_fitted, demand_resid
      2. Weather differentials + rain / heavy-rain flags
      3. Lagged rain / heavy-rain flags (lag0 explicitly)
      4. precip_lag0 alias from precip_1h_mm_total
      5. Revenue/trip metrics (fare_per_mile, margin_per_mile, etc.)
      6. Residualize key economics metrics:
           - avg_base_passenger_fare
           - margin_per_mile
           - avg_trip_miles
           - avg_trip_time_min
           - driver_pay_pct_of_base_fare
           - avg_speed_mph
           - median_speed_mph
    """
    # 1. Load
    df = pd.read_parquet(path)

    # 2. Baseline demand residualization
    # NOTE: fit_baseline_demand_model hard-codes 'demand_fitted' and 'demand_resid'
    df_resid, base_model = fit_baseline_demand_model(
        df,
        dep_var="request_count",          # dep_var used in patsy formula
        time_col="datetime_hour",
        include_month_dummies=True,
    )
    print("Baseline demand model fitted; demand_resid added.")

    # 3. Weather differentials + rain/heavy-rain flags
    df_weather = add_weather_differentials(df_resid)
    print("Weather differentials and basic flags added.")

    # 4. Lagged flags (we only really need lag0, but this is future-proof)
    df_weather = create_lagged_flags(df_weather, flag_col="rain_flag", max_lag=0)
    df_weather = create_lagged_flags(df_weather, flag_col="heavy_rain_flag", max_lag=0)

    # explicit aliases
    df_weather["rain_flag_lag0"] = df_weather["rain_flag_lag0"]
    df_weather["heavy_rain_flag_lag0"] = df_weather["heavy_rain_flag_lag0"]

    # 5. Precipitation lag0 = current hour precip
    df_weather["precip_lag0"] = df_weather["precip_1h_mm_total"]
    
    # demand_resid lag 1 (for use in all downstream models)
    df_weather["demand_resid_lag1"] = df_resid["demand_resid"].shift(1)

    # 6. Add revenue, pay, margin, and trip metrics
    df_metrics = add_revenue_and_trip_metrics(df_weather)
    print("Revenue and trip metrics added.")

    # 7. Residualize economics metrics by hour_of_week + month
    metrics_to_resid = [
        "avg_base_passenger_fare",
        "margin_per_mile",
        "avg_trip_miles",
        "avg_trip_time_min",
        "driver_pay_pct_of_base_fare",
        "avg_speed_mph",
        "median_speed_mph",
    ]

    df_metrics_resid, resid_models = add_residuals_for_metrics(
        df_metrics,
        metrics=metrics_to_resid,
        time_col="datetime_hour",
    )
    print("Economics metrics residualized with hour_of_week + month FEs.")

    return df_metrics_resid


# -----------------------------------------------------------------------------
# Core modeling routines
# -----------------------------------------------------------------------------

def run_weather_monetization_models(df: pd.DataFrame):
    """
    Run the three key model groups and keep both models and estimation subsets.

      (1) Demand shock model:
            demand_resid ~ wind_chill_f + heavy_rain_flag_lag0 + precip_lag0

      (2) Fare models:
            E0(f): avg_base_passenger_fare_resid ~ base_controls
            E1(f): avg_base_passenger_fare_resid ~ base_controls + weather_block

      (3) Margin models:
            E0(m): margin_per_mile_resid ~ base_controls
            E1(m): margin_per_mile_resid ~ base_controls + weather_block
    """
    results = {}

    # ------------------------------------------------------------------
    # 1. Demand shock model
    # ------------------------------------------------------------------
    demand_x = ["wind_chill_f", "heavy_rain_flag_lag0", "precip_lag0"]
    label_demand = "demand_resid ~ wind_chill_f + heavy_rain_flag_lag0 + precip_lag0"

    m_demand, sub_demand = run_ols(
        df,
        y_col="demand_resid",
        x_cols=demand_x,
        label=label_demand,
    )
    results["demand"] = {"model": m_demand, "sub": sub_demand}

    # ------------------------------------------------------------------
    # 2. Fare models: E0(f), E1(f)
    # ------------------------------------------------------------------
    base_controls = [
        "avg_trip_miles",
        "avg_trip_time_min",
        "demand_resid_lag1",
        "driver_pay_pct_of_base_fare",
    ]

    weather_block = [
        "rain_flag_lag0",
        "heavy_rain_flag_lag0",
        "precip_1h_mm_total",
        "wind_chill_f",
    ]

    fare_y = "avg_base_passenger_fare_resid"
    margin_y = "margin_per_mile_resid"

    # E0(f): baseline economics, no weather
    label_E0_f = f"{fare_y} ~ " + " + ".join(base_controls)
    m_E0_f, sub_E0_f = run_ols(
        df,
        y_col=fare_y,
        x_cols=base_controls,
        label=label_E0_f,
    )
    results["E0_fare"] = {"model": m_E0_f, "sub": sub_E0_f}

    # E1(f): baseline + weather
    x_E1_f = base_controls + weather_block
    label_E1_f = f"{fare_y} ~ " + " + ".join(x_E1_f)
    m_E1_f, sub_E1_f = run_ols(
        df,
        y_col=fare_y,
        x_cols=x_E1_f,
        label=label_E1_f,
    )
    results["E1_fare"] = {"model": m_E1_f, "sub": sub_E1_f}

    # ------------------------------------------------------------------
    # 3. Margin models: E0(m), E1(m)
    # ------------------------------------------------------------------
    label_E0_m = f"{margin_y} ~ " + " + ".join(base_controls)
    m_E0_m, sub_E0_m = run_ols(
        df,
        y_col=margin_y,
        x_cols=base_controls,
        label=label_E0_m,
    )
    results["E0_margin"] = {"model": m_E0_m, "sub": sub_E0_m}

    x_E1_m = base_controls + weather_block
    label_E1_m = f"{margin_y} ~ " + " + ".join(x_E1_m)
    m_E1_m, sub_E1_m = run_ols(
        df,
        y_col=margin_y,
        x_cols=x_E1_m,
        label=label_E1_m,
    )
    results["E1_margin"] = {"model": m_E1_m, "sub": sub_E1_m}

    # ------------------------------------------------------------------
    # 4. Quick adj-R² comparison
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    if m_demand is not None:
        print("Demand model adj R²:", f"{m_demand.rsquared_adj:0.4f}")

    if m_E0_f is not None and m_E1_f is not None:
        print("\nFare models:")
        print("  E0(f) adj R²:", f"{m_E0_f.rsquared_adj:0.4f}")
        print("  E1(f) adj R²:", f"{m_E1_f.rsquared_adj:0.4f}")
        print("  Δ adj R²:", f"{m_E1_f.rsquared_adj - m_E0_f.rsquared_adj:0.4f}")

    if m_E0_m is not None and m_E1_m is not None:
        print("\nMargin models:")
        print("  E0(m) adj R²:", f"{m_E0_m.rsquared_adj:0.4f}")
        print("  E1(m) adj R²:", f"{m_E1_m.rsquared_adj:0.4f}")
        print("  Δ adj R²:", f"{m_E1_m.rsquared_adj - m_E0_m.rsquared_adj:0.4f}")
    print("-" * 80 + "\n")

    return results


# -----------------------------------------------------------------------------
# Turn predicted residuals back into full values
# -----------------------------------------------------------------------------

def add_level_predictions(df: pd.DataFrame, results: dict) -> pd.DataFrame:
    """
    For each residual model, construct full-level predictions by adding the
    model-predicted residuals back to the seasonal baseline.

    Output columns added:
      - demand_hat_weather
      - avg_base_passenger_fare_hat_E0
      - avg_base_passenger_fare_hat_E1
      - margin_per_mile_hat_E0
      - margin_per_mile_hat_E1
    """

    out = df.copy()

    # 1. Demand: demand_resid model
    res = results.get("demand")
    if res and res["model"] is not None and res["sub"] is not None:
        m = res["model"]
        sub = res["sub"]
        # model.fittedvalues are predicted demand_resid
        resid_hat = m.fittedvalues
        # baseline stored as "demand_fitted" from fit_baseline_demand_model
        base = out.loc[sub.index, "demand_fitted"]
        out.loc[sub.index, "demand_hat_weather"] = base + resid_hat

    # Helper to do the same for fare and margin
    def _add_level(metric_base: str, key_E0: str, key_E1: str,
                   base_fitted_col: str, resid_col: str):
        # E0
        res0 = results.get(key_E0)
        if res0 and res0["model"] is not None and res0["sub"] is not None:
            m0 = res0["model"]
            sub0 = res0["sub"]
            resid_hat0 = m0.fittedvalues  # predicted metric_resid
            base0 = out.loc[sub0.index, base_fitted_col]
            col_name0 = f"{metric_base}_hat_E0"
            out.loc[sub0.index, col_name0] = base0 + resid_hat0

        # E1
        res1 = results.get(key_E1)
        if res1 and res1["model"] is not None and res1["sub"] is not None:
            m1 = res1["model"]
            sub1 = res1["sub"]
            resid_hat1 = m1.fittedvalues
            base1 = out.loc[sub1.index, base_fitted_col]
            col_name1 = f"{metric_base}_hat_E1"
            out.loc[sub1.index, col_name1] = base1 + resid_hat1

    # 2. Fare: avg_base_passenger_fare_resid
    # add_residuals_for_metrics created:
    #   "avg_base_passenger_fare_fitted" and "avg_base_passenger_fare_resid"
    _add_level(
        metric_base="avg_base_passenger_fare",
        key_E0="E0_fare",
        key_E1="E1_fare",
        base_fitted_col="avg_base_passenger_fare_fitted",
        resid_col="avg_base_passenger_fare_resid",
    )

    # 3. Margin: margin_per_mile_resid
    # similarly: "margin_per_mile_fitted" and "margin_per_mile_resid"
    _add_level(
        metric_base="margin_per_mile",
        key_E0="E0_margin",
        key_E1="E1_margin",
        base_fitted_col="margin_per_mile_fitted",
        resid_col="margin_per_mile_resid",
    )

    return out


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    df_master = build_master_df(DATA_PATH)
    results = run_weather_monetization_models(df_master)
    df_with_preds = add_level_predictions(df_master, results)

    # example: filter to your rainy/cold case-study date 2025-03-20
    # df_day = df_with_preds[df_with_preds["date"] == "2025-03-20"]

    return df_with_preds, results


if __name__ == "__main__":
    df_with_preds, results = main()
    print("Weather monetization modeling + level predictions complete.")

    # --------------------------------------------------------------
    # 1. Print full summaries to console
    # --------------------------------------------------------------
    print("\nFULL MODEL SUMMARIES\n" + "-"*80)
    for key, bundle in results.items():
        model = bundle.get("model")
        if model is not None:
            print(f"\n--- {key} ---")
            print(model.summary())

    # --------------------------------------------------------------
    # 2. Save dataframe with predictions
    # --------------------------------------------------------------
    out_path = r"C:\Users\epicx\Projects\ubercasestudy\data\processed\fhvhv_lga_hourly_with_weather_2025_with_preds.parquet"
    df_with_preds.to_parquet(out_path, index=False)
    print(f"\nSaved dataframe with full predictions to:\n{out_path}")

    # --------------------------------------------------------------
    # 3. Save model summaries to text files
    # --------------------------------------------------------------
    summary_dir = r"C:\Users\epicx\Projects\ubercasestudy\reports\model_summaries"
    os.makedirs(summary_dir, exist_ok=True)

    for key, bundle in results.items():
        model = bundle.get("model")
        if model is None:
            continue
        summary_path = os.path.join(summary_dir, f"{key}.txt")
        with open(summary_path, "w") as f:
            f.write(model.summary().as_text())
        print(f"Saved summary for {key} → {summary_path}")

    print("\nAll model summaries and enriched dataframe saved.")
