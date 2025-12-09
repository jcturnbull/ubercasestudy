# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 19:25:52 2025

Baseline demand residualization and contemporaneous weather models.

@author: epicx
"""

import os
import pandas as pd
import statsmodels.api as sm
import patsy
import matplotlib.pyplot as plt

# import custom utilities
from model_utils import coeff_table, sig_code, add_to_summary, clean_sheet_name

BASELINE_XLSX = (
    r"C:\Users\epicx\Projects\ubercasestudy\reports\baseline_weather_models_results.xlsx"
)


# ------------------------------------------
# Calendar feature builder
# ------------------------------------------
def add_calendar_features(df: pd.DataFrame, time_col: str):
    out = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(out[time_col]):
        out[time_col] = pd.to_datetime(out[time_col])

    out["day_of_week"] = out[time_col].dt.dayofweek
    out["hour"] = out[time_col].dt.hour
    out["hour_of_week"] = out["day_of_week"] * 24 + out["hour"]
    out["month"] = out[time_col].dt.month

    return out


# ------------------------------------------
# Baseline FE model + residual extraction
# ------------------------------------------
def fit_baseline_demand_model(df, dep_var, time_col, include_month_dummies=True):
    df_feat = add_calendar_features(df, time_col)

    formula = f"{dep_var} ~ C(hour_of_week)"
    if include_month_dummies:
        formula += " + C(month)"

    y, X = patsy.dmatrices(formula, data=df_feat, return_type="dataframe")

    valid_idx = y.index.intersection(X.index)
    y = y.loc[valid_idx]
    X = X.loc[valid_idx]
    df_feat = df_feat.loc[valid_idx]

    model = sm.OLS(y, X).fit()

    df_feat["demand_fitted"] = model.fittedvalues
    df_feat["demand_resid"] = model.resid

    return df_feat, model


# ------------------------------------------
# Weather differentials + precip indicators
# ------------------------------------------
def add_weather_differentials(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - wind_chill_diff = wind_chill_f - temp_f_mean
      - heat_index_diff = heat_index_f - temp_f_mean
      - rain_flag       = 1 if precip_1h_mm_total > 0
      - heavy_rain_flag = 1 if precip_1h_mm_total >= 5.0 mm
    """
    out = df.copy()

    out["wind_chill_diff"] = out["wind_chill_f"] - out["temp_f_mean"]
    out["heat_index_diff"] = out["heat_index_f"] - out["temp_f_mean"]

    out["rain_flag"] = (out["precip_1h_mm_total"] > 0).astype(int)
    out["heavy_rain_flag"] = (out["precip_1h_mm_total"] >= 5.0).astype(int)

    return out


def run_univariate_regression(df: pd.DataFrame, y_col: str, x_col: str, label: str):
    """
    Run OLS: y_col ~ x_col (with intercept) on rows where both y and x are not null.
    Prints a compact summary and returns (model, df_used).
    """
    sub = df[[y_col, x_col]].dropna()
    if sub.empty:
        print(f"[{label}] No data available (all NA).")
        return None, None

    X = sm.add_constant(sub[x_col])
    y = sub[y_col]

    model = sm.OLS(y, X).fit()

    print("=" * 80)
    print(f"Regression: {y_col} ~ {x_col}   ({label})")
    print(f"Observations used: {len(sub)}")
    print(model.summary().tables[1])  # coefficients table only
    print("=" * 80)

    return model, sub


def run_quadratic_model(df, y_col, x_col, label):
    """
    Simple quadratic check: y ~ x + x^2
    Returns fitted model or None.
    """
    sub = df[[y_col, x_col]].dropna()
    if sub.empty:
        print(f"[{label}] No data available (all NA).")
        return None

    x_sq_col = f"{x_col}_sq"
    sub = sub.assign(**{x_sq_col: sub[x_col] ** 2})
    X = sm.add_constant(sub[[x_col, x_sq_col]])
    y = sub[y_col]

    model = sm.OLS(y, X).fit()

    print("=" * 90)
    print(f"Quadratic Model: {y_col} ~ {x_col} + {x_col}^2   ({label})")
    print(f"Observations used: {len(sub)}")
    print(model.summary().tables[1])
    print("=" * 90)

    return model


# ------------------------------------------
# Main execution block
# ------------------------------------------
if __name__ == "__main__":

    # containers for Excel summary
    summary_rows = []
    model_dict = {}

    # ------------------------------------------------------------------
    # 1. Load merged hourly LGA dataset (TLC + weather)
    # ------------------------------------------------------------------
    PATH = (
        r"C:\Users\epicx\Projects\ubercasestudy\data\processed\fhvhv_lga_hourly_with_weather_2025.parquet"
    )
    df = pd.read_parquet(PATH)

    # ------------------------------------------------------------------
    # 2. Baseline seasonality model: remove hour-of-week + month effects
    # ------------------------------------------------------------------
    df_resid, baseline_model = fit_baseline_demand_model(
        df,
        dep_var="request_count",
        time_col="datetime_hour",
        include_month_dummies=True,
    )

    print("Baseline seasonality-only model (request_count ~ hour_of_week + month):")
    print(baseline_model.summary().tables[1])

    # add baseline model to Excel summary
    add_to_summary(
        summary_rows,
        baseline_model,
        dep_var="request_count",
        label="Baseline: request_count ~ C(hour_of_week) + C(month)",
    )
    model_dict["Baseline"] = baseline_model

    # ------------------------------------------------------------------
    # 3. Add weather differential features + precip flags
    # ------------------------------------------------------------------
    df_weather = add_weather_differentials(df_resid)

    print("\nNon-null counts for weather differentials:")
    print(
        df_weather[
            ["wind_chill_diff", "heat_index_diff", "precip_1h_mm_total"]
        ].notna().sum()
    )

    # ------------------------------------------------------------------
    # 4. Univariate regressions: residual demand vs weather variables
    # ------------------------------------------------------------------
    # 4.1 Wind chill differential
    wind_chill_model, wind_chill_data = run_univariate_regression(
        df_weather,
        y_col="demand_resid",
        x_col="wind_chill_diff",
        label="Wind chill - temp diff",
    )
    if wind_chill_model is not None:
        add_to_summary(
            summary_rows,
            wind_chill_model,
            dep_var="demand_resid",
            label="demand_resid ~ wind_chill_diff",
        )
        model_dict["demand_resid ~ wind_chill_diff"] = wind_chill_model

    # 4.2 Heat index differential
    heat_index_model, heat_index_data = run_univariate_regression(
        df_weather,
        y_col="demand_resid",
        x_col="heat_index_diff",
        label="Heat index - temp diff",
    )
    if heat_index_model is not None:
        add_to_summary(
            summary_rows,
            heat_index_model,
            dep_var="demand_resid",
            label="demand_resid ~ heat_index_diff",
        )
        model_dict["demand_resid ~ heat_index_diff"] = heat_index_model

    # 4.3 Precipitation amount (mm)
    precip_amt_model, precip_amt_data = run_univariate_regression(
        df_weather,
        y_col="demand_resid",
        x_col="precip_1h_mm_total",
        label="Precipitation amount (mm)",
    )
    if precip_amt_model is not None:
        add_to_summary(
            summary_rows,
            precip_amt_model,
            dep_var="demand_resid",
            label="demand_resid ~ precip_1h_mm_total",
        )
        model_dict["demand_resid ~ precip_1h_mm_total"] = precip_amt_model

    # 4.4 Rain vs no rain (binary)
    rain_flag_model, rain_flag_data = run_univariate_regression(
        df_weather,
        y_col="demand_resid",
        x_col="rain_flag",
        label="Rain flag (any rain vs none)",
    )
    if rain_flag_model is not None:
        add_to_summary(
            summary_rows,
            rain_flag_model,
            dep_var="demand_resid",
            label="demand_resid ~ rain_flag",
        )
        model_dict["demand_resid ~ rain_flag"] = rain_flag_model

    # 4.5 Heavy rain flag
    heavy_rain_model, heavy_rain_data = run_univariate_regression(
        df_weather,
        y_col="demand_resid",
        x_col="heavy_rain_flag",
        label="Heavy rain flag (>= 5mm/hr)",
    )
    if heavy_rain_model is not None:
        add_to_summary(
            summary_rows,
            heavy_rain_model,
            dep_var="demand_resid",
            label="demand_resid ~ heavy_rain_flag",
        )
        model_dict["demand_resid ~ heavy_rain_flag"] = heavy_rain_model
    
    # 4.6 Raw wind chill (F)
    raw_wind_chill_model, raw_wind_chill_data = run_univariate_regression(
        df_weather,
        y_col="demand_resid",
        x_col="wind_chill_f",
        label="Raw wind chill (F)",
    )
    if raw_wind_chill_model is not None:
        add_to_summary(
            summary_rows,
            raw_wind_chill_model,
            dep_var="demand_resid",
            label="demand_resid ~ wind_chill_f",
        )
        model_dict["demand_resid ~ wind_chill_f"] = raw_wind_chill_model

    # 4.7 Raw heat index (F)
    raw_heat_index_model, raw_heat_index_data = run_univariate_regression(
        df_weather,
        y_col="demand_resid",
        x_col="heat_index_f",
        label="Raw heat index (F)",
    )
    if raw_heat_index_model is not None:
        add_to_summary(
            summary_rows,
            raw_heat_index_model,
            dep_var="demand_resid",
            label="demand_resid ~ heat_index_f",
        )
        model_dict["demand_resid ~ heat_index_f"] = raw_heat_index_model

    # ------------------------------------------------------------------
    # 5. Diagnostics plots (unchanged)
    # ------------------------------------------------------------------
    if wind_chill_data is not None and not wind_chill_data.empty:
        plt.figure(figsize=(8, 5))
        plt.scatter(
            wind_chill_data["wind_chill_diff"],
            wind_chill_data["demand_resid"],
            alpha=0.4,
        )
        plt.axhline(0, linestyle="--")
        plt.xlabel("Wind chill - Temp (F)")
        plt.ylabel("Demand residual")
        plt.title("Residual demand vs wind chill differential (winter hours)")
        plt.tight_layout()
        plt.show()

    if heat_index_data is not None and not heat_index_data.empty:
        plt.figure(figsize=(8, 5))
        plt.scatter(
            heat_index_data["heat_index_diff"],
            heat_index_data["demand_resid"],
            alpha=0.4,
        )
        plt.axhline(0, linestyle="--")
        plt.xlabel("Heat index - Temp (F)")
        plt.ylabel("Demand residual")
        plt.title("Residual demand vs heat index differential (summer hours)")
        plt.tight_layout()
        plt.show()

    if precip_amt_data is not None and not precip_amt_data.empty:
        plt.figure(figsize=(8, 5))
        plt.scatter(
            precip_amt_data["precip_1h_mm_total"],
            precip_amt_data["demand_resid"],
            alpha=0.4,
        )
        plt.axhline(0, linestyle="--")
        plt.xlabel("Precipitation (mm)")
        plt.ylabel("Demand residual")
        plt.title("Residual demand vs precipitation amount")
        plt.tight_layout()
        plt.show()
    
    if raw_wind_chill_data is not None and not raw_wind_chill_data.empty:
        plt.figure(figsize=(8, 5))
        plt.scatter(
            raw_wind_chill_data["wind_chill_f"],
            raw_wind_chill_data["demand_resid"],
            alpha=0.4,
        )
        plt.axhline(0, linestyle="--")
        plt.xlabel("Wind chill (F)")
        plt.ylabel("Demand residual")
        plt.title("Residual demand vs raw wind chill (F)")
        plt.tight_layout()
        plt.show()

    if raw_heat_index_data is not None and not raw_heat_index_data.empty:
        plt.figure(figsize=(8, 5))
        plt.scatter(
            raw_heat_index_data["heat_index_f"],
            raw_heat_index_data["demand_resid"],
            alpha=0.4,
        )
        plt.axhline(0, linestyle="--")
        plt.xlabel("Heat index (F)")
        plt.ylabel("Demand residual")
        plt.title("Residual demand vs raw heat index (F)")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # 6. Quadratic relationships (optional; still captured in Excel)
    # ------------------------------------------------------------------
    precip_quad_model = run_quadratic_model(
        df_weather,
        y_col="demand_resid",
        x_col="precip_1h_mm_total",
        label="Precipitation amount (mm)",
    )
    if precip_quad_model is not None:
        add_to_summary(
            summary_rows,
            precip_quad_model,
            dep_var="demand_resid",
            label="Quadratic: demand_resid ~ precip + precip^2",
        )
        model_dict["Quadratic: demand_resid ~ precip"] = precip_quad_model

    wind_chill_quad_model = run_quadratic_model(
        df_weather,
        y_col="demand_resid",
        x_col="wind_chill_diff",
        label="Wind chill - temp differential",
    )
    if wind_chill_quad_model is not None:
        add_to_summary(
            summary_rows,
            wind_chill_quad_model,
            dep_var="demand_resid",
            label="Quadratic: demand_resid ~ wind_chill_diff + wind_chill_diff^2",
        )
        model_dict[
            "Quadratic: demand_resid ~ wind_chill_diff"
        ] = wind_chill_quad_model

    heat_index_quad_model = run_quadratic_model(
        df_weather,
        y_col="demand_resid",
        x_col="heat_index_diff",
        label="Heat index - temp differential",
    )
    if heat_index_quad_model is not None:
        add_to_summary(
            summary_rows,
            heat_index_quad_model,
            dep_var="demand_resid",
            label="Quadratic: demand_resid ~ heat_index_diff + heat_index_diff^2",
        )
        model_dict[
            "Quadratic: demand_resid ~ heat_index_diff"
        ] = heat_index_quad_model

    # ------------------------------------------------------------------
    # 7. Write Excel summary + per-model sheets
    # ------------------------------------------------------------------
    summary_df = pd.DataFrame(summary_rows)

    out_dir = os.path.dirname(BASELINE_XLSX)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with pd.ExcelWriter(BASELINE_XLSX, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        for label, model in model_dict.items():
            sheet = clean_sheet_name(label)
            coef_df = coeff_table(model, drop_const=False)
            coef_df.to_excel(writer, sheet_name=sheet, index=False)

    print(f"\nBaseline models written to: {BASELINE_XLSX}")
