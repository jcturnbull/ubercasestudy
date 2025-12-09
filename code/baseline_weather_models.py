# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 19:25:52 2025

@author: epicx
"""

import pandas as pd
import statsmodels.api as sm
import patsy
import matplotlib.pyplot as plt


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
      - wind_chill_diff = wind_chill_f - temp_f_mean (where both present)
      - heat_index_diff = heat_index_f - temp_f_mean (where both present)
      - rain_flag       = 1 if precip_1h_mm_total > 0, else 0
      - heavy_rain_flag = 1 if precip_1h_mm_total >= 5.0 mm, else 0 (tunable)
    """
    out = df.copy()

    # Start as NaN; only fill where both components defined
    out["wind_chill_diff"] = out["wind_chill_f"] - out["temp_f_mean"]
    out["heat_index_diff"] = out["heat_index_f"] - out["temp_f_mean"]

    # Precip flags
    out["rain_flag"] = (out["precip_1h_mm_total"] > 0).astype(int)

    # Threshold for "heavy" can be tuned; 5mm/hr is a reasonable starting point
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

    # ------------------------------------------------------------------
    # 1. Load merged hourly LGA dataset (TLC + weather)
    # ------------------------------------------------------------------
    PATH = r"C:\Users\epicx\Projects\ubercasestudy\data\processed\fhvhv_lga_hourly_with_weather_2025.parquet"
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
    #df_resid.to_parquet(r"C:\Users\epicx\Projects\ubercasestudy\data\processed\fhvhv_lga_resid_demand_2025.parquet")
    
    print("Baseline seasonality-only model (request_count ~ hour_of_week + month):")
    print(baseline_model.summary().tables[1])

    # demand_resid is your "raw residual demand after removing seasonality"
    # This is the Y for all subsequent weather analysis.
    # df_resid has all original columns + demand_resid, demand_fitted, calendar features.

    # ------------------------------------------------------------------
    # 3. Add weather differential features + precip flags
    # ------------------------------------------------------------------
    df_weather = add_weather_differentials(df_resid)

    # Quick sanity on missingness
    print("\nNon-null counts for weather differentials:")
    print(df_weather[["wind_chill_diff", "heat_index_diff", "precip_1h_mm_total"]].notna().sum())

    # ------------------------------------------------------------------
    # 4. Univariate regressions: residual demand vs weather variables
    # ------------------------------------------------------------------
    # 4.1 Wind chill differential (winter subset by construction)
    wind_chill_model, wind_chill_data = run_univariate_regression(
        df_weather, y_col="demand_resid", x_col="wind_chill_diff", label="Wind chill - temp diff"
    )

    # 4.2 Heat index differential (summer subset by construction)
    heat_index_model, heat_index_data = run_univariate_regression(
        df_weather, y_col="demand_resid", x_col="heat_index_diff", label="Heat index - temp diff"
    )

    # 4.3 Precipitation amount (mm)
    precip_amt_model, precip_amt_data = run_univariate_regression(
        df_weather, y_col="demand_resid", x_col="precip_1h_mm_total", label="Precipitation amount (mm)"
    )

    # 4.4 Rain vs no rain (binary)
    rain_flag_model, rain_flag_data = run_univariate_regression(
        df_weather, y_col="demand_resid", x_col="rain_flag", label="Rain flag (any rain vs none)"
    )

    # Optional: heavy rain flag; sample size may be small, so treat as exploratory
    heavy_rain_model, heavy_rain_data = run_univariate_regression(
        df_weather, y_col="demand_resid", x_col="heavy_rain_flag", label="Heavy rain flag (>= 5mm/hr)"
    )

    # ------------------------------------------------------------------
    # 5. Basic scatterplots for diagnostics
    # ------------------------------------------------------------------
    # Note: these are simple visuals; they don't add new modeling, just intuition.

    # 5.1 Wind chill diff vs demand_resid
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

    # 5.2 Heat index diff vs demand_resid
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

    # 5.3 Precipitation amount vs demand_resid
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
    
    # ------------------------------------------------------------------
    # 6. Quadratic relationships: demand_resid vs key weather variables
    #    - Precipitation amount (mm)
    #    - Wind chill differential
    #    - Heat index differential
    # ------------------------------------------------------------------
    
    # 6.1 Quadratic: demand_resid vs precipitation amount
    precip_quad_model = run_quadratic_model(
        df_weather,
        y_col="demand_resid",
        x_col="precip_1h_mm_total",
        label="Precipitation amount (mm)"
    )

    # 6.2 Quadratic: demand_resid vs wind chill differential
    wind_chill_quad_model = run_quadratic_model(
        df_weather,
        y_col="demand_resid",
        x_col="wind_chill_diff",
        label="Wind chill - temp differential"
    )

    # 6.3 Quadratic: demand_resid vs heat index differential
    heat_index_quad_model = run_quadratic_model(
        df_weather,
        y_col="demand_resid",
        x_col="heat_index_diff",
        label="Heat index - temp differential"
    )

