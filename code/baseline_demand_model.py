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
# Main execution block
# ------------------------------------------
if __name__ == "__main__":

    # 1. Load merged hourly LGA dataset (TLC + weather)
    PATH = r"C:\Users\epicx\Projects\ubercasestudy\data\processed\fhvhv_lga_hourly_with_weather_2025.parquet"
    df = pd.read_parquet(PATH)

    # 2. Fit baseline FE demand model
    df_resid, model = fit_baseline_demand_model(
        df,
        dep_var="request_count",
        time_col="datetime_hour",
        include_month_dummies=True
    )

    print(model.summary())

    # 3. Scatterplot: residual demand vs temperature
    plt.figure(figsize=(10,5))
    plt.scatter(df_resid["temp_f_mean"], df_resid["demand_resid"], alpha=0.3)
    plt.xlabel("Temperature (F) — temp_f_mean")
    plt.ylabel("Demand Residual (after removing seasonality)")
    plt.title("Residual Demand vs Temperature")
    plt.tight_layout()
    plt.show()

