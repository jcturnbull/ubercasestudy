# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 23:55:17 2025

@author: epicx
"""

from eda_plotting import run_day_vs_weekday_from_file

#demand
run_day_vs_weekday_from_file(
    file_name="fhvhv_lga_hourly_with_weather_2025_with_preds.parquet",
    target_date="2025-03-20",
    compare_dow1="Thursday",
    metric_col="request_count",
    est_col="demand_hat_weather",
    shade_hours=(21, 23),
    curfew=False,
    save_plots=False,
)
#avg fare E0
run_day_vs_weekday_from_file(
    file_name="fhvhv_lga_hourly_with_weather_2025_with_preds.parquet",
    target_date="2025-03-20",
    compare_dow1="Thursday",
    metric_col="avg_base_passenger_fare",
    est_col="avg_base_passenger_fare_hat_E0",
    shade_hours=(21, 23),
    curfew=False,
    save_plots=False,
)

#avg fare E1
run_day_vs_weekday_from_file(
    file_name="fhvhv_lga_hourly_with_weather_2025_with_preds.parquet",
    target_date="2025-03-20",
    compare_dow1="Thursday",
    metric_col="avg_base_passenger_fare",
    est_col="avg_base_passenger_fare_hat_E1",
    shade_hours=(21, 23),
    curfew=False,
    save_plots=False,
)

#margin E0
run_day_vs_weekday_from_file(
    file_name="fhvhv_lga_hourly_with_weather_2025_with_preds.parquet",
    target_date="2025-03-20",
    compare_dow1="Thursday",
    metric_col="margin_per_mile",
    est_col="margin_per_mile_hat_E0",
    shade_hours=(21, 23),
    curfew=False,
    save_plots=False,
)

#margin E1
run_day_vs_weekday_from_file(
    file_name="fhvhv_lga_hourly_with_weather_2025_with_preds.parquet",
    target_date="2025-03-20",
    compare_dow1="Thursday",
    metric_col="margin_per_mile",
    est_col="margin_per_mile_hat_E1",
    shade_hours=(21, 23),
    curfew=False,
    save_plots=False,
)


import pandas as pd

df = pd.read_parquet(
    r"C:\Users\epicx\Projects\ubercasestudy\data\processed\fhvhv_lga_hourly_with_weather_2025_with_preds.parquet"
)

# Ensure day_of_week exists
if "day_of_week" not in df.columns:
    df["day_of_week"] = df["datetime_hour"].dt.dayofweek  # Monday=0

# Filter the target day
day = pd.to_datetime("2025-03-20").date()
df_day = df[df["datetime_hour"].dt.date == day]

# -----------------------------
# 1. Total actual margin for the day
# -----------------------------
total_margin_actual = (df_day["trip_miles_sum"] * df_day["margin_per_mile"]).sum()

# -----------------------------
# 2. Total predicted margins for the day
# -----------------------------
total_margin_E0 = (df_day["trip_miles_sum"] * df_day["margin_per_mile_hat_E0"]).sum()
total_margin_E1 = (df_day["trip_miles_sum"] * df_day["margin_per_mile_hat_E1"]).sum()

# -----------------------------
# 3. Historical Thursday averages
# -----------------------------
df_thu = df[df["day_of_week"] == 3].copy()

# Hourly weighted margin contribution
df_thu["weighted_margin"] = df_thu["margin_per_mile"] * df_thu["trip_miles_sum"]

daily_thu = df_thu.groupby("date").agg(
    daily_trip_miles_sum=("trip_miles_sum", "sum"),
    daily_weighted_margin=("weighted_margin", "sum"),
)

historical_thursday_total_margin = daily_thu["daily_weighted_margin"].mean()

# -----------------------------
# 4. Output table
# -----------------------------
out = pd.DataFrame(
    {
        "total_margin_actual": [total_margin_actual],
        "total_margin_hat_E0": [total_margin_E0],
        "total_margin_hat_E1": [total_margin_E1],
        "historical_thursday_total_margin": [historical_thursday_total_margin],
    },
    index=[pd.to_datetime("2025-03-20")]
)

out


import pandas as pd
import numpy as np


def evaluate_models_daily_summary(
    df,
    actual_col,
    pred1_col,
    pred2_col,
    date_col="datetime_hour",
    target_date="2025-03-20",
    weight_col=None,  # if provided, treat metric as an average and weight by this col
):
    """
    Full-model summary and single-day summary for two prediction columns (pred1, pred2)
    versus an actual column.

    If weight_col is None:
        - Treat actual_col and preds as TOTAL-like metrics (e.g. margin_total, request_count).
        - 'actual'    : sum over rows
        - 'predicted' : sum over rows
        - 'error'     : total_pred - total_actual
        - 'pct error' : |error| / total_actual
        - MAE/MAPE/Bias/MSE: unweighted over rows

    If weight_col is provided:
        - Treat actual_col and preds as AVERAGE-like metrics (e.g. avg_base_passenger_fare).
        - 'actual'    : weighted mean over rows (sum(metric * w) / sum(w))
        - 'predicted' : weighted mean over rows
        - 'error'     : predicted - actual (in same units as metric)
        - 'pct error' : |error| / actual
        - MAE/MAPE/Bias/MSE: weighted over rows (weights = weight_col)
    """

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["date"] = df[date_col].dt.date

    # convenience columns
    df["actual"] = df[actual_col]
    df["E0_pred"] = df[pred1_col]
    df["E1_pred"] = df[pred2_col]

    def compute_for_slice(data):
        out = {}

        if weight_col is None:
            # -------- TOTAL-like metric --------
            total_actual = data["actual"].sum()
            for model in ["E0", "E1"]:
                pred = data[f"{model}_pred"]
                total_pred = pred.sum()
                total_error = total_pred - total_actual
                total_pct_error = (
                    abs(total_error) / total_actual if total_actual != 0 else np.nan
                )

                err = pred - data["actual"]
                abs_err = err.abs()
                pct_err = abs_err / data["actual"].replace(0, np.nan)

                MAE = abs_err.mean()
                MAPE = pct_err.mean()
                Bias = err.mean()
                MSE = (err**2).mean()

                out[model] = {
                    "total_actual": total_actual,
                    "total_pred": total_pred,
                    "total_error": total_error,
                    "total_pct_error": total_pct_error,
                    "MAE": MAE,
                    "MAPE": MAPE,
                    "Bias": Bias,
                    "MSE": MSE,
                }

        else:
            # -------- AVERAGE-like metric (weighted) --------
            w = data[weight_col].astype(float)
            total_w = w.sum()
            if total_w == 0:
                raise ValueError("Total weight is zero in this slice.")

            total_actual = (data["actual"] * w).sum() / total_w

            for model in ["E0", "E1"]:
                pred = data[f"{model}_pred"]
                pred_mean = (pred * w).sum() / total_w
                total_error = pred_mean - total_actual
                total_pct_error = (
                    abs(total_error) / total_actual if total_actual != 0 else np.nan
                )

                err = pred - data["actual"]
                abs_err = err.abs()
                pct_err = abs_err / data["actual"].replace(0, np.nan)

                # weighted error metrics
                MAE = (w * abs_err).sum() / total_w
                MAPE = (w * pct_err).sum() / total_w
                Bias = (w * err).sum() / total_w
                MSE = (w * (err**2)).sum() / total_w

                out[model] = {
                    "total_actual": total_actual,
                    "total_pred": pred_mean,
                    "total_error": total_error,
                    "total_pct_error": total_pct_error,
                    "MAE": MAE,
                    "MAPE": MAPE,
                    "Bias": Bias,
                    "MSE": MSE,
                }

        # Build the 8×3 table: statistic | E0 | E1
        summary_slice = pd.DataFrame(
            {
                "statistic": [
                    "actual",
                    "predicted",
                    "error",
                    "pct error",
                    "MAE",
                    "MAPE",
                    "Bias",
                    "MSE",
                ],
                "E0": [
                    out["E0"]["total_actual"],
                    out["E0"]["total_pred"],
                    out["E0"]["total_error"],
                    out["E0"]["total_pct_error"],
                    out["E0"]["MAE"],
                    out["E0"]["MAPE"],
                    out["E0"]["Bias"],
                    out["E0"]["MSE"],
                ],
                "E1": [
                    out["E1"]["total_actual"],
                    out["E1"]["total_pred"],
                    out["E1"]["total_error"],
                    out["E1"]["total_pct_error"],
                    out["E1"]["MAE"],
                    out["E1"]["MAPE"],
                    out["E1"]["Bias"],
                    out["E1"]["MSE"],
                ],
            }
        )

        return summary_slice

    # 1) Full sample
    summary = compute_for_slice(df)

    # 2) Single day
    tdate = pd.to_datetime(target_date).date()
    df_day = df[df["date"] == tdate]
    if df_day.empty:
        raise ValueError(f"No observations found for target_date={target_date}")

    daily = compute_for_slice(df_day)

    return summary, daily


def evaluate_single_model_summary(
    df,
    actual_col,
    pred_col,
    date_col="datetime_hour",
    target_date="2025-03-20",
):
    """
    One-model version of the summary:

    Returns
    -------
    summary : full-model summary over ALL rows in df
    daily   : same table, but restricted to target_date

    Both tables have shape:

        statistic | model

        'actual'    : total actual over slice
        'predicted' : total predicted over slice
        'error'     : total_pred - total_actual
        'pct error' : |total_error| / total_actual
        'MAE'       : mean absolute error over rows
        'MAPE'      : mean absolute percentage error over rows
        'Bias'      : mean(pred - actual)
        'MSE'       : mean squared error
    """

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["date"] = df[date_col].dt.date

    df["actual"] = df[actual_col]
    df["pred"] = df[pred_col]

    def compute_for_slice(data):
        total_actual = data["actual"].sum()
        total_pred = data["pred"].sum()
        total_error = total_pred - total_actual
        total_pct_error = (
            abs(total_error) / total_actual if total_actual != 0 else np.nan
        )

        err = data["pred"] - data["actual"]
        abs_err = err.abs()
        pct_err = abs_err / data["actual"].replace(0, np.nan)

        MAE = abs_err.mean()
        MAPE = pct_err.mean()
        Bias = err.mean()
        MSE = (err**2).mean()

        summary_slice = pd.DataFrame(
            {
                "statistic": [
                    "actual",
                    "predicted",
                    "error",
                    "pct error",
                    "MAE",
                    "MAPE",
                    "Bias",
                    "MSE",
                ],
                "model": [
                    total_actual,
                    total_pred,
                    total_error,
                    total_pct_error,
                    MAE,
                    MAPE,
                    Bias,
                    MSE,
                ],
            }
        )
        return summary_slice

    # full model
    summary = compute_for_slice(df)

    # target day
    tdate = pd.to_datetime(target_date).date()
    df_day = df[df["date"] == tdate]
    if df_day.empty:
        raise ValueError(f"No observations found for target_date={target_date}")

    daily = compute_for_slice(df_day)

    return summary, daily

def summarize_day_vs_avg_weekday(
    df,
    metric_col,
    date_col="datetime_hour",
    target_date="2025-03-20",
    weekday=3,  # 3 = Thursday (Mon=0)
):
    """
    Compare a specific day's total metric against the average total
    for a given weekday (default Thursday).

    Returns a DataFrame:

        statistic        | value
        -----------------|----------------
        day_total        | total metric on target_date
        avg_weekday      | average total metric for that weekday
        difference       | day_total - avg_weekday
        pct_difference   | |difference| / avg_weekday
    """

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["date"] = df[date_col].dt.date
    df["dow"] = df[date_col].dt.dayofweek  # Monday=0

    # Daily totals for the metric
    daily_totals = df.groupby("date")[metric_col].sum().reset_index(name="day_total")

    # Attach weekday info to daily frame
    daily_totals["dow"] = pd.to_datetime(daily_totals["date"]).dt.dayofweek

    # Target day total
    tdate = pd.to_datetime(target_date).date()
    row_day = daily_totals.loc[daily_totals["date"] == tdate]

    if row_day.empty:
        raise ValueError(f"No data for target_date={target_date}")

    day_total = float(row_day["day_total"].iloc[0])

    # Average weekday total (e.g. average Thursday)
    weekday_mask = daily_totals["dow"] == weekday
    avg_weekday = daily_totals.loc[weekday_mask, "day_total"].mean()

    diff = day_total - avg_weekday
    pct_diff = abs(diff) / avg_weekday if avg_weekday != 0 else np.nan

    summary = pd.DataFrame(
        {
            "statistic": ["day_total", "avg_weekday", "difference", "pct_difference"],
            "value": [day_total, avg_weekday, diff, pct_diff],
        }
    )

    return summary


def evaluate_day_vs_weekday_baseline(
    df,
    metric_col,
    date_col="datetime_hour",
    target_date="2025-03-20",
    weekday=3,      # 3 = Thursday (Mon=0)
    weight_col=None # if provided, metric_col is treated as a per-unit average
):
    """
    Compare a single day against a weekday baseline, treated as a 'model'.

    Baseline:
        For each hour of day, use the average metric for that hour across all
        days with day_of_week == weekday.

    If weight_col is None:
        metric_col is assumed to be a TOTAL-like metric (e.g. margin_total, request_count).
        - 'actual'    : sum of metric_col on target_date
        - 'predicted' : sum of baseline_hour on target_date
        - 'error'     : predicted - actual
        - 'pct error' : |error| / actual
        - 'MAE'/'MAPE'/'Bias'/'MSE' based on hourly errors (unweighted)

    If weight_col is provided:
        metric_col is assumed to be an AVERAGE-like metric (e.g. avg fare per trip).
        - 'actual'    : weighted daily average on target_date
                        sum(metric_hour * weight_hour) / sum(weight_hour)
        - 'predicted' : weighted daily average baseline using the SAME weights
                        sum(baseline_hour * weight_hour) / sum(weight_hour)
        - 'error'     : predicted - actual (in the same units as metric_col)
        - 'pct error' : |error| / actual
        - 'MAE'/'MAPE'/'Bias'/'MSE' based on hourly errors, weighted by weight_col
    """

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["date"] = df[date_col].dt.date
    df["hour"] = df[date_col].dt.hour
    df["dow"] = df[date_col].dt.dayofweek  # Monday=0

    # 1. Build hourly weekday baseline (e.g., average Thursday by hour)
    df_weekday = df[df["dow"] == weekday]

    hourly_baseline = (
        df_weekday.groupby("hour")[metric_col]
        .mean()
        .rename("baseline_hour")
    )

    # 2. Extract target day and attach baseline
    tdate = pd.to_datetime(target_date).date()
    df_day = df[df["date"] == tdate].copy()
    if df_day.empty:
        raise ValueError(f"No data for target_date={target_date}")

    df_day = df_day.merge(
        hourly_baseline,
        how="left",
        left_on="hour",
        right_index=True,
    )

    df_day = df_day.dropna(subset=["baseline_hour", metric_col])

    actual_hour = df_day[metric_col]
    pred_hour = df_day["baseline_hour"]

    if weight_col is None:
        # -------- TOTAL-like metric (margin_total, request_count) --------
        total_actual = actual_hour.sum()
        total_pred = pred_hour.sum()
        total_error = total_pred - total_actual
        total_pct_error = (
            abs(total_error) / total_actual if total_actual != 0 else np.nan
        )

        err = pred_hour - actual_hour
        abs_err = err.abs()
        pct_err = abs_err / actual_hour.replace(0, np.nan)

        MAE = abs_err.mean()
        MAPE = pct_err.mean()
        Bias = err.mean()
        MSE = (err**2).mean()

    else:
        # -------- AVERAGE-like metric (avg_base_passenger_fare, etc.) --------
        w = df_day[weight_col].astype(float)
        total_w = w.sum()
        if total_w == 0:
            raise ValueError("Total weight is zero for the target day.")

        # Weighted daily averages
        total_actual = (actual_hour * w).sum() / total_w
        total_pred = (pred_hour * w).sum() / total_w
        total_error = total_pred - total_actual
        total_pct_error = (
            abs(total_error) / total_actual if total_actual != 0 else np.nan
        )

        err = pred_hour - actual_hour
        abs_err = err.abs()
        pct_err = abs_err / actual_hour.replace(0, np.nan)

        # Weighted error metrics
        MAE = (w * abs_err).sum() / total_w
        MAPE = (w * pct_err).sum() / total_w
        Bias = (w * err).sum() / total_w
        MSE = (w * (err**2)).sum() / total_w

    summary = pd.DataFrame(
        {
            "statistic": [
                "actual",
                "predicted",
                "error",
                "pct error",
                "MAE",
                "MAPE",
                "Bias",
                "MSE",
            ],
            "baseline": [
                total_actual,
                total_pred,
                total_error,
                total_pct_error,
                MAE,
                MAPE,
                Bias,
                MSE,
            ],
        }
    )

    return summary

# df = pd.read_parquet(
#     r"C:\Users\epicx\Projects\ubercasestudy\data\processed\fhvhv_lga_hourly_with_weather_2025_with_preds.parquet"
# )

df_valid = df[df["wind_chill_f"].notna()].copy()

df_valid["margin_total"] = df_valid["margin_per_mile"] * df_valid["trip_miles_sum"]
df_valid["margin_total_E0"] = df_valid["margin_per_mile_hat_E0"] * df_valid["trip_miles_sum"]
df_valid["margin_total_E1"] = df_valid["margin_per_mile_hat_E1"] * df_valid["trip_miles_sum"]

summary_margin, daily_margin = evaluate_models_daily_summary(
    df_valid,
    actual_col="margin_total",
    pred1_col="margin_total_E0",
    pred2_col="margin_total_E1",
    date_col="datetime_hour",
    target_date="2025-03-20",
    weight_col=None,  # total metric
)
# formatted, march20 = evaluate_metric_formatted(
#     df_valid,
#     actual_col="margin_total",
#     pred1_col="margin_total_E0",
#     pred2_col="margin_total_E1",
#     group_daily=True,
#     target_date="2025-03-20"
# )

# results, march20 = evaluate_metric(
#     df_valid,
#     actual_col="margin_total",
#     pred1_col="margin_total_E0",
#     pred2_col="margin_total_E1",
#     group_daily=True,
#     target_date="2025-03-20"
# )

print("Model metrics:")
print(summary_margin)
print("\nMetrics for March 20th:")
print(daily_margin)

summary_demand, daily_demand = evaluate_single_model_summary(
    df_valid,
    actual_col="request_count",
    pred_col="demand_hat_weather",
    date_col="datetime_hour",
    target_date="2025-03-20",
)
print("Model metrics:")
print(summary_demand)
print("\nMetrics for March 20th:")
print(daily_demand)


summary_fare, daily_fare = evaluate_models_daily_summary(
    df_valid,
    actual_col="avg_base_passenger_fare",
    pred1_col="avg_base_passenger_fare_hat_E0",
    pred2_col="avg_base_passenger_fare_hat_E1",
    date_col="datetime_hour",
    target_date="2025-03-20",
    weight_col="request_count",  # weight by trips
)
print("Model metrics:")
print(summary_fare)
print("\nMetrics for March 20th:")
print(daily_fare)

# summary_thu_margin = summarize_day_vs_avg_weekday(
#     df_valid,
#     metric_col="margin_total",
#     date_col="datetime_hour",
#     target_date="2025-03-20",
#     weekday=3,  # Thursday
# )
summary_thu_margin = evaluate_day_vs_weekday_baseline(
    df_valid,
    metric_col="margin_total",
    date_col="datetime_hour",
    target_date="2025-03-20",
    weekday=3,         # Thursday
    weight_col=None    # total metric
)

print(summary_thu_margin)

summary_thu_fare = evaluate_day_vs_weekday_baseline(
    df_valid,
    metric_col="avg_base_passenger_fare",
    date_col="datetime_hour",
    target_date="2025-03-20",
    weekday=3,                 # Thursday
    weight_col="request_count" # weight by number of trips
)

print(summary_thu_fare)

summary_thu_demand = evaluate_day_vs_weekday_baseline(
    df_valid,
    metric_col="request_count",
    date_col="datetime_hour",
    target_date="2025-03-20",
    weekday=3,                 # Thursday
    weight_col="request_count" # weight by number of trips
)
print(summary_thu_demand)


# Demand baseline (already have this, just restating)
dow_hour_demand = (
    df_valid
    .groupby(["day_of_week", "hour"])["request_count"]
    .mean()
    .rename("demand_hat_average")
    .reset_index()
)

# Margin baseline: average hourly margin_total by DOW + hour
dow_hour_margin = (
    df_valid
    .groupby(["day_of_week", "hour"])["margin_total"]
    .mean()
    .rename("margin_hat_average")
    .reset_index()
)

# Avg fare baseline: trip-weighted average by DOW + hour
dow_hour_fare = (
    df_valid
    .assign(weighted_fare=lambda d: d["avg_base_passenger_fare"] * d["request_count"])
    .groupby(["day_of_week", "hour"])
    .agg(
        total_weighted_fare=("weighted_fare", "sum"),
        total_trips=("request_count", "sum"),
    )
    .assign(avg_fare_hat_average=lambda g: g["total_weighted_fare"] / g["total_trips"])
    [["avg_fare_hat_average"]]
    .reset_index()
)

# 2. Merge back into df_valid on BOTH day_of_week and hour
df_valid = df_valid.merge(dow_hour_demand, on=["day_of_week", "hour"], how="left")


# Filter to 2025-03-20
mask_320 = df_valid["datetime_hour"].dt.date == pd.to_datetime("2025-03-20").date()
df_320 = df_valid.loc[mask_320].copy()

# Merge all three baselines on (day_of_week, hour)
df_320 = (
    df_320
    .merge(dow_hour_demand, on=["day_of_week", "hour"], how="left")
    .merge(dow_hour_margin, on=["day_of_week", "hour"], how="left")
    .merge(dow_hour_fare,   on=["day_of_week", "hour"], how="left")
)
df_320_20_23 = df_320[df_320["hour"].between(20, 23)]

df_320_20_23[
    [
        "datetime_hour",
        "request_count", "demand_hat_weather", "demand_hat_average",
        "margin_total", "margin_hat_average",
        "avg_base_passenger_fare", "avg_fare_hat_average",
    ]
]











import pandas as pd
import matplotlib.pyplot as plt

def plot_weather_day(df, target_date, dt_col="datetime_hour"):
    """
    Plot hourly weather metrics for a given day with full-height shading bands for:
        - rain_flag
        - heavy_rain_flag

    Wind chill is on the primary axis.
    Precip amount is on the secondary axis.
    """

    df = df.copy()
    df[dt_col] = pd.to_datetime(df[dt_col])

    day = pd.to_datetime(target_date).date()
    df_day = df[df[dt_col].dt.date == day].sort_values(dt_col)

    if df_day.empty:
        raise ValueError(f"No data found for {target_date}")

    x = df_day[dt_col]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    # Primary axis grid (aligned across both axes)
    ax1.grid(True, linestyle="-", alpha=0.25)
    
    # --- Colors and styles ---
    color_windchill = "#66c2ff"      # icy blue
    color_rain_flag = "0.85"         # light gray for shading
    color_heavy_rain = "cornflowerblue"
    color_precip = "navy"

    # ================================================================
    # 1. SHADE THE BACKGROUND FOR RAIN + HEAVY RAIN HOURS
    # ================================================================
    for idx, row in df_day.iterrows():
        hour_start = row[dt_col]
        hour_end = hour_start + pd.Timedelta(hours=1)

        # Heavy rain shading (dominant)
        if row["heavy_rain_flag"] == 1:
            ax1.axvspan(hour_start, hour_end, color=color_heavy_rain, alpha=0.20)

        # Lighter shading for rain_flag (but only if not heavy rain)
        elif row["rain_flag"] == 1:
            ax1.axvspan(hour_start, hour_end, color=color_rain_flag, alpha=0.40)

    # ================================================================
    # 2. WIND CHILL — line on primary y-axis
    # ================================================================
    ax1.plot(
        x,
        df_day["wind_chill_f"],
        label="Wind Chill (°F)",
        color=color_windchill,
        linewidth=2.2,
    )

    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Wind Chill (°F)")

    # ================================================================
    # 3. PRECIP AMOUNT — dashed line on secondary axis
    # ================================================================
    ax2 = ax1.twinx()
    ax2.grid(False)
    ax2.plot(
        x,
        df_day["precip_1h_mm_total"],
        label="Precipitation (mm)",
        color=color_precip,
        linestyle="--",
        linewidth=2.2,
    )
    ax2.set_ylabel("Precipitation (mm)")

    # ================================================================
    # TITLE + LEGEND
    # ================================================================
    ax1.set_title(f"Weather Effects on {day}")

    # Build combined legend
    lines1, labels1 = ax1.get_lines(), [l.get_label() for l in ax1.get_lines()]
    lines2, labels2 = ax2.get_lines(), [l.get_label() for l in ax2.get_lines()]
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    return fig, ax1, ax2

fig, ax1, ax2 = plot_weather_day(df, "2025-03-20")
