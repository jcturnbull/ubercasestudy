# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 20:30:50 2025

@author: epicx

tlc_aggregate_hourly.py

"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Dict, List

import numpy as np
import pandas as pd

from config import INTERIM_TLC_DIR, PROCESSED_TLC_DIR


# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

DEFAULT_INPUT = INTERIM_TLC_DIR / "fhvhv_tripdata_2025_all_clean.parquet"
DEFAULT_OUTPUT = PROCESSED_TLC_DIR / "fhvhv_lga_hourly_agg_2025.parquet"

# In the combined data this is the wait-time column (seconds)
DEFAULT_WAIT_TIME_COL = "pickup_wait_time_sec"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_datetime(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Coerce listed columns to datetime if present."""
    df = df.copy()
    for col in cols:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def add_fare_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add fare / earnings derived columns:

        subtotal_fare        = base_fare + tolls + bcf + sales_tax
                               + congestion_surcharge + airport_fee
                               + cbd_congestion_fee (where present)

        rider_total          = subtotal_fare + tips

        driver_pay_with_tips = driver_pay + tips
    """
    df = df.copy()

    fee_cols_all = [
        "base_passenger_fare",
        "tolls",
        "bcf",
        "sales_tax",
        "congestion_surcharge",
        "airport_fee",
        "cbd_congestion_fee",
    ]
    fee_cols = [c for c in fee_cols_all if c in df.columns]

    # Ensure money columns are numeric, with 0.0 for missing
    numeric_cols = fee_cols + ["tips", "driver_pay"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Subtotal: all fare / tax / fee components, excluding tips
    if fee_cols:
        df["subtotal_fare"] = df[fee_cols].sum(axis=1)
    else:
        df["subtotal_fare"] = 0.0

    # Rider total: subtotal + tips
    if "tips" in df.columns:
        df["rider_total"] = df["subtotal_fare"] + df["tips"]
    else:
        df["rider_total"] = df["subtotal_fare"]

    # Driver pay including tips
    if "driver_pay" in df.columns:
        df["driver_pay_with_tips"] = df["driver_pay"] + df.get("tips", 0.0)
    else:
        df["driver_pay_with_tips"] = df.get("tips", 0.0)

    return df


def aggregate_hourly(
    df: pd.DataFrame,
    wait_time_col: str = DEFAULT_WAIT_TIME_COL,
) -> pd.DataFrame:
    """
    Aggregate cleaned HVFHV LGA data to hourly metrics keyed by request hour.

    Output columns (per row):

      Keys:
        - datetime_hour: datetime64[ns] (request datetime floored to hour)
        - date: date
        - request_hour: int 0–23 (hour of day from request time)

      Counts:
        - request_count
        - pickup_count
        - dropoff_count

      Geometry / time:
        - trip_miles_sum, trip_miles_mean, trip_miles_median
        - trip_time_sum, trip_time_mean, trip_time_median
        - avg_speed_mph, median_speed_mph

      Fares (rider side):
        - base_passenger_fare_sum, avg_base_passenger_fare
        - subtotal_fare_sum, avg_subtotal_fare
        - rider_total_sum, avg_rider_total
        - tips_sum, avg_tips

      Driver economics:
        - driver_pay_sum, avg_driver_pay
        - driver_pay_with_tips_sum, avg_driver_pay_with_tips
        - driver_pay_pct_of_base_fare (driver_pay_sum / base_passenger_fare_sum)

      Wait time (from pickup_wait_time_sec, if present):
        - wait_time_sec_mean
        - wait_time_sec_median
    """
    df = df.copy()

    # -----------------------------------------------------------------------
    # Filter to LGA pickups
    # -----------------------------------------------------------------------
    if "is_lga_pickup" in df.columns:
        df = df[df["is_lga_pickup"].astype(bool)]
    else:
        raise KeyError("Expected column 'is_lga_pickup' not found in DataFrame.")

    if df.empty:
        raise ValueError("No rows after filtering to is_lga_pickup == 1.")

    # -----------------------------------------------------------------------
    # Time columns / hourly buckets
    # -----------------------------------------------------------------------
    df = ensure_datetime(
        df,
        ["request_datetime", "pickup_datetime", "dropoff_datetime"],
    )

    # Request is the primary hour key
    df["request_hour_dt"] = df["request_datetime"].dt.floor("h")
    df["pickup_hour_dt"] = df["pickup_datetime"].dt.floor("h")
    df["dropoff_hour_dt"] = df["dropoff_datetime"].dt.floor("h")

    # -----------------------------------------------------------------------
    # Fare and numeric geometry/time
    # -----------------------------------------------------------------------
    df = add_fare_columns(df)

    for col in ["trip_miles", "trip_time", "imputed_trip_time_sec"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Per-trip average speed in mph
    # Prefer imputed_trip_time_sec if available; otherwise assume trip_time is minutes.
    if "trip_miles" in df.columns:
        if "imputed_trip_time_sec" in df.columns:
            denom_hours = df["imputed_trip_time_sec"] / 3600.0
        elif "trip_time" in df.columns:
            # Assuming trip_time is minutes
            denom_hours = df["trip_time"] / 60.0
        else:
            denom_hours = np.nan

        with np.errstate(divide="ignore", invalid="ignore"):
            df["avg_speed_mph"] = df["trip_miles"] / denom_hours

        df.loc[~np.isfinite(df["avg_speed_mph"]), "avg_speed_mph"] = np.nan

    # -----------------------------------------------------------------------
    # Base aggregation by request_hour_dt
    # -----------------------------------------------------------------------
    group = df.groupby("request_hour_dt")

    agg_dict: Dict[str, List[str]] = {
        # geometry / time
        "trip_miles": ["sum", "mean", "median"],
        "trip_time": ["sum", "mean", "median"],

        # fares
        "base_passenger_fare": ["sum", "mean"],
        "subtotal_fare": ["sum", "mean"],
        "rider_total": ["sum", "mean"],
        "tips": ["sum", "mean"],

        # driver economics
        "driver_pay": ["sum", "mean"],
        "driver_pay_with_tips": ["sum", "mean"],
    }

    # avg_speed_mph if we created it
    if "avg_speed_mph" in df.columns:
        agg_dict["avg_speed_mph"] = ["mean", "median"]

    # wait time (seconds) if present
    include_wait = wait_time_col in df.columns
    if include_wait:
        df[wait_time_col] = pd.to_numeric(df[wait_time_col], errors="coerce")
        agg_dict[wait_time_col] = ["mean", "median"]

    agg = group.agg(agg_dict)

    # Flatten MultiIndex columns
    agg.columns = [
        "_".join(col).rstrip("_")
        for col in agg.columns.to_flat_index()
    ]

    # -----------------------------------------------------------------------
    # Rename to final column names
    # -----------------------------------------------------------------------
    rename_map = {
        "trip_miles_sum": "trip_miles_sum",
        "trip_miles_mean": "trip_miles_mean",
        "trip_miles_median": "trip_miles_median",

        "trip_time_sum": "trip_time_sum",
        "trip_time_mean": "trip_time_mean",
        "trip_time_median": "trip_time_median",

        "base_passenger_fare_sum": "base_passenger_fare_sum",
        "base_passenger_fare_mean": "avg_base_passenger_fare",

        "subtotal_fare_sum": "subtotal_fare_sum",
        "subtotal_fare_mean": "avg_subtotal_fare",

        "rider_total_sum": "rider_total_sum",
        "rider_total_mean": "avg_rider_total",

        "tips_sum": "tips_sum",
        "tips_mean": "avg_tips",

        "driver_pay_sum": "driver_pay_sum",
        "driver_pay_mean": "avg_driver_pay",

        "driver_pay_with_tips_sum": "driver_pay_with_tips_sum",
        "driver_pay_with_tips_mean": "avg_driver_pay_with_tips",
    }

    if "avg_speed_mph_mean" in agg.columns:
        rename_map.update({
            "avg_speed_mph_mean": "avg_speed_mph",
            "avg_speed_mph_median": "median_speed_mph",
        })

    if include_wait:
        rename_map.update({
            f"{wait_time_col}_mean": "wait_time_sec_mean",
            f"{wait_time_col}_median": "wait_time_sec_median",
        })

    agg = agg.rename(columns=rename_map)

    # -----------------------------------------------------------------------
    # Counts: requests, pickups, dropoffs per hour
    # -----------------------------------------------------------------------
    request_counts = df.groupby("request_hour_dt").size().rename("request_count")
    pickup_counts = df.groupby("pickup_hour_dt").size().rename("pickup_count")
    dropoff_counts = df.groupby("dropoff_hour_dt").size().rename("dropoff_count")

    hourly = agg.join(request_counts, how="left")
    hourly = hourly.join(pickup_counts, how="left")
    hourly = hourly.join(dropoff_counts, how="left")

    for col in ["request_count", "pickup_count", "dropoff_count"]:
        if col in hourly.columns:
            hourly[col] = hourly[col].fillna(0).astype("int64")

    # -----------------------------------------------------------------------
    # Driver pay as % of base fare (no tips / fees)
    # -----------------------------------------------------------------------
    if "driver_pay_sum" in hourly.columns and "base_passenger_fare_sum" in hourly.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = hourly["driver_pay_sum"] / hourly["base_passenger_fare_sum"]
        pct[~np.isfinite(pct)] = np.nan
        hourly["driver_pay_pct_of_base_fare"] = pct

    # -----------------------------------------------------------------------
    # Final hour keys: datetime_hour, date, request_hour (0–23 int)
    # -----------------------------------------------------------------------
    hourly = hourly.sort_index().reset_index()  # request_hour_dt now a column
    hourly = hourly.rename(columns={"request_hour_dt": "datetime_hour"})

    hourly["date"] = hourly["datetime_hour"].dt.date
    hourly["request_hour"] = hourly["datetime_hour"].dt.hour  # 0–23 int

    key_cols = ["datetime_hour", "date", "request_hour"]
    other_cols = [c for c in hourly.columns if c not in key_cols]
    hourly = hourly[key_cols + other_cols]

    return hourly


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def main(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    wait_time_col: str = DEFAULT_WAIT_TIME_COL,
) -> None:
    PROCESSED_TLC_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading trips from: {input_path}")
    df = pd.read_parquet(input_path)

    # Pre-step: drop discarded rows (bad QC) before aggregation
    if "discard" in df.columns:
        df = df[df["discard"] == 0]

    hourly = aggregate_hourly(df, wait_time_col=wait_time_col)

    print(f"Writing hourly LGA aggregation to: {output_path}")
    hourly.to_parquet(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate cleaned HVFHV trips to hourly LGA metrics."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input parquet file (clean combined trip-level data).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output parquet file (hourly LGA aggregation).",
    )
    parser.add_argument(
        "--wait-time-col",
        type=str,
        default=DEFAULT_WAIT_TIME_COL,
        help="Name of wait time column (default: pickup_wait_time_sec).",
    )

    args = parser.parse_args()
    main(
        input_path=args.input,
        output_path=args.output,
        wait_time_col=args.wait_time_col,
    )
