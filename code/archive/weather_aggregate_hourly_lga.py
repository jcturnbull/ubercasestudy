# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 00:09:18 2025

@author: epicx
"""
import pandas as pd
from pathlib import Path
from config import WEATHER_RAW_DIR  # we reuse this to locate /data/


def aggregate_lga_hourly():
    """
    Load interim LGA weather and aggregate to hourly resolution.

    For each hour, compute:
      - temp_f_max, temp_f_min, temp_f_mean
      - dewpoint_f_max, dewpoint_f_min, dewpoint_f_mean
      - wind_speed_mph_max, wind_speed_mph_min, wind_speed_mph_mean
      - precip_1h_mm_total

    Output to data/processed/weather_lga_hourly_agg_2025.parquet
    """

    # Derive paths from WEATHER_RAW_DIR: .../data/raw/weather -> .../data
    data_dir = WEATHER_RAW_DIR.parent.parent
    interim_dir = data_dir / "interim"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Input: interim LGA file created by weather_build_interim.py
    in_path = interim_dir / "weather_lga_hourly_2025.parquet"
    print(f"Loading {in_path}")
    df = pd.read_parquet(in_path)

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df = df.set_index("datetime")
        else:
            raise ValueError("Expected DatetimeIndex or 'datetime' column in LGA weather data.")

    df = df.sort_index()

    # Hourly aggregation:
    # Resample to hourly bins: each hour label is the left edge (00:00, 01:00, ...)
    hourly = df.resample("H").agg(
        temp_f_max=("temp_f", "max"),
        temp_f_min=("temp_f", "min"),
        temp_f_mean=("temp_f", "mean"),
        dewpoint_f_max=("dewpoint_f", "max"),
        dewpoint_f_min=("dewpoint_f", "min"),
        dewpoint_f_mean=("dewpoint_f", "mean"),
        wind_speed_mph_max=("wind_speed_mph", "max"),
        wind_speed_mph_min=("wind_speed_mph", "min"),
        wind_speed_mph_mean=("wind_speed_mph", "mean"),
        precip_1h_mm_total=("precip_1h_mm", "sum"),
    )

    # Add convenience columns for date and hour-of-day (0–23)
    hourly["date"] = hourly.index.date
    hourly["hour"] = hourly.index.hour

    # Reorder columns a bit
    cols = [
        "date",
        "hour",
        "temp_f_max",
        "temp_f_min",
        "temp_f_mean",
        "dewpoint_f_max",
        "dewpoint_f_min",
        "dewpoint_f_mean",
        "wind_speed_mph_max",
        "wind_speed_mph_min",
        "wind_speed_mph_mean",
        "precip_1h_mm_total",
    ]
    hourly = hourly[cols]

    out_path = processed_dir / "weather_lga_hourly_agg_2025.parquet"
    hourly.to_parquet(out_path)

    print(f"Saved {len(hourly):,} hourly rows to {out_path}")


if __name__ == "__main__":
    aggregate_lga_hourly()
