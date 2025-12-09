# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 16:14:41 2025

@author: epicx

join_tlc_weather_hourly.py

Align hourly TLC metrics (by datetime_hour) with LGA hourly weather for 2025.

Key points:
- Join on df_tlc['datetime_hour'] to df_weather.index (DatetimeIndex).
- Inner join → keep only overlapping hours.
"""

from pathlib import Path
import pandas as pd

from config import PROCESSED_DIR


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load hourly aggregated TLC and enhanced weather datasets.
    """
    tlc_path = PROCESSED_DIR / "tlc" / "fhvhv_lga_hourly_agg_2025.parquet"
    weather_path = PROCESSED_DIR / "weather_lga_hourly_agg_2025_enhanced.parquet"

    df_tlc = pd.read_parquet(tlc_path)
    df_weather = pd.read_parquet(weather_path)

    return df_tlc, df_weather


def prepare_time_columns(df_tlc: pd.DataFrame, df_weather: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ensure TLC datetime_hour is datetime and weather index is DatetimeIndex.

    TLC:
        - Use datetime_hour as the hourly key for merging.
    Weather:
        - Use DatetimeIndex (already hourly) as the key.
    """
    df_tlc = df_tlc.copy()
    df_weather = df_weather.copy()

    # TLC time key
    if "datetime_hour" not in df_tlc.columns:
        raise KeyError("TLC dataframe is missing 'datetime_hour' column.")

    df_tlc["datetime_hour"] = pd.to_datetime(df_tlc["datetime_hour"])

    # Weather: index must be DatetimeIndex
    if not isinstance(df_weather.index, pd.DatetimeIndex):
        raise TypeError("Weather dataframe index must be a DatetimeIndex.")

    # Sort for sanity
    df_tlc = df_tlc.sort_values("datetime_hour")
    df_weather = df_weather.sort_index()

    # Optional: quick uniqueness check on TLC key
    dup_count = df_tlc["datetime_hour"].duplicated().sum()
    if dup_count > 0:
        print(f"WARNING: {dup_count} duplicate datetime_hour values in TLC data.")
        # If you want to hard-fail instead of allow m:1 merge, uncomment:
        # raise ValueError("datetime_hour is not unique in TLC; fix upstream aggregation.")

    return df_tlc, df_weather


def join_hourly(df_tlc: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    """
    Inner join:
        TLC:      datetime_hour (column)
        Weather:  datetime index

    Assumes one weather row per hour; TLC may be 1:1 or m:1 depending on aggregation.
    """

    # Before merge:
    df_weather = df_weather.drop(columns=["date", "hour"], errors="ignore")

    merged = df_tlc.merge(
        df_weather,
        left_on="datetime_hour",
        right_index=True,
        how="inner",
        validate="1:1",  # change to "m:1" if needed
    )

    print(f"TLC rows:     {len(df_tlc):,}")
    print(f"Weather rows: {len(df_weather):,}")
    print(f"Merged rows:  {len(merged):,}")

    if not merged.empty:
        print(
            "Merged range:",
            merged["datetime_hour"].min(),
            "→",
            merged["datetime_hour"].max(),
        )

    return merged


def save_output(df_merged: pd.DataFrame) -> Path:
    """
    Save merged dataset.
    """
    out_path = PROCESSED_DIR / "fhvhv_lga_hourly_with_weather_2025.parquet"
    df_merged.to_parquet(out_path, index=False)
    print(f"Saved merged file to: {out_path}")
    return out_path


def main() -> None:
    df_tlc, df_weather = load_inputs()
    df_tlc, df_weather = prepare_time_columns(df_tlc, df_weather)
    df_merged = join_hourly(df_tlc, df_weather)
    save_output(df_merged)


if __name__ == "__main__":
    main()