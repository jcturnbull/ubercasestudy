# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 00:09:18 2025

@author: epicx
"""
import pandas as pd
from pathlib import Path
from config import WEATHER_RAW_DIR  # used only to derive /data path


# Mapping of logical station names -> suffix used in filenames
STATIONS = {
    "lga": "lga",
    "jfk": "jfk",
    "central_park": "central_park",
}

# Filename templates (parquet; change to .csv if you ever want that)
INTERIM_TEMPLATE = "weather_{station}_hourly_2025.parquet"
OUTPUT_TEMPLATE = "weather_{station}_hourly_agg_2025.parquet"


def aggregate_station_hourly(station: str) -> Path:
    """
    Aggregate one station's interim hourly weather to 1-hour buckets.

    Inputs (per station):
      data/interim/weather_{station}_hourly_2025.parquet

    Outputs:
      data/processed/weather_{station}_hourly_agg_2025.parquet
    """
    if station not in STATIONS:
        raise ValueError(f"Unknown station '{station}'. Expected one of {list(STATIONS.keys())}.")

    # Derive data dirs from WEATHER_RAW_DIR: .../data/raw/weather -> .../data
    data_dir = WEATHER_RAW_DIR.parent.parent
    interim_dir = data_dir / "interim"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Build paths
    interim_name = INTERIM_TEMPLATE.format(station=STATIONS[station])
    in_path = interim_dir / interim_name

    print(f"[{station}] Loading {in_path}")
    df = pd.read_parquet(in_path)

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df = df.set_index("datetime")
        else:
            raise ValueError(f"[{station}] Expected DatetimeIndex or 'datetime' column.")

    df = df.sort_index()

    # --- NEW: convert from UTC -> America/New_York (DST-aware) ---
    # Assume timestamps in the interim file are UTC.
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        # Just in case upstream already tagged them as something else.
        df.index = df.index.tz_convert("UTC")

    # Convert to local NYC time (handles DST transitions correctly)
    df.index = df.index.tz_convert("America/New_York")
    # --------------------------------------------------------------

    # Resample to hourly bins: 00:00–00:59 => 00:00 label, etc.
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
    # pandas agg functions ignore NaN, so your 3, 5, NaN, 10 -> 6 behavior holds.

    # If you prefer naive local timestamps, drop tz here:
    hourly.index = hourly.index.tz_localize(None)

    # Add convenience fields
    hourly["date"] = hourly.index.date
    hourly["hour"] = hourly.index.hour
    # Add a short station identifier column so we can distinguish rows in "all"
    hourly["station"] = station  # 'lga', 'jfk', or 'central_park'

    cols = [
        "date",
        "hour",
        "station",
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

    out_name = OUTPUT_TEMPLATE.format(station=STATIONS[station])
    out_path = processed_dir / out_name
    hourly.to_parquet(out_path)

    print(f"[{station}] Saved {len(hourly):,} hourly rows to {out_path}")
    return out_path


def aggregate_multiple(stations: list[str]) -> None:
    """Aggregate a list of stations into separate per-station files."""
    for s in stations:
        aggregate_station_hourly(s)


def aggregate_all() -> Path:
    """
    Aggregate all stations and combine into a single file:

      data/processed/weather_hourly_agg_2025.parquet

    with columns: date, hour, station, ...
    """
    data_dir = WEATHER_RAW_DIR.parent.parent
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for s in STATIONS.keys():
        # This both writes per-station file and returns its path
        path = aggregate_station_hourly(s)
        df_s = pd.read_parquet(path)
        frames.append(df_s)

    combined = pd.concat(frames, ignore_index=False)
    # Optional: sort by date/hour/station for sanity
    combined = combined.sort_values(["date", "hour", "station"])

    out_path = processed_dir / "weather_hourly_agg_2025.parquet"
    combined.to_parquet(out_path)
    print(f"[all] Saved {len(combined):,} hourly rows to {out_path}")
    return out_path


if __name__ == "__main__":
    # MODE 1: single station
    aggregate_station_hourly("lga")
    # aggregate_station_hourly("jfk")
    # aggregate_station_hourly("central_park")

    # MODE 2: specific subset (separate files)
    # aggregate_multiple(["lga", "jfk"])

    # MODE 3: all stations into one combined file with a 'station' column
    # aggregate_all()
