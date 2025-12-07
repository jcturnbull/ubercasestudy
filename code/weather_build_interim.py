# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 22:13:40 2025

@author: epicx
"""

import pandas as pd
import numpy as np
from pathlib import Path

from config import WEATHER_RAW_DIR   # points to data/raw/weather


def parse_temp_block(val):
    """
    Parse ISD TMP/DEW block: e.g. '+0039,5' or '00390,1' → 3.9°C.
    Returns float in °C, or NaN.
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()

    # Drop anything after comma (quality flag)
    if "," in s:
        s = s.split(",")[0].strip()

    # Missing codes
    if s in {"", "+9999", "-9999", "9999", "99999"}:
        return np.nan

    try:
        return int(s) / 10.0
    except ValueError:
        return np.nan


def parse_wind_block(val):
    """
    Parse ISD WND block in the form:
        'DDD,dir_q,type,SSSS,spd_q'
    Example: '050,5,N,0031,5'
    Returns (dir_deg, speed_ms) with NaN for missing.
    """
    if pd.isna(val):
        return np.nan, np.nan

    s = str(val).strip()
    parts = s.split(",")

    # Expect at least 5 components: dir, dir_q, type, speed, speed_q
    if len(parts) < 5:
        return np.nan, np.nan

    dir_str = parts[0].strip()
    spd_str = parts[3].strip()

    # Direction
    try:
        dir_raw = int(dir_str)
    except ValueError:
        dir_raw = 999  # treat as missing

    if dir_raw >= 990:
        dir_deg = np.nan
    else:
        dir_deg = float(dir_raw)

    # Speed (tenths of m/s)
    try:
        spd_raw = int(spd_str)
    except ValueError:
        spd_raw = 9999  # treat as missing

    if spd_raw >= 9999:
        speed_ms = np.nan
    else:
        speed_ms = spd_raw / 10.0

    return dir_deg, speed_ms



def parse_vis_block(val):
    """
    Parse VIS block in the expanded ISD form:
        'VVVVVV,qual,code,qual'
    Example: '016093,5,N,5' -> 16093 meters.
    """
    if pd.isna(val):
        return np.nan

    s = str(val).strip()
    parts = s.split(",")

    if len(parts) == 0:
        return np.nan

    vis_str = parts[0].strip()

    # Ignore missing codes
    if vis_str in {"", "999999", "99999", "9999"}:
        return np.nan

    try:
        vis_m = int(vis_str)
    except ValueError:
        return np.nan

    # Extremely large values are missing indicators
    if vis_m >= 999999:
        return np.nan

    return float(vis_m)


def parse_precip_1h(val):
    """
    Parse AA1 in expanded ISD form:
        duration,depth,condition,quality

    Count precipitation ONLY when:
        duration == "01"
        condition == "9"

    Convert depth from tenths of millimeters to millimeters.
    """
    import pandas as pd

    if pd.isna(val):
        return 0.0

    s = str(val).strip()
    if not s:
        return 0.0

    parts = s.split(",")
    if len(parts) < 4:
        return 0.0

    dur_str     = parts[0].strip()   # e.g. "01"
    depth_str   = parts[1].strip()   # e.g. "0013"
    cond_str    = parts[2].strip()   # e.g. "9"
    quality_str = parts[3].strip()   # e.g. "5", "6", "1", "9"

    # --- Duration filter: only 1-hour AA1 rows ---
    try:
        dur = int(dur_str)
    except ValueError:
        return 0.0
    if dur != 1:
        return 0.0

    # --- Condition filter: keep ONLY condition == "9" ---
    # As found empirically for LGA, condition 9 contains the real hourly increments.
    if cond_str != "9":
        return 0.0

    # --- Depth parsing ---
    # 9999 = missing; blank = missing
    if depth_str in {"", "9999"}:
        return 0.0

    try:
        raw = int(depth_str)
    except ValueError:
        return 0.0

    if raw > 9998:  # ignore sentinel
        return 0.0

    # Convert tenths of mm to mm
    return raw / 10.0


def process_station_file(file_path: Path) -> pd.DataFrame:
    """
    Read a single ISD CSV and return a tidy hourly DataFrame with:
      station, temp/dew (C/F), wind_dir_deg, wind_speed_ms/mph,
      visibility_m, precip_1h_mm (from AA1 with duration == 1h).
    Indexed by datetime, filtered to 2025.
    """
    print(f"Processing {file_path.name}")
    df = pd.read_csv(file_path, low_memory=False)

    # Normalize column names
    df.columns = [c.strip().upper() for c in df.columns]

    if "DATE" not in df.columns:
        print(f"Skipping {file_path.name}: no DATE column")
        return pd.DataFrame()

    # Datetime index
    df["datetime"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.set_index("datetime").sort_index()

    # Keep only 2025
    df = df[df.index.year == 2025]
    if df.empty:
        print(f"No 2025 rows in {file_path.name}, skipping")
        return pd.DataFrame()

    station_id = file_path.stem

    # Temperature & dew point in °C/°F
    if "TMP" in df.columns:
        temp_c = df["TMP"].map(parse_temp_block)
    else:
        temp_c = pd.Series(np.nan, index=df.index)

    if "DEW" in df.columns:
        dew_c = df["DEW"].map(parse_temp_block)
    else:
        dew_c = pd.Series(np.nan, index=df.index)

    temp_f = temp_c * 9.0 / 5.0 + 32.0
    dew_f = dew_c * 9.0 / 5.0 + 32.0

    # Wind
    if "WND" in df.columns:
        wind_parsed = df["WND"].apply(parse_wind_block)
        wind_dir_deg = wind_parsed.apply(lambda x: x[0])
        wind_speed_ms = wind_parsed.apply(lambda x: x[1])
    else:
        wind_dir_deg = pd.Series(np.nan, index=df.index)
        wind_speed_ms = pd.Series(np.nan, index=df.index)

    wind_speed_mph = wind_speed_ms * 2.23694

    # Visibility
    if "VIS" in df.columns:
        vis_m = df["VIS"].map(parse_vis_block)
    else:
        vis_m = pd.Series(np.nan, index=df.index)

    # Hourly precip (AA1, duration == 1 hour only)
    if "AA1" in df.columns:
        precip_1h_mm = df["AA1"].map(parse_precip_1h)
    else:
        precip_1h_mm = pd.Series(0.0, index=df.index)

    # Assemble final DataFrame
    out = pd.DataFrame(
        {
            "station": station_id,
            "temp_c": temp_c,
            "temp_f": temp_f,
            "dewpoint_c": dew_c,
            "dewpoint_f": dew_f,
            "wind_dir_deg": wind_dir_deg,
            "wind_speed_ms": wind_speed_ms,
            "wind_speed_mph": wind_speed_mph,
            "visibility_m": vis_m,
            "precip_1h_mm": precip_1h_mm,
        },
        index=df.index,
    )

    return out


def build_and_save_interim():
    """
    Process all top-level station CSVs in WEATHER_RAW_DIR and
    save a combined 2025 hourly dataset to data/interim/.
    """
    # Use only top-level station files (exclude daily summaries & manual/)
    files = [
        f
        for f in WEATHER_RAW_DIR.glob("*.csv")
        if "daily-summaries" not in f.name.lower()
    ]

    if not files:
        raise FileNotFoundError(f"No station CSVs found in {WEATHER_RAW_DIR}")

    frames = []
    for f in files:
        df_station = process_station_file(f)
        if not df_station.empty:
            frames.append(df_station)

    if not frames:
        raise RuntimeError("No non-empty station data for 2025.")

    big = pd.concat(frames).sort_index()

    # Compute interim directory from WEATHER_RAW_DIR: data/raw/weather → data/interim
    data_dir = WEATHER_RAW_DIR.parent.parent  # .../data
    interim_dir = data_dir / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    out_path = interim_dir / "weather_hourly_2025.parquet"
    big.to_parquet(out_path)

    print(f"Saved {len(big):,} rows to {out_path}")


if __name__ == "__main__":
    # Map short station codes to their raw NOAA ISD filenames
    STATION_FILES = {
        "lga": "72505394728_2025-01-01_2025-08-27.csv",
        "jfk": "74486094789_2025-01-01_2025-08-27.csv",
        "central_park": "72503014732_2025-01-01_2025-08-27.csv",
    }

    # ---- CHOOSE THE STATIONS TO PROCESS HERE ----
    # Options:
    #   ["lga"]                     → only LGA
    #   ["jfk", "lga"]             → two stations
    #   ["all"]                    → all stations, combined into one file
    stations_to_process = ["lga"]
    # -------------------------------------------------

    data_dir = WEATHER_RAW_DIR.parent.parent  # .../data
    interim_dir = data_dir / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    # Handle "all" logic
    if stations_to_process == ["all"]:
        stations_to_run = list(STATION_FILES.keys())
        combined_output = True
    else:
        stations_to_run = stations_to_process
        combined_output = False

    frames = []

    # Process each requested station
    for code in stations_to_run:
        if code not in STATION_FILES:
            raise ValueError(f"Unknown station code: '{code}'")

        filename = STATION_FILES[code]
        file_path = WEATHER_RAW_DIR / filename

        print(f"Processing station: {code.upper()} from {filename}")
        df_station = process_station_file(file_path)

        if df_station.empty:
            raise RuntimeError(f"No 2025 data found for station '{code}'")

        # Save individual file only if NOT combining
        if not combined_output:
            out_path = interim_dir / f"weather_{code}_hourly_2025.parquet"
            df_station.to_parquet(out_path)
            print(f"Saved {len(df_station):,} rows to {out_path}")

        frames.append(df_station)

    # If combining into one file, concatenate and save
    if combined_output:
        combined = pd.concat(frames).sort_index()
        out_path = interim_dir / "weather_hourly_2025.parquet"
        combined.to_parquet(out_path)
        print(f"Saved combined file with {len(combined):,} rows to {out_path}")
