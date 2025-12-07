# -*- coding: utf-8 -*-
from __future__ import annotations
"""
tlc_processor.py

Created on Sun Dec 7 14:43:47 2025
@author: epicx

Process TLC HVFHV parquet files:
  fhvhv_tripdata_2025-<mm>.parquet

Usage (Spyder):
  - Edit MONTHS_TO_PROCESS in __main__
  - Run with F5/F9

Behavior:
  1. Load raw parquet from data/raw/tlc/
  2. Keep defined columns
  3. Fix March 9th DST pickups (02 → 01 hour shift)
  4. Save cleaned parquet to data/interim/tlc/
"""

from pathlib import Path
from datetime import date

import pandas as pd


YEAR = 2025

ALLOWED_COLS = [
    "hvfhs_license_num",
    "request_datetime",
    "on_scene_datetime",
    "pickup_datetime",
    "dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_miles",
    "trip_time",
    "base_passenger_fare",
    "tolls",
    "bcf",
    "sales_tax",
    "congestion_surcharge",
    "airport_fee",
    "tips",
    "driver_pay",
    "cbd_congestion_fee",
]

PROJECT_ROOT = Path(r"C:/Users/epicx/Projects/ubercasestudy")

RAW_DIR = PROJECT_ROOT / "data/raw/tlc"
INTERIM_DIR = PROJECT_ROOT / "data/interim/tlc"


def load_and_trim(month_code: str) -> pd.DataFrame:
    """
    Load fhvhv_tripdata_2025-<mm>.parquet and keep only ALLOWED_COLS.
    Any allowed columns missing from the file are simply skipped.
    """
    infile = RAW_DIR / f"fhvhv_tripdata_{YEAR}-{month_code}.parquet"
    if not infile.exists():
        raise FileNotFoundError(f"Input file not found: {infile}")

    df = pd.read_parquet(infile)

    present_cols = [c for c in ALLOWED_COLS if c in df.columns]
    df = df[present_cols].copy()
    return df


def ensure_datetimes(df: pd.DataFrame, cols: list[str]) -> None:
    """
    In-place: ensure the given columns are datetime64[ns] (coerce invalid to NaT).
    """
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")


def fix_march_dst_pickups(df: pd.DataFrame) -> pd.DataFrame:
    """
    For March 2025 data only:

    On 2025-03-09, when pickup_datetime falls in the nonexistent local hour
    02:00:00–02:59:59, and on_scene_datetime suggests the trip really belongs
    in the 01-hour on the same date, shift pickup_datetime back by one hour.

    Assumes df is the March (03) file for YEAR.
    """
    if "pickup_datetime" not in df.columns:
        return df

    ensure_datetimes(df, [
        "request_datetime", "on_scene_datetime",
        "pickup_datetime", "dropoff_datetime"
    ])

    pickup = df["pickup_datetime"]
    on_scene = df.get("on_scene_datetime")

    # 2025-03-09
    target_day = date(YEAR, 3, 9)

    day_mask = pickup.dt.date == target_day
    hour_2_mask = pickup.dt.hour == 2

    if on_scene is not None:
        on_scene_ok = (
            (on_scene.dt.date == target_day)
            & (on_scene.dt.hour == 1)
        )
    else:
        # If on_scene doesn't exist, don't fix anything by default.
        on_scene_ok = pd.Series(False, index=df.index)

    fix_mask = day_mask & hour_2_mask & on_scene_ok

    # Shift pickup back one hour for affected rows
    df.loc[fix_mask, "pickup_datetime"] = (
        df.loc[fix_mask, "pickup_datetime"] - pd.Timedelta(hours=1)
    )

    return df


def process_month(month_code: str) -> Path:
    """
    Process a single month code ('01', '02', ..., '12'):
      - load -> trim columns -> optional DST fix (March only) -> save.
    """
    month_code = f"{int(month_code):02d}"  # normalize

    df = load_and_trim(month_code)

    # Apply DST correction only for March (03)
    if month_code == "03":
        df = fix_march_dst_pickups(df)

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    outfile = INTERIM_DIR / f"fhvhv_tripdata_{YEAR}-{month_code}_clean.parquet"
    df.to_parquet(outfile, index=False)

    return outfile


def main(months_to_process: list[str]) -> None:
    """
    Run processing for a list of month codes (e.g., ["03", "04"]).
    """
    for m in months_to_process:
        try:
            out_path = process_month(m)
            print(f"Processed {m} -> {out_path}")
        except FileNotFoundError as e:
            print(f"[WARN] {e}")


if __name__ == "__main__":
    # EDIT THIS LIST IN SPYDER TO CONTROL WHICH MONTHS RUN
    MONTHS_TO_PROCESS = ["03"]  # e.g. ["01", "02", "03", "04"]

    main(MONTHS_TO_PROCESS)
