# -*- coding: utf-8 -*-
from __future__ import annotations
"""
tlc_processor.py

Process TLC HVFHV parquet files of the form:
    fhvhv_tripdata_2025-<mm>.parquet

Usage (Spyder):
    - Open this file in Spyder.
    - Edit MONTHS_TO_PROCESS in the __main__ block (e.g. ["03"]).
    - Run the script (F5 / F9).

Behavior:
1. Read from data/raw/tlc/.
2. Keep only the specified columns.
3. For March 2025 (03) only:
   - On 2025-03-09, if pickup_datetime is in 02:00:00–02:59:59
     AND on_scene_datetime is in 01:xx:xx that same date,
     shift pickup_datetime back by one hour.
4. Add trip-level metrics (wait time, revenue per mile/min, flags, etc.).
5. Write cleaned parquet to data/interim/tlc/ with suffix "_clean".
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

# TLC taxi zone IDs for airports (standard mapping)
# Newark = 1, JFK = 132, LaGuardia = 138
AIRPORT_LOCATION_IDS = {1, 132, 138}
LGA_LOCATION_IDS = {138}


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

    target_day = date(YEAR, 3, 9)

    day_mask = pickup.dt.date == target_day
    hour_2_mask = pickup.dt.hour == 2

    if on_scene is not None:
        on_scene_ok = (
            (on_scene.dt.date == target_day) &
            (on_scene.dt.hour == 1)
        )
    else:
        # If on_scene doesn't exist, don't fix anything by default.
        on_scene_ok = pd.Series(False, index=df.index)

    fix_mask = day_mask & hour_2_mask & on_scene_ok

    df.loc[fix_mask, "pickup_datetime"] = (
        df.loc[fix_mask, "pickup_datetime"] - pd.Timedelta(hours=1)
    )

    return df


def format_seconds_to_mmss(sec_series: pd.Series) -> pd.Series:
    """
    Convert a series of seconds (float/int) to 'MM:SS' string format.
    69 minutes 36 seconds -> '69:36'. NaN -> None.
    """
    def _fmt(x):
        if pd.isna(x):
            return None
        total = int(x)
        minutes = total // 60
        seconds = total % 60
        return f"{minutes:d}:{seconds:02d}"

    return sec_series.apply(_fmt)


def safe_divide(num: pd.Series, denom: pd.Series) -> pd.Series:
    """
    Element-wise num / denom with:
      - both coerced to float
      - denom==0 or NaN -> result NaN
    """
    num = pd.to_numeric(num, errors="coerce")
    denom = pd.to_numeric(denom, errors="coerce")
    denom = denom.replace(0, pd.NA)
    return num / denom


TZ_NAME = "America/New_York"


def add_imputed_trip_time(df: pd.DataFrame, tz_name: str = TZ_NAME) -> pd.DataFrame:
    """
    Add DST-aware imputed trip time based on pickup/dropoff timestamps.

    - imputed_trip_time_sec: (dropoff - pickup) in real elapsed seconds,
      computed in a timezone-aware way so the 2025-03-09 DST jump is correct.
    - trip_time_impute_diff_sec: reported trip_time - imputed_trip_time_sec.

    If required columns are missing, fills with NA.
    """
    if not {"pickup_datetime", "dropoff_datetime"}.issubset(df.columns):
        df["imputed_trip_time_sec"] = pd.NA
        df["trip_time_impute_diff_sec"] = pd.NA
        return df

    # Ensure datetime types
    ensure_datetimes(df, ["pickup_datetime", "dropoff_datetime"])

    pickup = df["pickup_datetime"]
    dropoff = df["dropoff_datetime"]

    # Localize to NY time; handle nonexistent DST times by shifting forward into valid time
    pickup_tz = pickup.dt.tz_localize(
        tz_name, nonexistent="shift_forward", ambiguous="NaT"
    )
    dropoff_tz = dropoff.dt.tz_localize(
        tz_name, nonexistent="shift_forward", ambiguous="NaT"
    )

    df["imputed_trip_time_sec"] = (dropoff_tz - pickup_tz).dt.total_seconds()

    if "trip_time" in df.columns:
        trip_sec = pd.to_numeric(df["trip_time"], errors="coerce")
        df["trip_time_impute_diff_sec"] = trip_sec - df["imputed_trip_time_sec"]
    else:
        df["trip_time_impute_diff_sec"] = pd.NA

    return df


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trip-level metrics and flags:

    - pickup_wait_time_sec, pickup_wait_time_mmss
    - trip_time_mmss
    - fare_per_mile, fare_per_minute
    - driver_pay_per_mile, driver_pay_per_hour
    - payout_ratio = (driver_pay + tips) / base_passenger_fare
    - is_airport_trip, is_lga_pickup, is_lga_dropoff, is_cbd_trip
    - pickup_date, pickup_hour, pickup_dayofweek, pickup_is_weekend
    - discard flag for bad/missing wait times
    """
    # Ensure datetimes again in case this is called without DST fix
    ensure_datetimes(df, [
        "request_datetime", "on_scene_datetime",
        "pickup_datetime", "dropoff_datetime"
    ])
    
    # Add DST-aware imputed trip time and comparison vs reported trip_time
    df = add_imputed_trip_time(df)

    # 1. Passenger wait time: pickup - request
    if {"request_datetime", "pickup_datetime"}.issubset(df.columns):
        wait_td = df["pickup_datetime"] - df["request_datetime"]
        df["pickup_wait_time_sec"] = wait_td.dt.total_seconds()
        df["pickup_wait_time_mmss"] = format_seconds_to_mmss(df["pickup_wait_time_sec"])
    else:
        df["pickup_wait_time_sec"] = pd.NA
        df["pickup_wait_time_mmss"] = None

    # Discard flag: bad wait time OR invalid congestion fees OR negative driver pay

    # 1) Negative or missing wait time
    discard_mask = (
        df["pickup_wait_time_sec"].isna() |
        (df["pickup_wait_time_sec"] < 0)
    )

    # Invalid congestion_surcharge (allowed: 0, 0.75, 2.75)
    if "congestion_surcharge" in df.columns:
        cong = pd.to_numeric(df["congestion_surcharge"], errors="coerce")
        valid_cong_vals = {0.0, 0.75, 2.75}
        invalid_cong = cong.notna() & ~cong.isin(valid_cong_vals)
        discard_mask |= invalid_cong

    # Invalid cbd_congestion_fee (allowed: 0, 1.50)
    if "cbd_congestion_fee" in df.columns:
        cbd = pd.to_numeric(df["cbd_congestion_fee"], errors="coerce")
        valid_cbd_vals = {0.0, 1.50}
        invalid_cbd = cbd.notna() & ~cbd.isin(valid_cbd_vals)
        discard_mask |= invalid_cbd

    # Negative driver pay
    if "driver_pay" in df.columns:
        driver_pay_num = pd.to_numeric(df["driver_pay"], errors="coerce")
        discard_mask |= (driver_pay_num < 0)

    # NEGATIVE DISTANCE
    if "trip_miles" in df.columns:
        miles_num = pd.to_numeric(df["trip_miles"], errors="coerce")
        discard_mask |= (miles_num < 0)

    # NEGATIVE TRIP TIME
    if "trip_time" in df.columns:
        ttime_num = pd.to_numeric(df["trip_time"], errors="coerce")
        discard_mask |= (ttime_num < 0)

    # NEGATIVE TIPS
    if "tips" in df.columns:
        tips_num = pd.to_numeric(df["tips"], errors="coerce")
        discard_mask |= (tips_num < 0)
    
    # Ensure datetime types
    req = df["request_datetime"]
    ons = df["on_scene_datetime"]
    pu  = df["pickup_datetime"]
    do  = df["dropoff_datetime"]

    # 1. request < pickup
    invalid_req_pu = req.notna() & pu.notna() & (req >= pu)
    discard_mask |= invalid_req_pu

    # 2. pickup < dropoff
    invalid_pu_do = pu.notna() & do.notna() & (pu >= do)
    discard_mask |= invalid_pu_do

    # 3. request < on_scene < pickup (but only if on_scene exists)
    has_ons = ons.notna()

    invalid_req_ons = has_ons & req.notna() & (req >= ons)
    invalid_ons_pu  = has_ons & pu.notna()  & (ons >= pu)

    discard_mask |= invalid_req_ons
    discard_mask |= invalid_ons_pu
    
    df["discard"] = discard_mask.astype("int8")

    # 2. Trip time in MM:SS
    if "trip_time" in df.columns:
        trip_sec = pd.to_numeric(df["trip_time"], errors="coerce")
        df["trip_time_mmss"] = format_seconds_to_mmss(trip_sec)
    else:
        df["trip_time_mmss"] = None

    # 3. Revenue per mile/minute
    if {"base_passenger_fare", "trip_miles", "trip_time"}.issubset(df.columns):
        trip_miles = pd.to_numeric(df["trip_miles"], errors="coerce")
        trip_time_sec = pd.to_numeric(df["trip_time"], errors="coerce")

        df["fare_per_mile"] = safe_divide(df["base_passenger_fare"], trip_miles)
        df["fare_per_minute"] = safe_divide(
            df["base_passenger_fare"],
            trip_time_sec / 60.0
        )
    else:
        df["fare_per_mile"] = pd.NA
        df["fare_per_minute"] = pd.NA

    # 4. Driver pay per mile / per hour + payout ratio
    if {"driver_pay", "trip_miles", "trip_time"}.issubset(df.columns):
        trip_miles = pd.to_numeric(df["trip_miles"], errors="coerce")
        trip_time_sec = pd.to_numeric(df["trip_time"], errors="coerce")

        df["driver_pay_per_mile"] = safe_divide(df["driver_pay"], trip_miles)
        df["driver_pay_per_hour"] = safe_divide(
            df["driver_pay"],
            trip_time_sec / 3600.0
        )
    else:
        df["driver_pay_per_mile"] = pd.NA
        df["driver_pay_per_hour"] = pd.NA

    if {"driver_pay", "tips", "base_passenger_fare"}.issubset(df.columns):
        df["payout_ratio"] = safe_divide(
            df["driver_pay"] + df["tips"],
            df["base_passenger_fare"]
        )
    else:
        df["payout_ratio"] = pd.NA

    # 5. Trip flags: airport / LGA / CBD
    if {"PULocationID", "DOLocationID"}.issubset(df.columns):
        df["is_airport_trip"] = (
            df["PULocationID"].isin(AIRPORT_LOCATION_IDS) |
            df["DOLocationID"].isin(AIRPORT_LOCATION_IDS)
        ).astype("int8")

        df["is_lga_pickup"] = df["PULocationID"].isin(LGA_LOCATION_IDS).astype("int8")
        df["is_lga_dropoff"] = df["DOLocationID"].isin(LGA_LOCATION_IDS).astype("int8")
    else:
        df["is_airport_trip"] = 0
        df["is_lga_pickup"] = 0
        df["is_lga_dropoff"] = 0

    if "cbd_congestion_fee" in df.columns:
        cbd_fee = pd.to_numeric(df["cbd_congestion_fee"], errors="coerce")
        df["is_cbd_trip"] = (cbd_fee > 0).astype("int8")
    else:
        df["is_cbd_trip"] = 0

    # 6. Calendar fields: date, hour, dayofweek, weekend flag
    if "pickup_datetime" in df.columns:
        pickup = df["pickup_datetime"]
        df["pickup_date"] = pickup.dt.date
        df["pickup_hour"] = pickup.dt.hour
        df["pickup_dayofweek"] = pickup.dt.dayofweek  # 0=Mon, 6=Sun
        df["pickup_is_weekend"] = pickup.dt.dayofweek.isin([5, 6]).astype("int8")
    else:
        df["pickup_date"] = None
        df["pickup_hour"] = pd.NA
        df["pickup_dayofweek"] = pd.NA
        df["pickup_is_weekend"] = 0

    return df


def process_month(month_code: str) -> Path:
    """
    Process a single month code ('01', '02', ..., '12'):
      - load -> trim columns -> optional DST fix (March only) -> add metrics -> save.
    """
    month_code = f"{int(month_code):02d}"  # normalize

    df = load_and_trim(month_code)

    # Apply DST correction only for March (03)
    if month_code == "03":
        df = fix_march_dst_pickups(df)

    # Add trip-level metrics
    df = add_derived_metrics(df)
    
    # Reduce file size by discarding bad rows and non-airport trips
    df = df[df["discard"] == 0]
    
    if "is_airport_trip" in df.columns:
        df = df[df["is_airport_trip"] == 1]

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
    MONTHS_TO_PROCESS = ["01", "02", "03", "04", "05", "06", "07", "08"]  # e.g. ["03"] or ["01", "02", "03", "04"]

    main(MONTHS_TO_PROCESS)
