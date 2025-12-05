# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 00:21:40 2025

@author: epicx
"""

import os
from pathlib import Path
import requests
from typing import Optional

from config import WEATHER_RAW_DIR

NOAA_BASE_URL = "https://www.ncei.noaa.gov/access/services/data/v1"

# ISD (USAF-WBAN) IDs for your three stations
STATIONS = {
    "JFK": "74486094789",
    "LGA": "72503014732",
    "CENTRAL_PARK": "72505394728",
}


def download_isd_hourly(
    station_id: str,
    start_date: str,
    end_date: str,
    token: Optional[str] = None,
) -> Path:
    """
    Download hourly ISD (global-hourly) data for a single station to CSV.

    station_id: e.g. '725033-94789'
    dates: 'YYYY-MM-DD'
    """
    if token is None:
        token = os.getenv("NOAA_TOKEN")  # optional, but useful if you have one
    headers = {}
    if token:
        headers["token"] = token

    params = {
        "dataset": "global-hourly",
        "stations": station_id,
        "startDate": start_date,
        "endDate": end_date,
        "format": "csv",
        "units": "standard",
        "includeAttributes": "false",
    }

    WEATHER_RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_name = f"{station_id}_{start_date}_{end_date}.csv"
    out_path = WEATHER_RAW_DIR / out_name

    print(f"Requesting {station_id} {start_date} → {end_date}")
    r = requests.get(NOAA_BASE_URL, params=params, headers=headers, timeout=120)
    r.raise_for_status()

    out_path.write_text(r.text, encoding="utf-8")
    print(f"Saved {out_path}")
    return out_path


if __name__ == "__main__":
    # match what you did manually: 2025-01-01 to 2025-10-31 (or 2025-08-27)
    START = "2025-01-01"
    END = "2025-08-27"     # or "2025-08-27" if you want to be exact

    for name, sid in STATIONS.items():
        download_isd_hourly(sid, START, END)
