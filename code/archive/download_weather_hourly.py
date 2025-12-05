# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 23:57:17 2025

@author: epicx
"""

import os
from pathlib import Path
from typing import Optional
import requests

from config import WEATHER_RAW_DIR

NOAA_BASE_URL = "https://www.ncei.noaa.gov/access/services/data/v1"


def download_noaa_hourly(
    stations: str,
    start_date: str,
    end_date: str,
    token: Optional[str] = None,
    out_name: Optional[str] = None,
) -> Path:
    """
    Download NOAA NCEI global-hourly (hourly observations) for given stations and dates.

    stations: comma-separated station IDs, e.g. "USW00094789,USW00014732,USW00094728"
    dates: "YYYY-MM-DD"
    """
    if token is None:
        token = os.getenv("NOAA_TOKEN")
    if token is None:
        raise ValueError("NOAA token not provided; set NOAA_TOKEN env or pass token=")

    params = {
        "dataset": "global-hourly",
        "stations": stations,
        "startDate": start_date,
        "endDate": end_date,
        "format": "csv",
        "units": "standard",
        "includeAttributes": "false",
    }

    headers = {"token": token}
    resp = requests.get(NOAA_BASE_URL, params=params, headers=headers, timeout=120)
    resp.raise_for_status()

    WEATHER_RAW_DIR.mkdir(parents=True, exist_ok=True)

    if out_name is None:
        out_name = (
            f"global_hourly_{stations}_{start_date}_{end_date}.csv".replace(",", "_")
        )

    out_path = WEATHER_RAW_DIR / out_name
    out_path.write_text(resp.text, encoding="utf-8")
    print(f"Saved hourly NOAA data to {out_path}")
    return out_path


if __name__ == "__main__":
    stations = "USW00094728"  # Central Park only, to start
    download_noaa_hourly(
        stations=stations,
        start_date="2024-01-01",
        end_date="2024-01-31",
    )
