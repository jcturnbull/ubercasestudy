# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 23:13:58 2025

@author: epicx
"""

import os
from pathlib import Path
from typing import Optional
import requests

from config import WEATHER_RAW_DIR

NOAA_BASE_URL = "https://www.ncei.noaa.gov/access/services/data/v1"

def download_noaa_daily(
    dataset: str,
    stations: str,
    start_date: str,
    end_date: str,
    token: Optional[str] = None,
    out_name: Optional[str] = None,
) -> Path:
    """
    dataset: e.g. 'daily-summaries'
    stations: comma-separated station IDs, e.g. 'USW00094728' (JFK-type station)
    dates: 'YYYY-MM-DD'
    token: NOAA token (or read from NOAA_TOKEN env)
    """
    if token is None:
        token = os.getenv("NOAA_TOKEN")
    if token is None:
        raise ValueError("NOAA token not provided; set NOAA_TOKEN env or pass token=")

    params = {
        "dataset": dataset,
        "stations": stations,
        "startDate": start_date,
        "endDate": end_date,
        "format": "csv",
        "units": "standard",  # Fahrenheit, inches, etc.
        # add 'dataTypes', 'units', etc. once you pin what you want
    }

    headers = {"token": token}
    resp = requests.get(NOAA_BASE_URL, params=params, headers=headers, timeout=60)
    resp.raise_for_status()

    WEATHER_RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    if out_name is None:
        out_name = f"{dataset}_{stations}_{start_date}_{end_date}.csv".replace(",", "_")
    
    out_path = WEATHER_RAW_DIR / out_name
    out_path.write_text(resp.text, encoding="utf-8")
    print(f"Saved NOAA data to {out_path}")
    return out_path


if __name__ == "__main__":
    # Example (fill in real station IDs and dates):
    stations="USW00094728,USW00014732,USW00094789"
    download_noaa_daily(
        dataset="daily-summaries",
        stations=stations,
        start_date="2025-01-01",
        end_date="2025-11-01",
    )
