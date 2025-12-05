# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 00:17:30 2025

@author: epicx
"""

import pandas as pd
from pathlib import Path
from config import WEATHER_RAW_DIR


def load_isd_station(file_path: Path) -> pd.DataFrame:
    """
    Load a manually downloaded ISD CSV file.
    Normalizes column names and parses timestamp into a single datetime index.
    """
    df = pd.read_csv(file_path)

    # Normalize column names (NOAA uses uppercase)
    df.columns = [c.strip().upper() for c in df.columns]

    # ISD timestamps come as separate fields in some cases, or as DATE in others.
    # If DATE exists and is a full timestamp string, use it.
    if "DATE" in df.columns:
        df["datetime"] = pd.to_datetime(df["DATE"], errors="coerce")
    else:
        raise ValueError(f"No DATE column in {file_path}. Columns found: {df.columns}")

    df = df.set_index("datetime").sort_index()

    # Attach station ID from filename
    df["station"] = file_path.stem

    # Minimal useful subset for your analysis — adjust as needed
    useful_cols = [
        "TEMP",      # temperature
        "DEW",       # dew point
        "WND",       # wind block
        "AA1",       # precipitation block
        "AA2",
        "station",
    ]

    # Keep only available columns
    keep = [c for c in useful_cols if c in df.columns]
    return df[keep]
    


def load_all_isd_2025() -> pd.DataFrame:
    """
    Loads all ISD CSVs placed in data/raw/weather for 2025.
    Automatically detects the station files.
    """
    files = list(WEATHER_RAW_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No ISD CSV files found in data/raw/weather/*.csv")

    dfs = []
    for f in files:
        print(f"Loading {f.name}")
        df = load_isd_station(f)
        # filter year 2025 only
        df = df[df.index.year == 2025]
        dfs.append(df)

    big = pd.concat(dfs).sort_index()
    return big
