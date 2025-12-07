# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 14:28:59 2025

@author: epicx
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Make sure we can import from code/
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(CURRENT_DIR))

from config import WEATHER_RAW_DIR
from load_isd_hourly import load_isd_station


def parse_aa_precip(aa_value):
    """
    Very rough parser for ISD AA1/AA2 precipitation blocks.
    Assumes standard ISD formatting where the first 4 digits are
    precipitation in tenths of mm. Adjust if this looks wrong in the data.
    """
    if pd.isna(aa_value):
        return 0.0
    s = str(aa_value)
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) < 4:
        return 0.0
    try:
        amt_tenths_mm = int(digits[:4])
    except ValueError:
        return 0.0
    return amt_tenths_mm / 10.0  # mm


def load_lga_january():
    """Load LGA weather and slice to January 2025."""
    file_path = WEATHER_RAW_DIR / "72505394728_2025-01-01_2025-08-27.csv"
    lga = load_isd_station(file_path)

    # Basic cleaning
    if "TEMP" in lga.columns:
        lga["TEMP"] = pd.to_numeric(lga["TEMP"], errors="coerce")

    # Parse precip from AA1/AA2 if present
    precip_cols = [c for c in ["AA1", "AA2"] if c in lga.columns]
    if precip_cols:
        precip = lga[precip_cols].applymap(parse_aa_precip).sum(axis=1)
        lga["precip_mm"] = precip
    else:
        lga["precip_mm"] = 0.0

    # Restrict to Jan 2025
    jan_lga = lga.loc["2025-01-01":"2025-01-31"].copy()
    return jan_lga


def plot_daily_temp_and_precip(jan_lga: pd.DataFrame):
    """Daily mean temp and total precip with dual y-axes."""
    daily = jan_lga.resample("D").agg(
        {
            "TEMP": "mean",
            "precip_mm": "sum",
        }
    )

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(daily.index, daily["TEMP"], label="Temperature", linewidth=1.5)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Temperature (raw TEMP units)")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    ax2.bar(daily.index, daily["precip_mm"], alpha=0.3)
    ax2.set_ylabel("Precipitation (mm, rough parse)")
    ax2.tick_params(axis="y")

    fig.tight_layout()
    plt.title("LGA – Daily Temperature and Precipitation (Jan 2025)")
    plt.show()


def main():
    jan_lga = load_lga_january()

    # Quick sanity checks
    print("LGA January 2025 head():")
    print(jan_lga.head())
    print("\nLGA January 2025 info():")
    print(jan_lga.info())
    print("\nLGA January 2025 TEMP/precip stats:")
    print(jan_lga[["TEMP", "precip_mm"]].describe())

    plot_daily_temp_and_precip(jan_lga)


if __name__ == "__main__":
    main()


