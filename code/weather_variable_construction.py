#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 11:07:21 2025

@author: jacksonturnbull
"""
#!/usr/bin/env python3
"""
weather_variable_construction.py

Step 1: Load processed hourly LGA weather data and inspect columns.
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Base project path
project_root = pathlib.Path("/Users/jacksonturnbull/Projects/ubercasestudy")

# Directory containing processed weather files (no file name here)
weather_path = project_root / "data" / "processed"


def get_file_info(file_name: str):
    """
    Load a weather parquet file from the processed directory and print schema info.

    Parameters
    ----------
    file_name : str, optional
        File name within data/processed/. Defaults to 'weather_lga_hourly_agg_2025.parquet'.
    """

    file_path = (weather_path / file_name).resolve()

    print(f"\nLoading file: {file_path}")
    df = pd.read_parquet(file_path)

    print("\n=== DataFrame shape ===")
    print(df.shape)

    print("\n=== Columns and dtypes ===")
    print(df.dtypes)

    print("\n=== First 5 rows ===")
    print(df.head())

    non_numeric = df.select_dtypes(exclude="number")
    if not non_numeric.empty:
        print("\n=== Sample of non-numeric columns ===")
        print(non_numeric.head())

    print("\n=== Summary statistics (numeric) ===")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print(df.describe().T.round(2))

    return df

def compute_relative_humidity(temp_f, dewpoint_f):
    """
    Compute relative humidity (%) using temperature and dewpoint in °F.
    Formula uses the Magnus approximation.
    """

    # Convert °F → °C for the formula
    temp_c = (temp_f - 32) * 5/9
    dew_c = (dewpoint_f - 32) * 5/9

    # Saturation vapor pressure (hPa)
    es = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
    e  = 6.112 * np.exp((17.67 * dew_c) / (dew_c + 243.5))

    rh = 100 * (e / es)
    return rh

def compute_wind_chill(temp_f, wind_mph):
    """
    Compute NWS wind chill in °F.
    Valid only when temp <= 50°F and wind >= 3 mph.
    Otherwise return NaN.
    """

    temp_f = temp_f.astype(float)
    wind_mph = wind_mph.astype(float)

    wc = (
        35.74
        + 0.6215 * temp_f
        - 35.75 * (wind_mph ** 0.16)
        + 0.4275 * temp_f * (wind_mph ** 0.16)
    )

    # Mask invalid domains
    mask = (temp_f <= 50) & (wind_mph >= 3)
    return wc.where(mask, np.nan)

def compute_heat_index(temp_f, rh):
    """
    Compute heat index in °F using NOAA regression.
    Only valid when temp >= 80°F and RH >= 40%.
    Else return NaN.
    """

    T = temp_f.astype(float)
    R = rh.astype(float)

    # NOAA regression coefficients
    HI = (
        -42.379
        + 2.04901523*T
        + 10.14333127*R
        - 0.22475541*T*R
        - 6.83783e-3*(T**2)
        - 5.481717e-2*(R**2)
        + 1.22874e-3*(T**2)*R
        + 8.5282e-4*T*(R**2)
        - 1.99e-6*(T**2)*(R**2)
    )

    mask = (T >= 80) & (R >= 40)
    return HI.where(mask, np.nan)

def add_weather_variables(df):
    """
    Given a dataframe with:
        temp_f_mean
        dewpoint_f_mean
        wind_speed_mph_mean

    Add:
        relative_humidity
        wind_chill_f
        heat_index_f
    """

    df = df.copy()

    # Relative humidity (%)
    df["relative_humidity"] = compute_relative_humidity(
        df["temp_f_mean"], df["dewpoint_f_mean"]
    )

    # Wind chill (°F)
    df["wind_chill_f"] = compute_wind_chill(
        df["temp_f_mean"], df["wind_speed_mph_mean"]
    )

    # Heat index (°F)
    df["heat_index_f"] = compute_heat_index(
        df["temp_f_mean"], df["relative_humidity"]
    )

    return df

def plot_temp_windchill_heat(df):
    plt.figure(figsize=(14, 6))

    plt.plot(df.index, df["temp_f_mean"], label="Temperature (F)", linewidth=1.2)
    plt.plot(df.index, df["wind_chill_f"], label="Wind Chill (F)", linewidth=1.2)
    plt.plot(df.index, df["heat_index_f"], label="Heat Index (F)", linewidth=1.2)

    plt.title("Temperature, Wind Chill, Heat Index vs Time")
    plt.xlabel("Time")
    plt.ylabel("°F")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_temp_and_precip(df):
    """
    Dual-axis plot:
      - temp_f_mean (°F) as a line on left axis
      - precip_1h_mm_total (mm) as bars on right axis
    Requires df.index to be a datetime index.
    """

    x = df.index
    temps = df["temp_f_mean"]
    precip = df["precip_1h_mm_total"]

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Left axis: Temperature (line)
    ax1.plot(x, temps, linewidth=1.4, label="Temperature (°F)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Temperature (°F)")
    ax1.tick_params(axis="y")

    # Right axis: Precipitation (bars)
    ax2 = ax1.twinx()
    ax2.bar(x, precip, width=0.02, alpha=0.9, label="Precipitation (mm)", color="tab:red")
    ax2.set_ylabel("Precipitation (mm)")
    ax2.tick_params(axis="y")

    # Legend combining both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    fig.tight_layout()
    plt.show()

# Example direct call when running as a script
if __name__ == "__main__":
    
    input_file = "weather_lga_hourly_agg_2025.parquet"
    df = get_file_info(input_file)
    df_enhanced = add_weather_variables(df)
    plot_temp_windchill_heat(df_enhanced)
    plot_temp_and_precip(df_enhanced)
    
    output_path = weather_path / "weather_lga_hourly_agg_2025_enhanced.parquet"
    df_enhanced.to_parquet(output_path)
    print(f"\nEnhanced file saved to:\n{output_path}\n")
