This directory contains active Python scripts for downloading, cleaning, and transforming TLC and NOAA/ISD data.

Files:

- config.py
  Centralized paths and settings for raw, interim, and processed data directories.

- download_tlc.py
  Fetches monthly NYC TLC HVFHV trip parquet files into data/raw/tlc/.

- download_weather_noaa.py
  Downloads NOAA daily summaries (GHCND) for JFK, LGA, and Central Park.

- download_isd_hourly.py
  Downloads hourly ISD (global-hourly) station data and saves raw CSVs.

- weather_build_interim.py
  Converts raw ISD files into cleaned hourly parquet files (still in UTC).

- weather_aggregate_hourly.py
  Aggregates interim weather to hourly metrics and converts timestamps from UTC to America/New_York (DST-aware).

- weather_eda.py
  Quick exploration utilities for inspecting weather completeness and anomalies.

archive/
  Older or deprecated scripts, including prior aggregation logic and diagnostics.

