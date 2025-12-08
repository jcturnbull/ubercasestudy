# Code

Active Python scripts for downloading, cleaning, and transforming TLC and NOAA/ISD data.

Files:

- `config.py`  
  Centralized paths and settings for raw, interim, and processed data directories.

- `download_tlc.py`  
  Fetches monthly NYC TLC HVFHV trip parquet files into `data/raw/tlc/`.

- `download_weather_noaa.py`  
  Downloads NOAA GHCND daily summaries for JFK, LGA, and Central Park.

- `download_isd_hourly.py`  
  Downloads hourly ISD (global-hourly) station data and saves raw CSVs.

- `weather_build_interim.py`  
  Converts raw ISD files into cleaned hourly parquet files (still UTC).

- `weather_aggregate_hourly.py`  
  Aggregates interim weather to hourly metrics and converts timestamps to America/New_York (DST-aware).

- `weather_variable_construction.py`  
  Adds derived features such as wind chill and heat index to the aggregated weather dataset.

- `weather_eda.py`  
  Lightweight tools for sanity-checking completeness, anomalies, and timestamp alignment.

- `tlc_processor.py`  
  Cleans raw HVFHV trip data, applies discard rules, filters to airport trips, and writes monthly `_clean.parquet` files.

- `tlc_combine_cleaned.py`  
  Combines monthly cleaned TLC outputs into a single `fhvhv_tripdata_2025_all_clean.parquet`.

- `tlc_aggregate_hourly.py`  
  Builds hourly LGA demand metrics from cleaned TLC data (counts, fares, speed, etc.).

archive/  
  Deprecated or superseded scripts retained for reference (old loaders, diagnostic utilities, prior aggregation logic).
