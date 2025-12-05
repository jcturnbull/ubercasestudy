This directory contains all active Python source files used for data ingestion, preprocessing, and metric generation for the project.

Current active scripts:

- config.py  
  Defines project paths (raw, interim, processed directories) and initializes required folder structure.

- download_tlc.py  
  Downloads NYC TLC HVFHV monthly trip data into data/raw/tlc/.  
  Uses public CloudFront URLs for fhvhv_tripdata_YYYY-MM.parquet.

- download_weather_noaa.py  
  Downloads NOAA daily-summaries (GHCND) weather data for JFK, LGA, and Central Park using GHCND station IDs.

- download_isd_hourly.py  
  Downloads hourly ISD (global-hourly) weather data using USAF–WBAN station identifiers (hyphens removed when saved).  
  Produces raw hourly CSVs for each station in data/raw/weather/.

- load_isd_hourly.py  
  Parses raw ISD CSVs, normalizes column names, constructs datetime index, and returns combined hourly weather DataFrames.

Other notes:
- Only actively maintained code should remain in this directory.
- Deprecated or superseded files are moved into the archive/ subfolder.
