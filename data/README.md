This directory holds all raw, interim, and processed datasets used in the project.

Structure:
- raw/: source data (TLC trips, ISD hourly CSVs, NOAA daily summaries)
- interim/: cleaned hourly weather data (station-level parquet files)
- processed/: final aggregated datasets used in analysis

Recent updates:
- Weather files now use local NYC timestamps after UTC→America/New_York conversion.
- Confirmed Feb 21–28 missing ISD hours as true station outages.
