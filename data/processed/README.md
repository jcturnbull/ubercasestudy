Final analysis-ready datasets. 

Files:
- weather_lga_hourly_agg_2025.parquet
  Hourly aggregated metrics (temp, dewpoint, wind, precipitation), indexed in local NY time.

Notes:
- UTC→America/New_York conversion occurs here during aggregation.
- Missing-hour blocks reflect upstream ISD outages (e.g., Feb 21–28).

