# Processed Data

Final analysis-ready datasets.

Files:

- `weather_lga_hourly_agg_2025.parquet`  
  Hourly aggregated weather metrics (temp, dewpoint, wind, precipitation), indexed in America/New_York.

- `weather_lga_hourly_agg_2025_enhanced.parquet`  
  Same as above but with derived fields added (wind chill, heat index, and other constructed variables).

Notes:
- UTC → America/New_York conversion occurs during aggregation.
- Missing-hour gaps reflect upstream ISD outages (e.g., Feb 21–28).
- These outputs feed directly into demand/weather modeling and hourly joins with TLC data.
