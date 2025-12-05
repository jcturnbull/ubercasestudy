This directory contains deprecated or superseded Python scripts retained for reference.  
These files are no longer part of the active data ingestion or processing pipeline.

Archived scripts:

- download_weather_hourly.py  
  Earlier version of hourly NOAA weather ingestion using GHCND-style station IDs.  
  Retired because global-hourly requires USAF–WBAN identifiers and returned empty datasets for 2025.  
  Replaced by download_isd_hourly.py in the parent directory.

Notes:
- Archived files are not executed or imported by the active project.
- Files here may be removed in future cleanups once no longer needed for reference.
