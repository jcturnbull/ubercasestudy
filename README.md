# Uber Case Study

Project scaffold.

Public-data analysis of how weather and other conditions affect HVFHV (Uber/Lyft) demand at LaGuardia Airport in 2025.  
The project uses NYC TLC trip records and NOAA ISD/GHCND weather data to build hourly, analysis-ready datasets.

## Structure

- `code/`  
  Scripts for downloading TLC/NOAA data, cleaning it, constructing interim datasets, and producing hourly aggregates.

- `data/`  
  Raw source files, cleaned intermediates, and final processed outputs.

- `notebooks/`  
  Exploratory work, planning notes, and analysis steps.

- `reports/`  
  Draft write-ups, limitations, and presentation materials.

- `references/`  
  External documentation used to interpret TLC and NOAA fields.

## Pipeline summary

1. Download raw HVFHV and weather data.
2. Clean TLC records (discard rules, airport filtering).
3. Clean and standardize hourly weather (UTC → America/New_York).
4. Aggregate both datasets to hourly resolution.
5. Construct derived variables (wind chill, heat index, etc.).
6. Produce final joined inputs for modeling and hypothesis testing.

The repository is designed for reproducibility: no manual edits to interim or processed data.
