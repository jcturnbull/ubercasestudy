# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 17:57:05 2025

@author: epicx

tlc_combine_cleaned.py

Combine multiple monthly cleaned TLC HVFHV parquet files into one dataset.

Usage (Spyder):
    - Edit MONTHS_TO_COMBINE in __main__
    - Run with F5
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"C:/Users/epicx/Projects/ubercasestudy")

INTERIM_DIR = PROJECT_ROOT / "data/interim/tlc"


def combine_months(month_codes):
    """
    Combine files like fhvhv_tripdata_2025-<mm>_clean.parquet
    into a single master DataFrame, saved in the interim folder.
    """
    dfs = []
    for m in month_codes:
        m = f"{int(m):02d}"   # normalize to two digits
        fp = INTERIM_DIR / f"fhvhv_tripdata_2025-{m}_clean.parquet"

        if not fp.exists():
            print(f"[WARN] Missing file: {fp}")
            continue

        print(f"Reading {fp} ...")
        df = pd.read_parquet(fp)
        dfs.append(df)

    if not dfs:
        raise ValueError("No monthly cleaned files were found to combine.")

    combined = pd.concat(dfs, ignore_index=True)

    # Optional but recommended: sort by pickup time
    if "pickup_datetime" in combined.columns:
        combined = combined.sort_values("pickup_datetime").reset_index(drop=True)

    out_fp = INTERIM_DIR / "fhvhv_tripdata_2025_all_clean.parquet"
    print(f"Saving combined dataset to {out_fp} ...")
    combined.to_parquet(out_fp, index=False)

    print("Done.")
    return combined


def main(months):
    return combine_months(months)


if __name__ == "__main__":
    MONTHS_TO_COMBINE = ["01", "02", "03", "04", "05", "06", "07", "08"]
    main(MONTHS_TO_COMBINE)
