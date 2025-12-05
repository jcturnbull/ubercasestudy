# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 22:43:59 2025

@author: epicx
"""

import sys
from pathlib import Path
from typing import Iterable
import requests

from config import TLC_RAW_DIR

def default_tlc_url(service: str, year: int, month: int) -> str:
    m = f"{month:02d}"
    BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
    filename = f"{service}_tripdata_{year}-{m}.parquet"
    return f"{BASE_URL}/{filename}"


def download_file(url: str, dest_path: Path, chunk_size: int = 1 << 20) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest_path}")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()

    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    with tmp_path.open("wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    tmp_path.replace(dest_path)
    print(f"Saved {dest_path}")


def download_tlc_months(
    service: str,
    year: int,
    months: Iterable[int],
    url_builder=default_tlc_url,
) -> None:
    """
    service: e.g. 'fhvhv', 'yellow', 'green'
    year: e.g. 2023
    months: iterable of ints, 1–12
    """
    for m in months:
        url = url_builder(service, year, m)
        filename = Path(url).name  # keep same filename
        dest = TLC_RAW_DIR / filename

        if dest.exists():
            print(f"Already exists, skipping: {dest}")
            continue

        try:
            download_file(url, dest)
        except requests.HTTPError as e:
            print(f"HTTP error for {url}: {e}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")


if __name__ == "__main__":
    # Quick CLI usage: python code/download_tlc.py fhvhv 2023 1 2 3
    if len(sys.argv) < 4:
        print("Usage: python download_tlc.py <service> <year> <month1> [<month2> ...]")
        sys.exit(1)

    svc = sys.argv[1]
    yr = int(sys.argv[2])
    months = [int(x) for x in sys.argv[3:]]
    download_tlc_months(svc, yr, months)
