# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 21:45:27 2025

@author: epicx
"""

from pathlib import Path

# Project root = parent of this file
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

TLC_RAW_DIR = RAW_DIR / "tlc"
WEATHER_RAW_DIR = RAW_DIR / "weather"
FLIGHTS_RAW_DIR = RAW_DIR / "flights"

for p in [TLC_RAW_DIR, WEATHER_RAW_DIR, FLIGHTS_RAW_DIR, INTERIM_DIR, PROCESSED_DIR]:
    p.mkdir(parents=True, exist_ok=True)