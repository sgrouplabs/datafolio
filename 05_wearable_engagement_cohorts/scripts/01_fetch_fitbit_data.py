#!/usr/bin/env python3
"""
01_fetch_fitbit_data.py — Download the FitBit Fitness Tracker Data (Mobius)
from Kaggle via kagglehub and stage the raw CSVs into data/raw/.

Usage:
    python scripts/01_fetch_fitbit_data.py
"""
import os
import shutil

import kagglehub

DATASET_ID = "arashnic/fitbit"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

FOCUS_FILES = [
    "dailyActivity_merged.csv",
    "sleepDay_merged.csv",
    "heartrate_seconds_merged.csv",
]


def main() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"Fetching Kaggle dataset '{DATASET_ID}' via kagglehub...")
    path = kagglehub.dataset_download(DATASET_ID)
    print(f"Downloaded and extracted to: {path}")

    copied, found = [], []
    for root, _dirs, files in os.walk(path):
        for fname in files:
            if fname.endswith(".csv"):
                found.append(fname)
                shutil.copy2(os.path.join(root, fname), os.path.join(RAW_DIR, fname))
                if fname in FOCUS_FILES:
                    copied.append(fname)

    print(f"Staged {len(found)} CSV files into {RAW_DIR}")
    print("Focus files present:", copied)


if __name__ == "__main__":
    main()
