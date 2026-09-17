"""Fetch the Flight Delay Dataset (2024) from Kaggle and stage it in data/raw/.
"""

import os

import kagglehub

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")


def main() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    existing = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
    if existing:
        print(f"CSV already present at {RAW_DIR}/{existing[0]} — skipping.")
        return

    print("Downloading flight-data-2024 via kagglehub ...")
    path = kagglehub.dataset_download("hrishitpatil/flight-data-2024")

    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".csv"):
                dest = os.path.join(RAW_DIR, f)
                if not os.path.exists(dest):
                    os.link(os.path.join(root, f), dest)
                print(f"Staged {f} -> data/raw/")
                return

    raise FileNotFoundError("No CSV found in the downloaded dataset")


if __name__ == "__main__":
    main()
