"""Fetch the US Accidents (2016-2023) dataset from Kaggle and stage it in data/raw/."""

import os

import kagglehub

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")


def main() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    if os.path.exists(os.path.join(RAW_DIR, "US_Accidents_March23.csv")):
        print("Raw CSV already staged — skipping download.")
        return

    print("Downloading us-accidents via kagglehub (large, >3GB) ...")
    path = kagglehub.dataset_download("sobhanmoosavi/us-accidents")

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
