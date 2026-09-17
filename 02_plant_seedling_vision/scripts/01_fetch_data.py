"""Fetch the VSB / Aarhus Plant Seedlings dataset via kagglehub and copy the
class folders into data/raw/."""

import os
import shutil

import kagglehub

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")


def main() -> None:
    if os.path.isdir(RAW_DIR) and len(os.listdir(RAW_DIR)) >= 12:
        print(f"Dataset already present at {RAW_DIR} — skipping download.")
        return

    print("Downloading Plant Seedlings dataset via kagglehub ...")
    path = kagglehub.dataset_download("vbookshelf/v2-plant-seedlings-dataset")

    os.makedirs(RAW_DIR, exist_ok=True)
    for entry in os.listdir(path):
        src = os.path.join(path, entry)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(RAW_DIR, entry),
                            dirs_exist_ok=True)

    # The kaggle mirror ships a 13th folder of leftover segmentation masks
    # from the original research pipeline. Not a real class — drop it.
    junk = os.path.join(RAW_DIR, "nonsegmentedv2")
    if os.path.isdir(junk):
        shutil.rmtree(junk)

    classes = sorted(os.listdir(RAW_DIR))
    total = sum(len(os.listdir(os.path.join(RAW_DIR, c))) for c in classes)
    print(f"Organized {total} images across {len(classes)} classes:")
    for c in classes:
        print(f"  {c}: {len(os.listdir(os.path.join(RAW_DIR, c)))}")


if __name__ == "__main__":
    main()
