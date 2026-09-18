"""Filter US Accidents to Texas and engineer actuarial risk features, chunked.

The raw file is >3GB, so we stream it in chunks rather than loading it whole.
Output: data/processed/tx_collision_risk.csv
"""

import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_PATH = os.path.join(BASE_DIR, "data", "processed", "tx_collision_risk.csv")

# Weather conditions collapse into a small set of rating groups.
WEATHER_MAP = {
    "clear": "Clear", "fair": "Clear",
    "cloudy": "Cloudy", "mostly cloudy": "Cloudy", "partly cloudy": "Cloudy",
    "overcast": "Cloudy", "scattered clouds": "Cloudy", "few clouds": "Cloudy",
    "broken clouds": "Cloudy",
    "rain": "Rain", "light rain": "Rain", "heavy rain": "Rain",
    "light rain shower": "Rain", "heavy rain shower": "Rain",
    "rain shower": "Rain", "drizzle": "Rain", "light drizzle": "Rain",
    "thunderstorm": "Storm", "thunderstorms and rain": "Storm",
    "t-storm": "Storm", "heavy thunderstorm": "Storm",
    "thunder in the vicinity": "Storm", "light thunderstorm rain": "Storm",
    "snow": "Snow", "light snow": "Snow", "heavy snow": "Snow",
    "snow grains": "Snow", "light snow showers": "Snow",
    "fog": "Fog", "haze": "Fog", "mist": "Fog", "shallow fog": "Fog",
    "smoke": "Fog", "fog patches": "Fog", "light haze": "Fog",
    "wintry mix": "Snow", "ice pellets": "Snow", "light ice pellets": "Snow",
    "sleet": "Snow", "freezing rain": "Snow", "light freezing rain": "Snow",
    "hail": "Storm", "volcanic ash": "Fog",
}


def main() -> None:
    csvs = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
    if not csvs:
        raise FileNotFoundError("No raw CSV found — run 01_kaggle_extract.py first")
    raw_path = os.path.join(RAW_DIR, csvs[0])

    chunks = []
    total = 0
    for chunk in pd.read_csv(raw_path, chunksize=500_000, low_memory=False):
        tx = chunk[chunk["State"] == "TX"].copy()
        total += len(chunk)
        if tx.empty:
            continue
        tx["Start_Time"] = pd.to_datetime(tx["Start_Time"], errors="coerce", format="mixed")
        tx = tx.dropna(subset=["Start_Time", "Start_Lat", "Start_Lng", "Severity"])
        tx["Hour"] = tx["Start_Time"].dt.hour
        tx["DayOfWeek"] = tx["Start_Time"].dt.day_name()
        tx["Month"] = tx["Start_Time"].dt.month
        cond = tx["Weather_Condition"].fillna("Unknown").str.strip().str.lower()
        tx["Weather_Group"] = cond.map(WEATHER_MAP).fillna("Other")
        keep = ["Severity", "Start_Lat", "Start_Lng", "Start_Time", "Hour",
                "DayOfWeek", "Month", "Weather_Group", "City", "Temperature(F)",
                "Visibility(mi)", "Humidity(%)", "Wind_Speed(mph)"]
        chunks.append(tx[keep])
        print(f"processed {total:,} rows, kept {len(chunks[-1]):,} TX rows", flush=True)

    df = pd.concat(chunks, ignore_index=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df):,} TX accidents -> {OUT_PATH}")


if __name__ == "__main__":
    main()
