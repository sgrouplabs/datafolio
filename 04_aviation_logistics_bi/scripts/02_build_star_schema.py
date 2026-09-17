"""Build a star schema from the raw flight data.

Cleans the BTS-format extract and emits four tables to data/processed/:
Dim_Date, Dim_Airline, Dim_Airport, and Fact_Flights, joined by surrogate
keys so Power BI can model them with one-to-many relationships.
"""

import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "flight_data_2024_sample.csv")
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")

DAY_NAMES = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday",
             5: "Thursday", 6: "Friday", 7: "Saturday"}


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(RAW_PATH)
    n_raw = len(df)

    # ---- Cleaning --------------------------------------------------------
    # Cancellations have no actual times; delay columns are NaN there and
    # that's real information, not missingness to impute.
    for col in ["dep_delay", "arr_delay", "air_time", "taxi_out", "taxi_in"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["fl_date"] = pd.to_datetime(df["fl_date"])
    df["cancelled"] = df["cancelled"].fillna(0).astype(int)
    df["diverted"] = df["diverted"].fillna(0).astype(int)

    # Drop exact duplicate flight records if any.
    dupe_keys = ["fl_date", "op_unique_carrier", "op_carrier_fl_num",
                 "origin", "dest", "crs_dep_time"]
    before = len(df)
    df = df.drop_duplicates(subset=dupe_keys)
    n_dupes = before - len(df)

    # ---- Dim_Date --------------------------------------------------------
    dates = df["fl_date"].drop_duplicates().sort_values()
    dim_date = pd.DataFrame({"date_key": dates.dt.strftime("%Y%m%d").astype(int),
                             "full_date": dates.dt.date,
                             "year": dates.dt.year,
                             "quarter": dates.dt.quarter,
                             "month": dates.dt.month,
                             "month_name": dates.dt.strftime("%B"),
                             "day_of_month": dates.dt.day,
                             "day_of_week": dates.dt.dayofweek + 1,
                             "day_name": (dates.dt.dayofweek + 1)
                                          .map(DAY_NAMES)})
    date_lookup = dict(zip(dim_date["full_date"], dim_date["date_key"]))

    # ---- Dim_Airline -----------------------------------------------------
    airlines = (df[["op_unique_carrier"]].drop_duplicates()
                .sort_values("op_unique_carrier"))
    airlines["airline_key"] = range(1, len(airlines) + 1)
    # The sample carries no carrier-name column; use the BTS code as the
    # display name. A production build would join the DOT lookup table here.
    airlines["airline_name"] = airlines["op_unique_carrier"]
    airline_lookup = dict(zip(airlines["op_unique_carrier"],
                              airlines["airline_key"]))

    # ---- Dim_Airport (origin + destination in one conformed dimension) ---
    origins = df[["origin", "origin_city_name", "origin_state_nm"]].rename(
        columns={"origin": "airport_code", "origin_city_name": "city",
                 "origin_state_nm": "state"})
    dests = df[["dest", "dest_city_name", "dest_state_nm"]].rename(
        columns={"dest": "airport_code", "dest_city_name": "city",
                 "dest_state_nm": "state"})
    airports = pd.concat([origins, dests]).drop_duplicates("airport_code") \
                   .sort_values("airport_code").reset_index(drop=True)
    airports["airport_key"] = range(1, len(airports) + 1)
    airport_lookup = dict(zip(airports["airport_code"],
                              airports["airport_key"]))

    # ---- Fact_Flights ----------------------------------------------------
    fact = df.copy()
    fact["date_key"] = fact["fl_date"].dt.date.map(date_lookup)
    fact["airline_key"] = fact["op_unique_carrier"].map(airline_lookup)
    fact["origin_key"] = fact["origin"].map(airport_lookup)
    fact["dest_key"] = fact["dest"].map(airport_lookup)

    assert fact[["date_key", "airline_key", "origin_key", "dest_key"]] \
        .notna().all().all(), "unmapped foreign key found"

    fact_cols = ["flight_id", "date_key", "airline_key", "origin_key",
                 "dest_key", "op_carrier_fl_num", "crs_dep_time",
                 "dep_time", "dep_delay", "crs_arr_time", "arr_time",
                 "arr_delay", "taxi_out", "taxi_in", "crs_elapsed_time",
                 "actual_elapsed_time", "air_time", "distance",
                 "carrier_delay", "weather_delay", "nas_delay",
                 "security_delay", "late_aircraft_delay",
                 "cancelled", "diverted", "cancellation_code"]
    fact["flight_id"] = range(1, len(fact) + 1)
    fact_out = fact[fact_cols]

    dim_date.to_csv(os.path.join(OUT_DIR, "Dim_Date.csv"), index=False)
    airlines.to_csv(os.path.join(OUT_DIR, "Dim_Airline.csv"), index=False)
    airports.to_csv(os.path.join(OUT_DIR, "Dim_Airport.csv"), index=False)
    fact_out.to_csv(os.path.join(OUT_DIR, "Fact_Flights.csv"), index=False)

    print(f"Raw rows: {n_raw} | duplicates dropped: {n_dupes}")
    print(f"Dim_Date:    {len(dim_date):>6} rows")
    print(f"Dim_Airline: {len(airlines):>6} rows")
    print(f"Dim_Airport: {len(airports):>6} rows")
    print(f"Fact_Flights:{len(fact_out):>6} rows")


if __name__ == "__main__":
    main()
