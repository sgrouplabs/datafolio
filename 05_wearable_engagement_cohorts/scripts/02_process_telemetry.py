#!/usr/bin/env python3
"""
02_process_telemetry.py — Clean and join raw FitBit telemetry, sessionize
events, and build Day-0 / Day-7 / Day-30 retention cohorts.

Inputs (data/raw/):
  dailyActivity_merged.csv, sleepDay_merged.csv, heartrate_seconds_merged.csv

Output (data/processed/):
  user_events.csv       — tidy event stream (one row per user-day per feature)
  user_cohorts.csv      — per-user cohort assignment + retention flags
  cohort_retention.csv  — retention matrix (Day0..Day30) for the heatmap
"""
import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(PROJECT_ROOT, "data", "raw")
OUT = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(OUT, exist_ok=True)


def load_events() -> pd.DataFrame:
    """Standardize the three telemetry sources into one tidy event table."""
    act = pd.read_csv(os.path.join(RAW, "dailyActivity_merged.csv"))
    act["date"] = pd.to_datetime(act["ActivityDate"], format="%m/%d/%Y")

    sleep = pd.read_csv(os.path.join(RAW, "sleepDay_merged.csv"))
    sleep["date"] = pd.to_datetime(
        sleep["SleepDay"], format="%m/%d/%Y %I:%M:%S %p"
    ).dt.normalize()

    hr = pd.read_csv(os.path.join(RAW, "heartrate_seconds_merged.csv"))
    hr["date"] = pd.to_datetime(hr["Time"], format="%m/%d/%Y %I:%M:%S %p").dt.normalize()

    # --- sessionization: each event stream becomes an (Id, date) activity session
    activity_sessions = (
        act.groupby(["Id", "date"], as_index=False)
        .agg(
            activity_events=("Id", "size"),
            total_steps=("TotalSteps", "sum"),
            calories=("Calories", "sum"),
            very_active_minutes=("VeryActiveMinutes", "sum"),
        )
    )
    activity_sessions["feature"] = "step_tracking"

    sleep_sessions = (
        sleep.groupby(["Id", "date"], as_index=False)
        .agg(
            sleep_records=("TotalSleepRecords", "sum"),
            minutes_asleep=("TotalMinutesAsleep", "sum"),
            time_in_bed=("TotalTimeInBed", "sum"),
        )
    )
    sleep_sessions["feature"] = "sleep_tracking"

    # heart rate: a user-day counts as engaged if >1000 second-level readings
    hr_counts = hr.groupby(["Id", "date"]).size().rename("hr_readings").reset_index()
    hr_sessions = hr_counts[hr_counts["hr_readings"] > 1000].copy()
    hr_sessions["feature"] = "heart_rate_monitoring"

    # join extra numeric columns into a shared schema
    events = pd.concat(
        [
            activity_sessions.assign(
                sleep_records=np.nan, minutes_asleep=np.nan, time_in_bed=np.nan,
                hr_readings=np.nan,
            ),
            sleep_sessions.assign(
                activity_events=np.nan, total_steps=np.nan, calories=np.nan,
                very_active_minutes=np.nan, hr_readings=np.nan,
            ),
            hr_sessions.assign(
                activity_events=np.nan, total_steps=np.nan, calories=np.nan,
                very_active_minutes=np.nan, sleep_records=np.nan,
                minutes_asleep=np.nan, time_in_bed=np.nan,
            ),
        ],
        ignore_index=True,
    )[["Id", "date", "feature", "total_steps", "calories", "very_active_minutes",
       "sleep_records", "minutes_asleep", "time_in_bed", "hr_readings"]]

    events["Id"] = events["Id"].astype(str)
    events = events.sort_values(["Id", "date", "feature"]).reset_index(drop=True)
    return events


def build_cohorts(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign each user a Day-0 cohort (first activity/sleep log date) and
    compute Day-0/7/14/21/30 retention flags plus the retention matrix."""
    # Day 0 = first date the user logged ANY activity or sleep event
    first_day = events.groupby("Id")["date"].min().rename("cohort_date").reset_index()
    users = events.merge(first_day, on="Id")
    users["day_offset"] = (users["date"] - users["cohort_date"]).dt.days

    # per-user activity-day matrix: 1 if user logged any feature on that offset
    active_days = (
        users.groupby(["Id", "day_offset"]).size().rename("events").reset_index()
    )

    def retained(day: int, window: int = 2) -> pd.Series:
        """Retained if the user logged an event within ±window days of `day`."""
        d = active_days.assign(hit=((active_days["day_offset"] - day).abs() <= window))
        return d.groupby("Id")["hit"].max()

    # weekly cohorts: assign each user to the ISO week of their Day-0 date
    first_day["cohort_week"] = (
        first_day["cohort_date"].dt.strftime("%G-W%V")
    )
    cohorts = first_day.copy()

    for day in [0, 7, 14, 21, 30]:
        cohorts[f"day_{day}_retained"] = cohorts["Id"].map(retained(day)).fillna(False)

    # engagement summary per user
    summary = events.pivot_table(
        index="Id", columns="feature", values="date", aggfunc="count"
    ).fillna(0).astype(int).reset_index()
    cohorts = cohorts.merge(summary, on="Id", how="left")
    cohorts["n_active_days"] = (
        active_days.groupby("Id").size().reindex(cohorts["Id"]).values
    )

    # retention matrix: % of cohort members retained at each offset
    offsets = [0, 1, 2, 3, 5, 7, 10, 14, 17, 21, 24, 27, 30]
    rows = []
    for week, grp in cohorts.groupby("cohort_week"):
        n_users = len(grp)
        for day in offsets:
            r = retained(day)
            wk_ids = set(grp["Id"])
            wk_ret = r[r.index.isin(wk_ids)]
            rows.append({
                "cohort_week": week,
                "cohort_size": n_users,
                "day_offset": day,
                "retained_users": int(wk_ret.sum()) if len(wk_ret) else 0,
                "retention_pct": round(100 * wk_ret.mean(), 1) if len(wk_ret) else np.nan,
            })
    matrix = pd.DataFrame(rows)
    # only offsets where the cohort has had enough time to be observed
    last_date = users["date"].max()
    keep = []
    for _, r in matrix.iterrows():
        cohort_start = cohorts.loc[cohorts["cohort_week"] == r["cohort_week"], "cohort_date"].min()
        if (last_date - cohort_start).days >= r["day_offset"] - 2:
            keep.append(True)
        else:
            keep.append(False)
    matrix = matrix[pd.Series(keep, index=matrix.index)]
    return cohorts, matrix


def main() -> None:
    events = load_events()
    print(f"Events: {len(events):,} rows | users: {events['Id'].nunique()}"
          f" | date range: {events['date'].min().date()} → {events['date'].max().date()}")

    cohorts, matrix = build_cohorts(events)
    print(f"Cohorts: {len(cohorts)} users")
    print(matrix.to_string(index=False))

    events.to_csv(os.path.join(OUT, "user_events.csv"), index=False)
    cohorts.to_csv(os.path.join(OUT, "user_cohorts.csv"), index=False)
    matrix.to_csv(os.path.join(OUT, "cohort_retention.csv"), index=False)
    print(f"Saved outputs to {OUT}")


if __name__ == "__main__":
    main()
