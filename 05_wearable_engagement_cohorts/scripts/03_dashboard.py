#!/usr/bin/env python3
"""
03_dashboard.py — Internal product analytics dashboard (Streamlit).

Displays:
  • Daily Active Users (DAU) trend line
  • Retention cohort heatmap (triangle)
  • Feature engagement breakdown (steps / sleep / heart rate)

Run:  streamlit run scripts/03_dashboard.py
"""
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJECT_ROOT, "data", "processed")

st.set_page_config(page_title="Wearable Engagement & Cohorts", layout="wide")


@st.cache_data
def load_data():
    events = pd.read_csv(os.path.join(OUT, "user_events.csv"), parse_dates=["date"])
    cohorts = pd.read_csv(os.path.join(OUT, "user_cohorts.csv"), parse_dates=["cohort_date"])
    matrix = pd.read_csv(os.path.join(OUT, "cohort_retention.csv"))
    return events, cohorts, matrix


events, cohorts, matrix = load_data()

# ───────────────────────── Header / KPIs ─────────────────────────
st.title("⌚ Wearable Engagement & Cohort Tracker")
st.caption("FitBit telemetry → DAU, retention cohorts, and feature engagement")

dau = events.groupby("date")["Id"].nunique().rename("dau").reset_index()
n_users = events["Id"].nunique()
avg_dau = dau["dau"].mean()
d30 = cohorts["day_30_retained"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Users", n_users)
c2.metric("Avg DAU", f"{avg_dau:.1f}")
c3.metric("Day-30 Retention", f"{d30:.0%}")
c4.metric("Event Days", f"{len(events):,}")

# ───────────────────────── DAU trend ─────────────────────────
st.subheader("Daily Active Users")
dau_fig = px.area(
    dau, x="date", y="dau",
    labels={"date": "Date", "dau": "Unique users logging activity"},
)
dau_fig.update_traces(line_color="#2E86AB")
st.plotly_chart(dau_fig, use_container_width=True)

# ───────────────────────── Retention heatmap ─────────────────────────
st.subheader("Retention Cohort Heatmap")
st.caption("Percentage of each weekly signup cohort logging any event (±2 day window)")

pivot = matrix.pivot_table(
    index="cohort_week", columns="day_offset", values="retention_pct"
)
heat = go.Figure(go.Heatmap(
    z=pivot.values,
    x=[f"Day {int(d)}" for d in pivot.columns],
    y=pivot.index,
    colorscale="RdYlGn",
    zmin=0, zmax=100,
    text=pivot.applymap(lambda v: "" if pd.isna(v) else f"{v:.0f}%").values,
    texttemplate="%{text}",
    textfont={"size": 11},
))
heat.update_layout(
    xaxis_title="Days since first activity", yaxis_title="Cohort (ISO week)",
    yaxis={"autorange": "reversed"},
)
st.plotly_chart(heat, use_container_width=True)

# ───────────────────────── Feature engagement ─────────────────────────
st.subheader("Feature Engagement")
feature_users = (
    events.groupby("feature")["Id"].nunique().rename("users").reset_index()
)
feature_days = (
    events.groupby("feature").size().rename("event_days").reset_index()
)
features = feature_users.merge(feature_days, on="feature").sort_values(
    "users", ascending=False
)
features["adoption_pct"] = (100 * features["users"] / n_users).round(1)

f1, f2 = st.columns(2)
with f1:
    bar = px.bar(
        features, x="feature", y="users", text="users",
        color="adoption_pct", color_continuous_scale="Blues",
        labels={"feature": "Feature", "users": "Users engaging"},
    )
    bar.update_layout(showlegend=False)
    st.plotly_chart(bar, use_container_width=True)
with f2:
    st.dataframe(features, hide_index=True, use_container_width=True)

st.info(
    f"Step tracking is the default engagement surface ({features.iloc[0]['adoption_pct']}% "
    "of users); sleep tracking and heart-rate monitoring are opt-in features with "
    "measurably lower adoption — a clear upgrade-funnel opportunity."
)
