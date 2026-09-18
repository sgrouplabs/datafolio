"""Auto Collision Risk Matrix — TX territory underwriting dashboard (Streamlit)."""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "tx_collision_risk.csv")

st.set_page_config(page_title="TX Auto Collision Risk Matrix", layout="wide")
st.title("🚗 Auto Collision Risk Matrix — Texas Territory Engine")
st.caption(
    "P&C underwriting tool built on the Kaggle US Accidents dataset (TX subset). "
    "Supports territory risk scoring, pricing-band selection, and weather "
    "multiplier analysis for auto liability and physical damage lines."
)

SEV_COLORS = {1: "#4dd964", 2: "#ffd54f", 3: "#ff9800", 4: "#e53935"}


@st.cache_data(show_spinner="Loading TX accident data ...")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["Start_Time"])
    df["Year"] = df["Start_Time"].dt.year
    return df


df = load_data()

# --- Sidebar filters ---------------------------------------------------------
st.sidebar.header("Filters")
years = st.sidebar.multiselect(
    "Years", sorted(df["Year"].unique(), reverse=True),
    default=sorted(df["Year"].unique(), reverse=True),
)
sev_filter = st.sidebar.multiselect(
    "Severity", sorted(df["Severity"].unique()), default=sorted(df["Severity"].unique())
)
weather_filter = st.sidebar.multiselect(
    "Weather Group", sorted(df["Weather_Group"].unique()),
    default=sorted(df["Weather_Group"].unique()),
)
max_rows = st.sidebar.slider("Max points on map", 2_000, 20_000, 8_000, step=1_000)

view = df[
    df["Year"].isin(years)
    & df["Severity"].isin(sev_filter)
    & df["Weather_Group"].isin(weather_filter)
]

# --- KPI row -----------------------------------------------------------------
n = len(view)
w, sev = view["Weather_Group"], view["Severity"]
clear_share = w.value_counts(normalize=True).get("Clear", 0.0)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Accidents (filtered)", f"{n:,}")
k2.metric("Avg Severity", f"{sev.mean():.2f} / 4")
k3.metric("Sev 3–4 Share", f"{(sev >= 3).mean() * 100:.1f}%")
k4.metric("Clear-weather share", f"{clear_share * 100:.1f}%")

tab_map, tab_time, tab_weather, tab_city = st.tabs(
    ["🗺️ Geospatial Risk Map", "⏰ Temporal Risk Heatmap",
     "🌧️ Weather Multiplier Analysis", "🏙️ Territory Ranking"]
)

with tab_map:
    st.markdown(
        "High-severity clusters mark candidate **high-rate territories**. "
        "Hover a point for severity, weather, and city context."
    )
    sample = view.sample(n=min(max_rows, len(view)), random_state=42) if n else view
    fig = px.scatter_map(
        sample, lat="Start_Lat", lon="Start_Lng", color="Severity",
        color_discrete_map=SEV_COLORS, category_orders={"Severity": [1, 2, 3, 4]},
        zoom=4.6, center={"lat": 31.2, "lon": -99.3}, height=560,
        hover_data={"Start_Lat": False, "Start_Lng": False,
                    "City": True, "Weather_Group": True},
    )
    fig.update_layout(map_style="carto-positron", margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

with tab_time:
    st.markdown(
        "Day-of-week × hour-of-day frequency. Rush-hour bands (7–9 AM, 4–7 PM) "
        "drive exposure assumptions in frequency-based rating plans."
    )
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = (
        view.pivot_table(index="DayOfWeek", columns="Hour", values="Severity",
                         aggfunc="count", fill_value=0)
        .reindex(days)
    )
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=[f"{h:02d}" for h in pivot.columns], y=days,
        colorscale="YlOrRd", colorbar=dict(title="Accidents"),
    ))
    fig.update_layout(height=520, xaxis_title="Hour of day", yaxis_title="Day of week")
    st.plotly_chart(fig, use_container_width=True)

with tab_weather:
    st.markdown(
        "Average severity by weather group, benchmarked against Clear conditions. "
        "The **severity index ratio** (group avg ÷ Clear avg) approximates the "
        "multiplier a ratemaking analyst would apply to that environmental class."
    )
    agg = (
        view.groupby("Weather_Group")
        .agg(accidents=("Severity", "size"), avg_severity=("Severity", "mean"))
        .reset_index()
    )
    clear_sev = agg.loc[agg["Weather_Group"] == "Clear", "avg_severity"]
    baseline = float(clear_sev.iloc[0]) if len(clear_sev) else 1.0
    agg["severity_index"] = (agg["avg_severity"] / baseline).round(3)
    agg["freq_share"] = (agg["accidents"] / n * 100).round(2)
    agg = agg.sort_values("avg_severity", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            agg, x="Weather_Group", y="avg_severity", color="severity_index",
            color_continuous_scale="RdYlGn_r", range_color=[0.8, 1.4],
            title="Average Severity by Weather Group", text="avg_severity",
        )
        fig.update_traces(texttemplate="%{text:.2f}")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(
            agg, x="Weather_Group", y="severity_index",
            title="Severity Index vs Clear Baseline (rating multiplier proxy)",
            color="severity_index", color_continuous_scale="RdYlGn_r",
            range_color=[0.8, 1.4], text="severity_index",
        )
        fig.update_traces(texttemplate="%{text:.2f}")
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(agg.rename(columns={
        "accidents": "Accidents", "avg_severity": "Avg Severity",
        "severity_index": "Severity Index", "freq_share": "Frequency Share (%)",
    }), use_container_width=True, hide_index=True)

with tab_city:
    st.markdown(
        "Territory ranking by accident volume and severity — a starting point for "
        "defining **territory pricing bands** (e.g., metro core vs exurban ring)."
    )
    top = st.slider("Top N cities", 10, 50, 20)
    city = (
        view.groupby("City")
        .agg(accidents=("Severity", "size"), avg_severity=("Severity", "mean"))
        .sort_values("accidents", ascending=False)
        .head(top)
        .reset_index()
    )
    fig = px.bar(
        city, x="City", y="accidents", color="avg_severity",
        color_continuous_scale="YlOrRd",
        title=f"Top {top} TX Cities — Volume (bar) × Severity (color)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(city, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Source: Sobhan Moosavi, US Accidents (2016–2023), Kaggle. Filtered to Texas. "
    "Severity 1 = least impact on traffic, 4 = most severe."
)
