# 05 · Wearable Engagement & Cohort Tracker 📊

[![Live Demo](https://img.shields.io/badge/Demo-Live%20Streamlit%20WASM-brightgreen)](https://sgrouplabs.github.io/datafolio/wearables/)

A **product analytics** project built on the FitBit Fitness Tracker dataset (Mobius, Kaggle): raw timestamped wearable telemetry is transformed into the KPIs a Product Analyst actually works with — Daily Active Users (DAU), retention cohorts, and feature engagement.

## What this dashboard answers

| Product question | KPI | Where |
|---|---|---|
| "How many users are actively logging every day?" | DAU trend | DAU area chart |
| "Do users come back after onboarding?" | Day-0/7/14/21/30 retention by signup cohort | Cohort heatmap (triangle) |
| "Which features drive engagement?" | Adoption % per feature (steps / sleep / heart rate) | Feature engagement bars |

## Methodology

1. **Event schemas & normalization** — Three heterogeneous raw sources (`dailyActivity`, `sleepDay`, `heartrate_seconds`) use different timestamp formats (`M/D/YYYY` vs `M/D/YYYY H:MM:SS AM/PM`). Each is parsed with its explicit format and normalized to a user-day (`Id`, `date`) grain, producing a tidy **event stream** of 1,003 user-day-feature events across 35 users (Mar 12 – May 12, 2016).
2. **Sessionization** — Each user-day is treated as a session. Heart-rate sessions require >1,000 second-level readings per day to count as genuine engagement (not a wrist-on blip).
3. **Cohort assignment** — A user's **Day 0** is the first date they log *any* activity or sleep event. Users are grouped into ISO-week signup cohorts (`2016-W10`, `W12`, `W13`).
4. **Retention definition** — A user is *retained at Day N* if they log any event within a ±2-day window of Day N (wearables have natural off-wrist gaps, so strict same-day matching overstates churn). Cells are masked where the cohort hasn't had enough calendar time to reach the offset.
5. **Feature engagement** — Adoption is computed as distinct users engaging per feature over total users, plus engagement-days per feature.

### Headline results

- **35 users**, avg DAU ≈ 33; strong early activity across the 8-week window
- **Day-30 retention ≈ 60%** (±2-day window definition)
- **Feature adoption funnel:** step tracking 100% → sleep tracking 69% → heart-rate monitoring 37% — sleep and HR are clear upgrade-funnel targets

## Reproduce

```bash
pip install -r ../../requirements.txt kagglehub
python scripts/01_fetch_fitbit_data.py     # Kaggle download → data/raw/
python scripts/02_process_telemetry.py     # cleaning, cohorting → data/processed/
streamlit run scripts/03_dashboard.py      # dashboard @ localhost:8501
```

## Outputs

- `data/raw/` — 18 Kaggle CSVs (not committed; see `.gitignore`)
- `data/processed/user_events.csv` — tidy event stream (1,003 rows)
- `data/processed/user_cohorts.csv` — per-user cohort assignment + retention flags
- `data/processed/cohort_retention.csv` — cohort × day-offset retention matrix
