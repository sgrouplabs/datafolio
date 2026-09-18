# Auto Collision Risk Matrix — Texas Territory Engine

I built this dashboard on the [US Accidents (2016–2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) Kaggle dataset, filtered to Texas (582,837 accidents). I wanted to turn raw crash records into territory-level intelligence that supports auto liability and physical damage rate-making — the same kind of evidence a product analyst pulls together before proposing rate territories.

**[Live demo](https://sgrouplabs.github.io/datafolio/autorisk/)** — runs in the browser via stlite, no installation needed.

## Why P&C teams care

- **Territory risk identification:** the geospatial map exposes high-severity clusters, which is the raw material for defining rate territories instead of leaning only on county or ZIP boundaries.
- **Pricing bands:** the city ranking table ranks metros by volume × severity. I use it as a defensible starting point for tiering metro core vs exurban ring territories.
- **Environmental risk multipliers:** the weather module computes a *severity index* — each weather group's average severity divided by the clear-conditions baseline. That ratio has the same shape as a ratemaking factor: if fog crashes average 1.2× the severity of clear-weather crashes, that number anchors the multiplier discussion for an adverse-weather relativities table.
- **Frequency assumptions:** the day-of-week × hour heatmap quantifies rush-hour exposure bands (7–9 AM, 4–7 PM), which matter for commercial auto, telematics-based usage rating, and policyholder engagement campaigns.

## How I built it

| Script | Purpose |
|---|---|
| `scripts/01_kaggle_extract.py` | Downloads the dataset via `kagglehub` (653MB archive, 7.7M rows). Skips if already staged. |
| `scripts/02_process_risk_data.py` | Streams the 3GB+ CSV in 500k-row chunks so it never sits fully in memory, keeps TX rows, and engineers Hour / DayOfWeek / Month / Weather_Group features. |
| `scripts/03_dashboard.py` | Streamlit app: geospatial severity map, temporal heatmap, weather multiplier analysis, territory ranking. |

The full processed file is ~582k rows. For the browser demo I ship a 25,000-row random sample (`tx_collision_risk_sample.csv`, ~2.3MB) so the page loads fast; the local pipeline always uses the full dataset.

## Run it

```bash
pip install kagglehub pandas streamlit plotly
python scripts/01_kaggle_extract.py
python scripts/02_process_risk_data.py
streamlit run scripts/03_dashboard.py
```

`data/raw/` and the full processed CSV are gitignored; only code, this README, and the demo sample ship to the repo.

## Data dictionary (processed output)

| Field | Description |
|---|---|
| `Severity` | 1 (least traffic impact) – 4 (most severe) |
| `Start_Lat`, `Start_Lng` | Accident coordinates (map plotting) |
| `Hour`, `DayOfWeek`, `Month` | Temporal features extracted from `Start_Time` |
| `Weather_Group` | Categorical rollup: Clear, Cloudy, Rain, Storm, Fog, Snow, Other |
| `City` | TX city for territory aggregation |
| `Temperature(F)`, `Visibility(mi)`, `Humidity(%)`, `Wind_Speed(mph)` | Environmental covariates for deeper modeling |

## Honest limitations

- Severity in this dataset reflects *traffic impact duration*, not injury scales. I treat it as a claims-severity proxy, not a casualty measure.
- Reporting density skews toward metro areas with more data-capturing agencies, so "low risk" rural areas may partly be low visibility.
- The clear-weather baseline carries a composition effect: clear-weather accidents skew toward high-exposure urban highways, which flatters the non-clear multipliers a bit.
