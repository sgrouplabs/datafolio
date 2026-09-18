# Auto Collision Risk Matrix — Texas Territory Engine

A P&C actuarial dashboard built on the [US Accidents (2016–2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) Kaggle dataset, filtered to Texas (582,837 accidents). It turns raw crash records into territory-level intelligence that supports auto liability and physical damage rate-making.

## Why P&C teams care

- **Territory risk identification:** the geospatial map exposes high-severity clusters, the raw material for defining rate territories instead of relying only on USPS/county boundaries.
- **Pricing bands:** the city ranking table ranks metros by volume × severity, a defensible starting point for tiering metro core vs exurban ring territories.
- **Environmental risk multipliers:** the weather module computes a *severity index* (group average severity ÷ Clear-conditions baseline) per weather class — the same shape as a ratemaking factor. If fog crashes average 1.2× the severity of clear-weather crashes, that ratio anchors the multiplier discussion for an adverse-weather relativities table.
- **Frequency assumptions:** the day-of-week × hour heatmap quantifies rush-hour exposure bands (7–9 AM, 4–7 PM), useful for commercial auto, telematics-based usage rating, and policyholder engagement campaigns.

## Pipeline

| Script | Purpose |
|---|---|
| `scripts/01_kaggle_extract.py` | Downloads the dataset via `kagglehub` (653MB archive, 7.7M rows). Skips if already staged. |
| `scripts/02_process_risk_data.py` | Streams the 3GB+ CSV in 500k-row chunks (never fully in memory), keeps TX rows, engineers Hour / DayOfWeek / Month / Weather_Group features. |
| `scripts/03_dashboard.py` | Streamlit app: geospatial severity map, temporal heatmap, weather multiplier analysis, territory ranking. |


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

- Severity reflects *traffic impact duration*, not injury/bodily-harm scales — treat it as a claims-severity proxy, not a casualty measure.
- Reporting density skews toward metro areas with more data-capturing agencies; rural "low risk" may partly be low visibility.
- Clear-weather baseline carries a composition effect: clear-weather accidents skew toward high-exposure urban highways.
