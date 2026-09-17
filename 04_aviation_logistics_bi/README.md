# 04 · Aviation Logistics BI — Star Schema ETL & Power BI Measures

I built a Python ETL pipeline that turns a flat Kaggle flight extract 
into a four-table star schema, then wrote the
DAX measures a Power BI executive dashboard needs.

## Why this matters

Flight delays are a logistics bottleneck problem.
Carriers, airports, and weather each contribute delay minutes, 
but you can only see which lever to pull if the delay data is
modeled by date, airline, and airport independently. A star schema is what
makes that slicing cheap in a BI engine — dimensions join to one fact table
on integer keys, and Power BI's VertiPaq store compresses it far better
than a wide flat file ever could.

## Pipeline

```
data/raw/flight_data_2024_sample.csv        (Kaggle extract, BTS format)
        │  scripts/01_kaggle_extract.py     kagglehub download
        ▼
data/processed/
├── Dim_Date.csv           366 rows   calendar: year, quarter, month, day names
├── Dim_Airline.csv         15 rows   unique carriers
├── Dim_Airport.csv        312 rows   conformed origin+destination airports
└── Fact_Flights.csv    10,000 rows   foreign keys + delay/air-time metrics
        │  scripts/02_build_star_schema.py
        ▼
dax_measures.txt                             copy-paste DAX for Power BI
```

## Data modeling decisions

**Conformed airport dimension.** I put origin and destination airports in
one `Dim_Airport` table instead of two separate dimensions. Both foreign
keys (`origin_key`, `dest_key`) point at it, and in Power BI the origin
relationship is active while the destination one is inactive, switchable
with `USERELATIONSHIP`. This is the standard role-playing-dimension
pattern, and it halves the dimension storage while keeping both directions
queryable.

**Delays on cancelled flights stay blank.** A cancelled flight has no arrival delay — that's
not missing data, it's a different fact. Imputing 0 would quietly flatter
the delay averages, and I refuse to let an imputation make a metric look
better than reality. Leaving NaN means DAX's `AVERAGE` and `SUM` ignore
them naturally, and my on-time rate measure excludes cancellations from its
denominator explicitly. 

**Surrogate keys.** I made `date_key` the YYYYMMDD integer (human-readable
and compresses well), and airline/airport keys are simple integers from 1.

**Carrier names.** The sample extract only carries the BTS carrier code, so
my `Dim_Airline` uses the code as its display name. A production build would
join the DOT's L_AIRLINE_ID lookup to get marketing names.

## DAX logic notes

The measures in `dax_measures.txt` cover what an executive dashboard needs:
volume (flights, miles, air time), delay averages and totals, the two rates
that get executives' attention, delay-cause attribution, and YoY
comparisons.

- **On-Time Performance Rate** counts a flight as on time only if both
  departure and arrival delay are ≤ 0 (early counts). Cancellations are
  excluded from the denominator, not counted as failures — I track them
  separately as Cancellation Rate, which matches how the DOT reports it.
  Mixing the two makes OTP look worse than it is and hides the actual
  cancellation problem. 
- **Flights Delayed 15+ Min** uses the DOT's 15-minute threshold rather than
  any delay at all. A 2-minute taxi delay is noise; a 15-minute one
  cascades into crew schedules and downstream legs.

## Business value

With these measures loaded, Power BI answers the questions ops teams
actually care about: which carriers run late and by how many minutes of
which cause, which airports are the bottleneck (origin vs destination
traffic, separable via the role-playing dimension), whether delay minutes
are trending up year over year, and how much of the delay bill is weather
(nobody's fault) versus late aircraft (the carrier's own network cascading).

That last split - Late-aircraft delay is internal and fixable; weather is not. 

## Local setup

```bash
git clone https://github.com/sgrouplabs/datafolio.git
cd datafolio/04_aviation_logistics_bi
python -m venv .venv && source .venv/bin/activate
pip install kagglehub pandas
python scripts/01_kaggle_extract.py
python scripts/02_build_star_schema.py
# then load data/processed/*.csv into Power BI and paste dax_measures.txt
```

## Repo layout

```
04_aviation_logistics_bi/
├── scripts/
│   ├── 01_kaggle_extract.py       # kagglehub download → data/raw/
│   └── 02_build_star_schema.py    # clean + star schema → data/processed/
├── data/
│   ├── raw/                       # Kaggle extract (not committed)
│   └── processed/                 # Dim_* and Fact_Flights CSVs
└── dax_measures.txt               # Power BI measure library
```
