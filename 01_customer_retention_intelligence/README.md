# 01 · Customer Retention & Churn Risk Intelligence

I built this dashboard because churn is the problem I understand best from
my accounting days. Every customer who leaves takes their monthly payment
with them for good, and winning a replacement costs several times what
keeping the original would have — that's revenue that never hits next
quarter's books. So I wanted to build something that makes attrition
actionable: show where the money is leaking by segment, and flag individual
accounts while there's still time to save them.

The result is a Streamlit + scikit-learn app that scores the IBM Telco
cohort, shows where revenue is leaking, and predicts an individual
customer's churn probability in real time.

## Why this matters

When I worked in accounting, I saw plenty of reports that described losses
after the fact. What I never saw was a tool that pointed at *who* to call
*before* they left. That's the gap I'm trying to close here: quantify MRR at
risk by segment first, then rank individual accounts while intervention is
still cheap.

## Data

I used the [IBM Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
dataset — 7,043 subscribers and 21 attributes. My training script downloads
it automatically, so you don't need to fetch anything by hand.

The target is `Churn` (Yes/No → 1/0), and only about 26.5% of customers
churn, so the classes are imbalanced. I handled that with
`class_weight='balanced'` in both models rather than resampling. I tried to
keep the data intact — old habit from auditing, where every transformed row
is a place a mistake can hide — and at this ratio it worked fine anyway.

One wrinkle I ran into: `TotalCharges` comes in as an object column because
blank strings show up for customers with zero tenure. I coerced it to
numeric and filled the blanks with 0.0. Those are brand-new customers who
genuinely haven't been billed yet, not missing data, so dropping them would
have thrown away exactly the population I care most about.

### Feature dictionary (key fields)

| Field | Type | Description |
|-------|------|-------------|
| `tenure` | numeric | Months with the company |
| `MonthlyCharges` / `TotalCharges` | numeric | Recurring / cumulative billing |
| `Contract` | categorical | Month-to-month, One-year, Two-year |
| `InternetService` | categorical | DSL, Fiber optic, None |
| `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport` | categorical | Service add-ons |
| `PaymentMethod`, `PaperlessBilling` | categorical | Billing profile |

## Methodology and results

Preprocessing is a `ColumnTransformer`: `StandardScaler` on the numerics,
`OneHotEncoder(drop='first', handle_unknown='ignore')` on the categoricals.
I compared models with stratified 5-fold cross-validation on precision,
recall, F1, and ROC-AUC.

| Model | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|--------|----|---------|
| Logistic Regression (balanced) | 0.519 | **0.802** | **0.630** | **0.766** |
| Random Forest (balanced, 300 trees) | **0.577** | 0.648 | 0.610 | 0.738 |

Logistic Regression won on CV ROC-AUC and scored **0.842** on my held-out
test set.

I want to explain why I chose recall over precision, because it's a judgment
call and I think it's the most defensible part of this project. A missed
churner costs the full lifetime value of the account. A false positive
costs one outreach call, maybe a discount I didn't need to give. From my
accounting brain: one is a real write-off, the other is a rounding error. So
I'll take LR's 0.80 recall over RF's 0.65 every time — yes, I'll annoy some
customers who were never going to leave, but I catch roughly 15 points more
of the ones who were. That trade is worth it.

## Local setup

```bash
git clone https://github.com/sgrouplabs/datafolio.git
cd datafolio/01_customer_retention_intelligence
python -m venv .venv && source .venv/bin/activate
pip install -r ../requirements.txt
python train_model.py        # downloads data, trains, saves models/
streamlit run app.py         # opens dashboard on http://localhost:8501
```

## Repo layout

```
01_customer_retention_intelligence/
├── app.py                       # 3-tab Streamlit dashboard
├── train_model.py               # data + pipeline + training + evaluation
├── data/telco_churn.csv         # dataset (auto-downloaded)
└── models/
    ├── churn_classifier.joblib  # best classifier
    ├── preprocessor.joblib      # ColumnTransformer
    └── cv_results.joblib        # CV metrics
```
