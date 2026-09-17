"""Customer Retention & Churn Risk Intelligence Dashboard.

Interactive multi-tab Streamlit application:
  1. Executive Overview & Cohort Analysis
  2. Individual Risk Simulator
  3. Model Performance & Diagnostics
"""

import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "telco_churn.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

st.set_page_config(page_title="Churn Risk Intelligence", layout="wide",
                   page_icon="📉")

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH if os.path.exists(DATA_PATH) else DATA_URL)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    df["ChurnFlag"] = (df["Churn"].str.strip().str.lower() == "yes").astype(int)
    # The preprocessor was fitted with every categorical column cast to str
    # (train_model.load_data). Feeding raw dtypes (SeniorCitizen as int64)
    # makes OneHotEncoder.transform compare int columns against string
    # categories, which crashes with "ufunc 'isnan' not supported" under
    # stlite/Pyodide. Cast here so app and training always agree.
    for col in ["gender", "SeniorCitizen", "Partner", "Dependents",
                "PhoneService", "MultipleLines", "InternetService",
                "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
                "PaperlessBilling", "PaymentMethod"]:
        df[col] = df[col].astype(str).str.strip()
    return df


@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(MODELS_DIR, "churn_classifier.joblib"))
    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
    # Version-proofing: the pickle was saved under sklearn >=1.8 (which dropped
    # the multi_class attribute), but the stlite/micropip runtime may install
    # an older sklearn (1.6.x) whose LogisticRegression.predict_proba still
    # reads self.multi_class. Restore the attribute so predict works on both.
    if hasattr(model, "coef_") and not hasattr(model, "multi_class"):
        model.multi_class = "auto"
    return model, preprocessor


df = load_data()
model, preprocessor = load_model()

st.title("📉 Customer Retention & Churn Risk Intelligence")
st.caption("IBM Telco Cohort · Graduate Portfolio in Data Science & ML")

# ---------------------------------------------------------------------------
# Sidebar filters (shared by overview tab; simulator has its own inputs)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Cohort Filters")
    contracts = st.multiselect("Contract Type",
                               sorted(df["Contract"].unique()),
                               default=sorted(df["Contract"].unique()))
    payments = st.multiselect("Payment Method",
                              sorted(df["PaymentMethod"].unique()),
                              default=sorted(df["PaymentMethod"].unique()))
    tech_support = st.multiselect("Tech Support",
                                  sorted(df["TechSupport"].unique()),
                                  default=sorted(df["TechSupport"].unique()))
    internet = st.multiselect("Internet Service",
                              sorted(df["InternetService"].unique()),
                              default=sorted(df["InternetService"].unique()))
    tenure_range = st.slider("Tenure Range (months)", 0, 72, (0, 72))

filtered = df[
    df["Contract"].isin(contracts)
    & df["PaymentMethod"].isin(payments)
    & df["TechSupport"].isin(tech_support)
    & df["InternetService"].isin(internet)
    & df["tenure"].between(*tenure_range)
]

tab1, tab2, tab3 = st.tabs(
    ["📊 Executive Overview", "🎯 Risk Simulator", "🧪 Model Diagnostics"])

# ---------------------------------------------------------------------------
# Tab 1 — Executive Overview & Cohort Analysis
# ---------------------------------------------------------------------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    cohort_size = len(filtered)
    churn_rate = filtered["ChurnFlag"].mean() * 100 if cohort_size else 0
    mrr_at_risk = filtered.loc[filtered["ChurnFlag"] == 1,
                               "MonthlyCharges"].sum()
    high_risk = int((filtered["ChurnFlag"] == 1).sum())

    c1.metric("Total Cohort Size", f"{cohort_size:,}")
    c2.metric("Observed Churn Rate", f"{churn_rate:.1f}%")
    c3.metric("MRR at Risk", f"${mrr_at_risk:,.0f}")
    c4.metric("High-Risk Customers", f"{high_risk:,}")

    left, right = st.columns(2)
    with left:
        by_contract = (filtered.groupby("Contract")["ChurnFlag"]
                       .mean().mul(100).reset_index(name="Churn Rate (%)"))
        fig = px.bar(by_contract, x="Contract", y="Churn Rate (%)",
                     color="Contract", title="Churn Rate by Contract Type")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.histogram(filtered, x="MonthlyCharges", color="Churn",
                           nbins=40, barmode="overlay", opacity=0.6,
                           title="Monthly Charges: Retained vs Churned")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 2 — Individual Risk Simulator
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Individual Churn Risk Simulator")
    s1, s2, s3 = st.columns(3)
    tenure = s1.slider("Tenure (months)", 0, 72, 12)
    monthly = s2.number_input("Monthly Charges ($)", 10.0, 200.0, 70.0, 5.0)
    contract = s3.selectbox("Contract Type", sorted(df["Contract"].unique()))
    addons = st.multiselect(
        "Service Add-ons",
        ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
         "StreamingTV", "StreamingMovies"],
        default=["OnlineSecurity"])

    if st.button("Score Customer", type="primary"):
        row = {f: "No" for f in ["OnlineSecurity", "OnlineBackup",
                                 "DeviceProtection", "TechSupport",
                                 "StreamingTV", "StreamingMovies"]}
        for a in addons:
            row[a] = "Yes"
        sample = pd.DataFrame([{
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": float(monthly * tenure),
            "gender": "Female", "SeniorCitizen": "0", "Partner": "No",
            "Dependents": "No", "PhoneService": "Yes",
            "MultipleLines": "No", "InternetService": "Fiber optic",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "Contract": contract, **row,
        }])
        Xs = preprocessor.transform(sample)
        prob = float(model.predict_proba(Xs)[0, 1])
        tier = "High" if prob >= 0.6 else "Medium" if prob >= 0.3 else "Low"

        st.metric("Predicted Churn Probability", f"{prob * 100:.1f}%")
        color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[tier]
        st.markdown(f"### Risk Tier: {color} **{tier}**")

        recs = {
            "High": ["Offer month-to-month → annual contract conversion "
                     "incentive", "Bundle Tech Support / Online Security",
                     "Assign a retention specialist within 48 hours"],
            "Medium": ["Proactive check-in and satisfaction survey",
                       "Promote paperless billing / autopay discount"],
            "Low": ["Maintain engagement; upsell premium add-ons"],
        }[tier]
        st.markdown("**Recommended Retention Actions**")
        for r in recs:
            st.markdown(f"- {r}")

# ---------------------------------------------------------------------------
# Tab 3 — Model Performance & Diagnostics
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Model Performance & Diagnostics")

    # Rebuild a test split consistent with training.
    from sklearn.model_selection import train_test_split

    NUM = ["tenure", "MonthlyCharges", "TotalCharges"]
    CAT = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
           "MultipleLines", "InternetService", "OnlineSecurity",
           "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
           "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
    X = df[NUM + CAT]
    y = df["ChurnFlag"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2,
                                            stratify=y, random_state=42)
    X_test_t = preprocessor.transform(X_test)
    proba = model.predict_proba(X_test_t)[:, 1]
    preds = (proba >= 0.5).astype(int)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", f"{precision_score(y_test, preds):.3f}")
    m2.metric("Recall", f"{recall_score(y_test, preds):.3f}")
    m3.metric("F1-Score", f"{f1_score(y_test, preds):.3f}")
    m4.metric("ROC-AUC", f"{roc_auc_score(y_test, proba):.3f}")

    left, right = st.columns(2)
    with left:
        cm = confusion_matrix(y_test, preds)
        fig = go.Figure(go.Heatmap(z=cm, x=["Pred: Stay", "Pred: Churn"],
                                   y=["Actual: Stay", "Actual: Churn"],
                                   text=cm, texttemplate="%{text}",
                                   colorscale="Blues"))
        fig.update_layout(title="Confusion Matrix (held-out test set)")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fpr, tpr, _ = roc_curve(y_test, proba)
        fig = go.Figure(go.Scatter(x=fpr, y=tpr, mode="lines",
                                   name=f"ROC (AUC={roc_auc_score(y_test, proba):.3f})"))
        fig.add_hline(y=0, line_dash="dot")
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line=dict(dash="dot"),
                                 name="Chance"))
        fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Feature Coefficients / Importances")
    feature_names = preprocessor.get_feature_names_out()
    if hasattr(model, "coef_"):
        vals = model.coef_[0]
    else:
        vals = model.feature_importances_
    imp = (pd.DataFrame({"feature": feature_names, "value": vals})
           .reindex(pd.Series(vals).abs().sort_values(ascending=False).index))
    fig = px.bar(imp.head(15), x="value", y="feature", orientation="h",
                 title="Top 15 Drivers", labels={"value": "Coefficient",
                                                 "feature": ""})
    st.plotly_chart(fig, use_container_width=True)
