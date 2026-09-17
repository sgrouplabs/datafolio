"""Train and evaluate churn models on the IBM Telco dataset.

Compares Logistic Regression and Random Forest with stratified 5-fold CV and
saves whichever wins on ROC-AUC, plus the fitted preprocessor.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (make_scorer, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "telco_churn.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.read_csv(DATA_URL)
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)

    # TotalCharges arrives as object with blanks for tenure-0 customers;
    # those really are zero-billed, so fillna with 0.0 rather than dropping.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).str.strip()

    df["Churn"] = (df["Churn"].str.strip().str.lower() == "yes").astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    # drop='first' avoids the dummy trap; unknown categories at inference
    # time map to all-zeros instead of erroring.
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"),
             CATEGORICAL_FEATURES),
        ]
    )


def get_models() -> dict:
    # class_weight='balanced' instead of resampling: churn is ~26.5% positive
    # and reweighting keeps the data intact at this ratio.
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


SCORING = {
    "precision": make_scorer(precision_score),
    "recall": make_scorer(recall_score),
    "f1": make_scorer(f1_score),
    "roc_auc": make_scorer(roc_auc_score),
}


def main() -> None:
    df = load_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Select on CV ROC-AUC. Recall ends up mattering more in practice (see
    # README), but AUC is threshold-free and the ranking agrees either way.
    results = {}
    best_name, best_auc = None, -np.inf
    for name, clf in get_models().items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", clf)])
        cv_res = cross_validate(pipe, X_train, y_train, cv=cv,
                                scoring=SCORING, n_jobs=-1)
        results[name] = {m: cv_res[f"test_{m}"].mean() for m in SCORING}
        print(f"\n{name} — Stratified 5-Fold CV:")
        for m, v in results[name].items():
            print(f"  {m:>10}: {v:.4f}")
        if results[name]["roc_auc"] > best_auc:
            best_name, best_auc = name, results[name]["roc_auc"]
            best_pipe = pipe

    print(f"\nBest model by CV ROC-AUC: {best_name} ({best_auc:.4f})")
    best_pipe.fit(X_train, y_train)
    test_auc = roc_auc_score(y_test, best_pipe.predict_proba(X_test)[:, 1])
    print(f"Held-out test ROC-AUC: {test_auc:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_pipe.named_steps["model"],
                os.path.join(MODELS_DIR, "churn_classifier.joblib"))
    joblib.dump(best_pipe.named_steps["preprocess"],
                os.path.join(MODELS_DIR, "preprocessor.joblib"))
    print("Saved models/churn_classifier.joblib and models/preprocessor.joblib")

    joblib.dump({"cv_results": results, "best_model": best_name,
                 "test_roc_auc": test_auc},
                os.path.join(MODELS_DIR, "cv_results.joblib"))


if __name__ == "__main__":
    main()
