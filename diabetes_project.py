"""
diabetes_project.py

Training script for Diabetes prediction (Pima dataset).
This script defaults to using the CSV path you gave in your screenshot.
You may override the path using --data if needed.
"""

import argparse
import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix
)

# ---------- EDITABLE DEFAULT: use the exact path shown in your screenshot ----------
DEFAULT_LOCAL_DATA = DEFAULT_LOCAL_DATA = r"C:\Users\Sinchana - Personal\Desktop\diabetes_project\diabetes_project\diabetes.csv"


DEFAULT_OUTPUT = "diabetes_model.joblib"
RANDOM_STATE = 42

PIMA_ZERO_AS_NA = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']


def load_data(path: str) -> pd.DataFrame:
    """Load CSV from path with helpful error messages."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found at: {path}\n"
            "Please check the path and make sure the file exists. "
            "You can pass --data to specify a different path."
        )
    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    print("Dataset loaded. Shape:", df.shape)
    return df


def preprocess(df: pd.DataFrame):
    """Preprocess Pima dataset: replace zeros with NaN for certain columns, median impute, scale."""
    df = df.copy()
    expected_target = "Outcome"
    if expected_target not in df.columns:
        raise ValueError(f"Expected target column '{expected_target}' not found. Columns: {df.columns.tolist()}")

    # Replace 0 with NaN for known Pima columns where 0 is invalid
    for c in PIMA_ZERO_AS_NA:
        if c in df.columns:
            df[c] = df[c].replace(0, np.nan)

    X = df.drop(columns=[expected_target])
    y = df[expected_target].astype(int)

    # Impute with median
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    feature_names = list(X.columns)
    return X_scaled, y, imputer, scaler, feature_names


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = None
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        pass

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    print(f"\nModel: {type(model).__name__}")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    if roc is not None:
        print(f"ROC AUC: {roc:.4f}")
    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': roc}


def train_and_save(df: pd.DataFrame, save_path: str = DEFAULT_OUTPUT):
    X, y, imputer, scaler, feature_names = preprocess(df)

    # train/test split (stratify only if >1 class)
    strat = y if len(set(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=strat
    )

    # train logistic regression
    print("\nTraining LogisticRegression...")
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)

    # train random forest
    print("\nTraining RandomForestClassifier...")
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)

    # evaluate
    print("\nEvaluating LogisticRegression on test set:")
    lr_metrics = evaluate(lr, X_test, y_test)

    print("\nEvaluating RandomForestClassifier on test set:")
    rf_metrics = evaluate(rf, X_test, y_test)

    # choose best by F1
    best_model = rf if rf_metrics['f1'] >= lr_metrics['f1'] else lr
    print(f"\nBest model selected: {type(best_model).__name__} (F1 lr={lr_metrics['f1']:.4f}, rf={rf_metrics['f1']:.4f})")

    artifact = {
        "model": best_model,
        "imputer": imputer,
        "scaler": scaler,
        "feature_columns": feature_names,
        "target_column": "Outcome"
    }

    joblib.dump(artifact, save_path)
    print(f"\nSaved artifact to: {save_path}")

    return artifact


def example_predict(artifact: dict):
    feat_cols = artifact['feature_columns']
    imputer = artifact['imputer']
    scaler = artifact['scaler']
    model = artifact['model']

    # build a sample using medians
    med = pd.Series(imputer.statistics_, index=feat_cols)
    sample = {c: float(med.get(c, 0.0)) if not pd.isnull(med.get(c, np.nan)) else 0.0 for c in feat_cols}

    x = pd.DataFrame([sample], columns=feat_cols)
    x_imp = pd.DataFrame(imputer.transform(x), columns=feat_cols)
    x_scaled = pd.DataFrame(scaler.transform(x_imp), columns=feat_cols)

    pred = int(model.predict(x_scaled)[0])
    proba = None
    try:
        proba = float(model.predict_proba(x_scaled)[0, 1])
    except Exception:
        proba = None

    print("\nExample sample keys:", list(sample.keys())[:8], "...")
    print("Predicted label:", pred, " probability:", proba)


def main():
    parser = argparse.ArgumentParser(description="Train diabetes prediction model (Pima dataset).")
    parser.add_argument("--data", type=str, help="Path to local CSV file (optional). If not provided uses default path shown in screenshot.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Path to save artifact (.joblib)")
    args = parser.parse_args()

    data_path = args.data if args.data else DEFAULT_LOCAL_DATA_if_exists()

    try:
        df = load_data(data_path)
    except FileNotFoundError as e:
        print("ERROR:", e)
        print("\nTip: place the CSV at the path shown in your screenshot, or run with --data <your_csv_path>")
        sys.exit(1)
    except Exception as e:
        print("ERROR loading CSV:", e)
        sys.exit(1)

    # Print columns to make sure correct file loaded
    print("Dataset columns:", df.columns.tolist())

    # Detect that dataset has Pima columns
    expected = {'Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome'}
    if not expected.issubset(set(df.columns)):
        print("\nERROR: This CSV does not contain the expected Pima columns.")
        print("Expected:", expected)
        print("Found  :", set(df.columns))
        print("\nIf your file has different columns, either provide a Pima-format CSV or edit the script.")
        sys.exit(1)

    artifact = train_and_save(df, save_path=args.output)
    example_predict(artifact)


# Helper to pick default path (keeps constant in top for readability)
def DEFAULT_LOCAL_DATA_if_exists():
    # default path from your screenshot:
    p = r"C:\Users\Sinchana - Personal\Desktop\diabetes_project\diabetes_project\diabetes.csv"
    return p


if __name__ == "__main__":
    main()
