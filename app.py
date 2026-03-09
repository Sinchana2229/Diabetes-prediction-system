# app.py (safe dynamic UI that matches artifact feature names)
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(layout="centered")
st.title("Diabetes Prediction App (dynamic feature UI

artifact = joblib.load("diabetes_model.joblib")
model = artifact['model']
imputer = artifact['imputer']
scaler = artifact['scaler']
feat_cols = artifact['feature_columns']
target_col = artifact.get('target_column', 'Outcome')

st.markdown(f"**Model type:** {type(model).__name__}  \n**Target column:** {target_col}")

# If the trained features are generic (V1..Vn), show them; if they are Pima names, show those.
median_vals = None
try:
    median_vals = pd.Series(imputer.statistics_, index=feat_cols)
except Exception:
    median_vals = pd.Series([0]*len(feat_cols), index=feat_cols)

st.subheader("Features (enter values)")
inputs = {}
for c in feat_cols:
    default = float(median_vals.get(c, 0.0)) if median_vals is not None else 0.0
    inputs[c] = st.number_input(label=c, value=default, format="%.3f")

if st.button("Predict"):
    # Build ordered DataFrame exactly matching training feature order/names
    user_df = pd.DataFrame([inputs])
    ordered = pd.DataFrame(columns=feat_cols)
    for col in feat_cols:
        ordered[col] = user_df[col] if col in user_df.columns else np.nan

    # replace zeros for known Pima columns only if they exist in training columns
    for p in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
        if p in ordered.columns:
            ordered[p] = ordered[p].replace(0, np.nan)

    # transform using saved preprocessors
    df_imp = pd.DataFrame(imputer.transform(ordered), columns=feat_cols)
    df_scaled = pd.DataFrame(scaler.transform(df_imp), columns=feat_cols)

    pred = model.predict(df_scaled)[0]
    proba = None
    try:
        proba = model.predict_proba(df_scaled)[0,1]
    except Exception:
        proba = None

    if proba is not None:
        st.write(f"**Prediction:** {int(pred)} — probability {proba:.3f}")
    else:
        st.write(f"**Prediction:** {int(pred)}"
