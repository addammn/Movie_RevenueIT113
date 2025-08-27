import os, json
import streamlit as st
import pandas as pd
from joblib import load

st.set_page_config(page_title="Movie Revenue – Two Models", layout="centered")
st.title("🎬 Movie Revenue Predictor (Two Models)")

# Expect artifacts committed to the repo after training in Colab
ART_DIR = "artifacts"
modelA_path = os.path.join(ART_DIR, "model_A.joblib")
modelB_path = os.path.join(ART_DIR, "model_B.joblib")
schema_path = os.path.join(ART_DIR, "feature_columns.json")

missing = [p for p in [modelA_path, modelB_path, schema_path] if not os.path.exists(p)]
if missing:
    st.warning("Artifacts not found. Please run the Colab notebook and commit the 'artifacts/' folder.\n"
               f"Missing: {', '.join(missing)}")
    st.stop()

# Load models + schema
modelA = load(modelA_path)  # baseline
modelB = load(modelB_path)  # stronger
with open(schema_path) as f:
    feature_cols = json.load(f).get("columns", [])

st.subheader("1) Provide input features")
mode = st.radio("Input method:", ["Manual Entry", "Upload Single-Row CSV"], horizontal=True)

def manual_form(cols):
    vals = {}
    with st.form("manual_input"):
        for c in cols:
            # free-text inputs keep UI simple; pipeline handles types/one-hot
            vals[c] = st.text_input(c, "")
        go = st.form_submit_button("Predict with Both Models")
    return pd.DataFrame([vals], columns=cols) if go else None

row_df = None
if mode == "Manual Entry":
    row_df = manual_form(feature_cols)
else:
    up = st.file_uploader(
        "Upload a CSV with exactly ONE row including these columns:\n" + ", ".join(feature_cols),
        type=["csv"]
    )
    if up is not None:
        df = pd.read_csv(up)
        if len(df) == 1 and set(feature_cols) <= set(df.columns):
            row_df = df[feature_cols].copy()
        else:
            st.error("CSV must have exactly 1 row and include all required columns.")

if row_df is not None:
    st.subheader("2) Predictions (millions USD)")
    try:
        predA = float(modelA.predict(row_df)[0])
        predB = float(modelB.predict(row_df)[0])
        st.metric("Model A (Baseline: Linear Regression)", f"{predA:.2f}")
        st.metric("Model B (Stronger: Random Forest)", f"{predB:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
