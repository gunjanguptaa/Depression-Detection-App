import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="Depression Detection – Classical ML",
    layout="centered"
)

st.title("🧠 Depression Detection using Classical ML Models")
st.markdown(
    """
This module compares **five classical machine learning models**
trained on **DAIC-WOZ audio features**.

**Models included**
- SVM (RBF)
- Random Forest
- XGBoost
- Logistic Regression
- MLP
"""
)

# ============================================================
# PATHS (RELATIVE – STREAMLIT SAFE)
# ============================================================
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

MODEL_PATHS = {
    "SVM (RBF)": os.path.join(MODEL_DIR, "svm.pkl"),
    "Random Forest": os.path.join(MODEL_DIR, "random_forest.pkl"),
    "XGBoost": os.path.join(MODEL_DIR, "xgboost.pkl"),
    "Logistic Regression": os.path.join(MODEL_DIR, "logistic_regression.pkl"),
    "MLP": os.path.join(MODEL_DIR, "mlp.pkl"),
}

# ============================================================
# LOAD MODELS (CACHED)
# ============================================================
@st.cache_resource
def load_models():
    models = {}
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            st.error(f"❌ Model file missing: {path}")
            continue
        models[name] = joblib.load(path)
    return models

@st.cache_resource
def load_scaler():
    if not os.path.exists(SCALER_PATH):
        st.error("❌ scaler.pkl not found")
        return None
    return joblib.load(SCALER_PATH)

models = load_models()
scaler = load_scaler()

if scaler is None or len(models) == 0:
    st.stop()

# ============================================================
# FEATURE INPUT UI
# ============================================================
st.subheader("🎧 Input Audio Features")

st.markdown(
    """
Enter **aggregated audio features** (same order used during training).

Example:  
`MFCC mean, MFCC std, spectral centroid, bandwidth, rolloff, ZCR`
"""
)

feature_input = st.text_area(
    "Feature Vector (comma separated)",
    placeholder="e.g. 0.23, -1.12, 345.6, 1200.4, 0.034, ..."
)

# ============================================================
# PREDICTION
# ============================================================
if st.button("🔍 Predict"):
    try:
        features = np.array(
            [float(x.strip()) for x in feature_input.split(",")],
            dtype=np.float32
        ).reshape(1, -1)

        features_scaled = scaler.transform(features)

        st.subheader("📊 Model Predictions")

        results = []

        for model_name, model in models.items():
            pred = model.predict(features_scaled)[0]

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(features_scaled)[0][1]
            else:
                prob = None

            label = "Depressed" if pred == 1 else "Not Depressed"

            results.append({
                "Model": model_name,
                "Prediction": label,
                "Depression Probability": round(prob, 3) if prob is not None else "N/A"
            })

        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "Models retrained with NumPy 1.26.4 • Pickle protocol 4 • DAIC-WOZ Dataset"
)
