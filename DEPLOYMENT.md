# Streamlit Deployment Guide

## Local deployment

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

1. Push this project to GitHub.
2. Keep the trained model and scaler files in `artifacts/models/` if they are small enough for your repository/LFS setup:
   - `best_attention_lstm.keras`
   - `feature_scaler.joblib`
   - `preprocess_config.json`
3. Do **not** push the full 5.75 GB DAIC-WOZ dataset to GitHub.
4. On Streamlit Cloud, set the main file as:

```text
app.py
```

5. The app can perform inference only after trained artifacts are present.

## Important

The full DAIC-WOZ dataset is large and should usually be stored outside the public app repository. Train locally or in Colab/Kaggle, then deploy only the trained artifacts and app code.
