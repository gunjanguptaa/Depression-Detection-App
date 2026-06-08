"""
verify_models.py
─────────────────
Run after training to confirm all .pkl files load correctly.
This simulates exactly what Streamlit Cloud does at startup.

Usage:
    python verify_models.py
"""

import sys
import numpy as np
import joblib
import json
from pathlib import Path

MODELS_DIR = Path("comparison/models")
RESULTS_DIR = Path("comparison/results")

MODEL_NAMES = {
    "svm": "SVM",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "logistic_regression": "Logistic Regression",
    "mlp": "MLP",
}


def check():
    print("="*50)
    print("  Model Compatibility Checker")
    print("="*50)
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  NumPy   : {np.__version__}")
    print()

    all_ok = True

    # Check scaler
    scaler_path = MODELS_DIR / "scaler.pkl"
    try:
        scaler = joblib.load(scaler_path)
        print(f"  ✓ scaler.pkl loaded  ({type(scaler).__name__})")
    except Exception as e:
        print(f"  ✗ scaler.pkl FAILED: {e}")
        all_ok = False

    # Check each model
    for fname, display_name in MODEL_NAMES.items():
        path = MODELS_DIR / f"{fname}.pkl"
        if not path.exists():
            print(f"  ✗ {fname}.pkl  NOT FOUND")
            all_ok = False
            continue
        try:
            model = joblib.load(path)
            # Quick smoke test: predict on dummy data
            dummy = np.random.randn(1, scaler.n_features_in_).astype(np.float32)
            dummy_scaled = scaler.transform(dummy)
            pred = model.predict(dummy_scaled)
            prob = model.predict_proba(dummy_scaled)[0]
            print(f"  ✓ {fname:<25} pred={pred[0]}  P(depressed)={prob[1]:.4f}")
        except Exception as e:
            print(f"  ✗ {fname:<25} FAILED: {e}")
            all_ok = False

    # Check results JSON
    results_path = RESULTS_DIR / "classical_ml_results.json"
    if results_path.exists():
        try:
            with open(results_path) as f:
                results = json.load(f)
            print(f"\n  ✓ classical_ml_results.json  ({len(results)} models)")
            for name, m in results.items():
                print(f"    {name:<22}  Acc={m['accuracy']:.3f}  F1={m['f1_score']:.3f}")
        except Exception as e:
            print(f"\n  ✗ classical_ml_results.json FAILED: {e}")
            all_ok = False
    else:
        print(f"\n  ✗ classical_ml_results.json NOT FOUND at {results_path}")
        all_ok = False

    print()
    if all_ok:
        print("  ✅  All checks passed — safe to push to GitHub!")
    else:
        print("  ❌  Some checks failed — do NOT push yet, fix errors first.")
    return all_ok


if __name__ == "__main__":
    ok = check()
    sys.exit(0 if ok else 1)
