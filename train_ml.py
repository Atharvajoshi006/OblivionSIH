"""Model Training Script for Production Shortfall Intelligence.
Generates the calibrated operational simulation dataset and trains the
XGBoost regressor with SHAP tree explainer.
"""

import os
import sys
import shutil

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services.production_service import train_production_model, MODEL_PKL

if __name__ == "__main__":
    print("[GeoOre-AI] Training production shortfall forecasting model...")
    meta = train_production_model(force_retrain=True)
    print(f"[GeoOre-AI] Model training complete: {meta['model_type']}")
    print(f"[GeoOre-AI] Validation MAE: {meta['metrics']['mae_tons']} Tons, R2: {meta['metrics']['r2_score']}")
    print(f"[GeoOre-AI] Features: {meta['features']}")

    # Copy to legacy model path if exists
    legacy_path = os.path.join(PROJECT_ROOT, "models", "shortfall_xgb.pkl")
    if os.path.exists(MODEL_PKL):
        shutil.copyfile(MODEL_PKL, legacy_path)