"""Production Intelligence & Explainability Service for GeoOre-AI.
Implements realistic open-cast mine operational data synthesis,
gradient-boosted tree production shortfall forecasting, and SHAP feature attribution.

TRANSPARENCY NOTICE:
In the absence of proprietary MOIL SCADA/dispatch telematics, the training dataset
is a calibrated operational engineering simulation explicitly labelled as
'mine_operations_simulated.csv'. It incorporates empirical mining physics:
shovel loading rates, haul cycle times, road rolling resistance, and blasting delays.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import shap

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SIMULATED_DATASET_CSV = os.path.join(PROCESSED_DIR, "mine_operations_simulated.csv")
MODEL_PKL = os.path.join(MODELS_DIR, "shortfall_predictor.pkl")
EXPLAINER_PKL = os.path.join(MODELS_DIR, "shap_explainer.pkl")
METADATA_JSON = os.path.join(MODELS_DIR, "model_metadata.json")

FEATURE_NAMES = [
    "planned_production_tons",
    "active_shovels",
    "shovel_availability_pct",
    "haul_trucks",
    "rainfall_mm",
    "soil_moisture_pct",
    "temperature_c",
    "blasting_delayed"
]

# Global cache for loaded model and explainer
_MODEL_INSTANCE: Optional[XGBRegressor] = None
_EXPLAINER_INSTANCE: Optional[shap.TreeExplainer] = None


def generate_calibrated_operational_data(n_records: int = 1200, random_seed: int = 42) -> pd.DataFrame:
    """Generates an operational engineering dataset simulating open-cast manganese mining.
    
    Models realistic bottlenecks:
    - Shovel fleet hourly loading capacity (nominal 180-220 tons/hr per shovel over 8-hr shift)
    - Truck cycle times degraded by haul-road slush and grade steepness
    - Weather-induced pit waterlogging and reduced tire traction
    - Blasting clearing delays
    """
    np.random.seed(random_seed)
    
    # 1. Base mine plan
    planned_production = np.random.choice([1000.0, 1200.0, 1400.0, 1600.0], size=n_records, p=[0.2, 0.4, 0.3, 0.1])
    
    # 2. Equipment Fleet
    active_shovels = np.random.randint(3, 8, size=n_records)
    shovel_availability = np.random.uniform(65.0, 100.0, size=n_records)  # mechanical health %
    haul_trucks = active_shovels * 2 + np.random.randint(-1, 5, size=n_records)
    haul_trucks = np.clip(haul_trucks, 6, 22)
    
    # 3. Environmental conditions (Calibrated to Balaghat seasonal monsoon & dry climate)
    is_monsoon = np.random.choice([0, 1], size=n_records, p=[0.70, 0.30])
    rainfall = np.where(
        is_monsoon == 1,
        np.random.exponential(scale=28.0, size=n_records),
        np.random.exponential(scale=2.5, size=n_records)
    )
    rainfall = np.clip(rainfall, 0.0, 110.0)
    
    soil_moisture = np.clip(22.0 + rainfall * 1.5 + np.random.normal(0, 5, n_records), 18.0, 95.0)
    temperature = np.random.uniform(20.0, 44.0, size=n_records)

    # 4. Blasting delays (heavy rain or lightning forces delay)
    blasting_delayed = np.where(
        (rainfall > 25.0) | (np.random.rand(n_records) < 0.12),
        1,
        0
    )

    # 5. Production physics calculations
    # Theoretical shovel capacity (tons per 8h shift)
    effective_shovel_cap = active_shovels * (shovel_availability / 100.0) * 210.0
    
    # Haulage capacity: standard 10 trips per truck per shift, 30t payload = 300t/truck
    # Wet/slushy roads increase rolling resistance and slow haul speed by up to 50%
    road_speed_factor = np.clip(1.0 - (soil_moisture - 35.0) / 100.0, 0.50, 1.0)
    effective_haul_cap = haul_trucks * 280.0 * road_speed_factor

    # Blasting delay penalty (lack of blasted muckpile starves shovels)
    blast_loss = np.where(blasting_delayed == 1, 260.0 + np.random.normal(0, 30, n_records), 0.0)

    # Extreme weather direct pit stoppage (rainfall > 50mm causes safety suspension)
    rain_loss = np.where(rainfall > 40.0, (rainfall - 40.0) * 12.0, 0.0)
    
    # Actual production is bounded by shovel and haul capacity minus operational disruptions
    attainable_capacity = np.minimum(effective_shovel_cap, effective_haul_cap)
    actual_production = attainable_capacity - blast_loss - rain_loss + np.random.normal(0, 35, n_records)
    actual_production = np.clip(actual_production, 150.0, planned_production * 1.05)
    
    shortfall = np.maximum(0.0, planned_production - actual_production)

    df = pd.DataFrame({
        "planned_production_tons": np.round(planned_production, 1),
        "active_shovels": active_shovels,
        "shovel_availability_pct": np.round(shovel_availability, 1),
        "haul_trucks": haul_trucks,
        "rainfall_mm": np.round(rainfall, 1),
        "soil_moisture_pct": np.round(soil_moisture, 1),
        "temperature_c": np.round(temperature, 1),
        "blasting_delayed": blasting_delayed,
        "actual_production_tons": np.round(actual_production, 1),
        "shortfall_tons": np.round(shortfall, 1)
    })

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(SIMULATED_DATASET_CSV, index=False)
    logger.info(f"Generated {n_records} simulated operational records at {SIMULATED_DATASET_CSV}")
    return df


def train_production_model(force_retrain: bool = False) -> Dict[str, Any]:
    """Trains an XGBoost Regressor on the calibrated operational dataset and computes SHAP explainer."""
    global _MODEL_INSTANCE, _EXPLAINER_INSTANCE
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    if not force_retrain and os.path.exists(MODEL_PKL) and os.path.exists(EXPLAINER_PKL):
        _MODEL_INSTANCE = joblib.load(MODEL_PKL)
        _EXPLAINER_INSTANCE = joblib.load(EXPLAINER_PKL)
        if os.path.exists(METADATA_JSON):
            with open(METADATA_JSON, "r") as f:
                return json.load(f)
        return {"status": "loaded_from_disk"}

    # Generate or load dataset
    if not os.path.exists(SIMULATED_DATASET_CSV):
        df = generate_calibrated_operational_data()
    else:
        df = pd.read_csv(SIMULATED_DATASET_CSV)

    X = df[FEATURE_NAMES]
    y = df["shortfall_tons"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    model = XGBRegressor(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    # Fit SHAP TreeExplainer on background sample
    explainer = shap.TreeExplainer(model, X_train.sample(min(150, len(X_train)), random_state=42))

    # Save artifacts
    joblib.dump(model, MODEL_PKL)
    joblib.dump(explainer, EXPLAINER_PKL)

    _MODEL_INSTANCE = model
    _EXPLAINER_INSTANCE = explainer

    metadata = {
        "status": "trained_success",
        "model_type": "XGBRegressor",
        "feature_count": len(FEATURE_NAMES),
        "features": FEATURE_NAMES,
        "metrics": {
            "mae_tons": round(mae, 2),
            "r2_score": round(r2, 4)
        },
        "evaluation_context": "Calibrated Operational Engineering Physics Simulation",
        "transparency_notice": "Trained on simulated open-cast mine physics (mine_operations_simulated.csv). Real-world dispatch accuracy requires telemetry logs from SCADA/fleet management systems."
    }
    with open(METADATA_JSON, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def _get_model_and_explainer() -> Tuple[XGBRegressor, shap.TreeExplainer]:
    global _MODEL_INSTANCE, _EXPLAINER_INSTANCE
    if _MODEL_INSTANCE is None or _EXPLAINER_INSTANCE is None:
        if not os.path.exists(MODEL_PKL) or not os.path.exists(EXPLAINER_PKL):
            train_production_model()
        else:
            _MODEL_INSTANCE = joblib.load(MODEL_PKL)
            _EXPLAINER_INSTANCE = joblib.load(EXPLAINER_PKL)
    return _MODEL_INSTANCE, _EXPLAINER_INSTANCE


def predict_production(operational_input: Dict[str, Any], include_trajectory: bool = True) -> Dict[str, Any]:
    """Dynamically predicts production, shortfall tons, risk rating, and key SHAP factors.
    
    Accepts:
        planned_production_tons: float (default 1200.0)
        active_shovels: int (default 5)
        shovel_availability_pct: float (default 85.0)
        haul_trucks: int (default 10)
        rainfall_mm: float (default 0.0)
        soil_moisture_pct: float (default 35.0)
        temperature_c: float (default 28.0)
        blasting_delayed: int (0 or 1, default 0)
    """
    model, explainer = _get_model_and_explainer()

    planned = float(operational_input.get("planned_production_tons") or 1200.0)
    shovels = int(operational_input.get("active_shovels") or 5)
    avail = float(operational_input.get("shovel_availability_pct") or 85.0)
    trucks = int(operational_input.get("haul_trucks") or (shovels * 2))
    rain = float(operational_input.get("rainfall_mm") if operational_input.get("rainfall_mm") is not None else 0.0)
    moisture = float(operational_input.get("soil_moisture_pct") if operational_input.get("soil_moisture_pct") is not None else 35.0)
    temp = float(operational_input.get("temperature_c") if operational_input.get("temperature_c") is not None else 28.0)
    blast = int(operational_input.get("blasting_delayed") or 0)

    feature_values = [planned, shovels, avail, trucks, rain, moisture, temp, blast]
    input_df = pd.DataFrame([feature_values], columns=FEATURE_NAMES)

    # Predict shortfall tons
    predicted_shortfall = float(model.predict(input_df)[0])
    predicted_shortfall = max(0.0, round(predicted_shortfall, 1))

    predicted_production = max(0.0, round(planned - predicted_shortfall, 1))

    # Calculate operational risk classification
    shortfall_ratio = predicted_shortfall / max(1.0, planned)
    if shortfall_ratio >= 0.30 or predicted_shortfall >= 380.0:
        risk_level = "HIGH"
    elif shortfall_ratio >= 0.12 or predicted_shortfall >= 140.0:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    # Compute SHAP feature contributions
    shap_values = explainer(input_df)
    values = shap_values.values[0]

    factor_descriptions = {
        "planned_production_tons": "Target production volume commitment",
        "active_shovels": "Excavator loading capacity at ore face",
        "shovel_availability_pct": "Mechanical uptime of loading fleet",
        "haul_trucks": "Haulage dumper fleet capacity",
        "rainfall_mm": "Direct rainfall pit flooding and haul ramp slowing",
        "soil_moisture_pct": "Haul road slush and rolling resistance",
        "temperature_c": "Ambient temperature heat stress",
        "blasting_delayed": "Muckpile fragmentation & blasting delay"
    }

    contributions = []
    for col, val, shap_val in zip(FEATURE_NAMES, feature_values, values):
        contributions.append({
            "factor": col,
            "input_value": val,
            "impact_tons": round(float(shap_val), 1),
            "description": factor_descriptions.get(col, col),
            "direction": "INCREASES_SHORTFALL" if shap_val > 0 else "REDUCES_SHORTFALL"
        })

    # Separate risk drivers (positive SHAP on shortfall) from buffers (negative SHAP)
    risk_drivers = [c for c in contributions if c["impact_tons"] > 2.0]
    risk_drivers.sort(key=lambda x: x["impact_tons"], reverse=True)

    buffers = [c for c in contributions if c["impact_tons"] <= -2.0]
    buffers.sort(key=lambda x: x["impact_tons"])

    if risk_drivers:
        top_factors = risk_drivers[:3]
    else:
        # If no significant risk drivers, show leading contributors
        contributions_by_abs = sorted(contributions, key=lambda x: abs(x["impact_tons"]), reverse=True)
        top_factors = contributions_by_abs[:2]

    result: Dict[str, Any] = {
        "target_production": planned,
        "predicted_production": predicted_production,
        "shortfall": predicted_shortfall,
        "risk": risk_level,
        "shortfall_pct": round((predicted_shortfall / max(1.0, planned)) * 100.0, 1),
        "key_factors": top_factors,
        "all_factor_attributions": contributions,
        "data_notice": "Trained on calibrated engineering physics simulation (mine_operations_simulated.csv)"
    }

    # Generate multi-day forecast trajectory if requested
    if include_trajectory:
        try:
            result["forecast_trajectory"] = predict_forecast_trajectory(operational_input)
        except Exception as exc:
            logger.warning(f"Multi-day forecast generation fallback: {exc}")
            result["forecast_trajectory"] = _generate_fallback_trajectory(planned, shovels, trucks)

    return result


def predict_forecast_trajectory(operational_input: Dict[str, Any], forecast_days: int = 7) -> Dict[str, Any]:
    """Generates a multi-day production forecast trajectory combining real weather forecast and ML model."""
    from services.environmental_service import get_environmental_data

    model, _ = _get_model_and_explainer()
    env_data = get_environmental_data()
    forecast = env_data.get("forecast_summary", {})
    daily_dates = forecast.get("daily_dates", [])[:forecast_days]
    daily_precip = forecast.get("daily_precipitation_mm", [])[:forecast_days]
    daily_temp = forecast.get("daily_temp_max_c", [])[:forecast_days]

    planned = float(operational_input.get("planned_production_tons") or 1200.0)
    shovels = int(operational_input.get("active_shovels") or 5)
    avail = float(operational_input.get("shovel_availability_pct") or 85.0)
    trucks = int(operational_input.get("haul_trucks") or (shovels * 2))

    current_moisture = float(env_data.get("current_conditions", {}).get("soil_moisture_pct", 38.0))

    labels = []
    targets = []
    forecasts = []
    shortfalls = []
    risks = []
    weather_summary = []

    moisture_tracker = current_moisture
    n_days = min(len(daily_dates), forecast_days) if daily_dates else 7

    for i in range(n_days):
        day_label = f"D{i+1}" if i > 0 else "Today"
        rain_i = float(daily_precip[i]) if i < len(daily_precip) else 2.0
        temp_i = float(daily_temp[i]) if i < len(daily_temp) else 30.0

        # Soil moisture dynamic model: drying factor 0.90 + daily rain contribution
        moisture_tracker = float(np.clip(moisture_tracker * 0.90 + rain_i * 1.4, 20.0, 95.0))
        blast_i = 1 if rain_i > 25.0 else 0

        features_i = [planned, shovels, avail, trucks, rain_i, moisture_tracker, temp_i, blast_i]
        df_i = pd.DataFrame([features_i], columns=FEATURE_NAMES)
        sf_i = max(0.0, round(float(model.predict(df_i)[0]), 1))
        prod_i = max(0.0, round(planned - sf_i, 1))

        ratio = sf_i / max(1.0, planned)
        risk_i = "HIGH" if (ratio >= 0.30 or sf_i >= 380.0) else "MODERATE" if (ratio >= 0.12 or sf_i >= 140.0) else "LOW"

        labels.append(day_label)
        targets.append(planned)
        forecasts.append(prod_i)
        shortfalls.append(sf_i)
        risks.append(risk_i)
        weather_summary.append({
            "date": daily_dates[i] if i < len(daily_dates) else f"Day {i+1}",
            "rainfall_mm": round(rain_i, 1),
            "soil_moisture_pct": round(moisture_tracker, 1),
            "temp_c": round(temp_i, 1)
        })

    return {
        "labels": labels,
        "target_production": targets,
        "forecast_production": forecasts,
        "predicted_shortfall": shortfalls,
        "risk_levels": risks,
        "weather_summary": weather_summary
    }


def _generate_fallback_trajectory(planned: float, shovels: int, trucks: int) -> Dict[str, Any]:
    """Generates baseline trajectory if live environmental forecast is unavailable."""
    base_cap = min(shovels * 210.0 * 0.85, trucks * 260.0 * 0.90)
    forecasts = [round(min(planned, base_cap * f), 1) for f in [0.98, 0.96, 0.88, 0.75, 0.80, 0.92, 0.95]]
    shortfalls = [max(0.0, round(planned - p, 1)) for p in forecasts]
    return {
        "labels": ["Today", "D2", "D3", "D4", "D5", "D6", "D7"],
        "target_production": [planned] * 7,
        "forecast_production": forecasts,
        "predicted_shortfall": shortfalls,
        "risk_levels": ["MODERATE" if s > 150 else "LOW" for s in shortfalls],
        "weather_summary": []
    }


if __name__ == "__main__":
    print("Training / Initializing Production Intelligence...")
    meta = train_production_model(force_retrain=True)
    print("Model Metadata:", meta)

    sample_input = {
        "planned_production_tons": 1200.0,
        "active_shovels": 4,
        "shovel_availability_pct": 75.0,
        "haul_trucks": 8,
        "rainfall_mm": 35.0,
        "soil_moisture_pct": 78.0,
        "temperature_c": 30.0,
        "blasting_delayed": 1
    }
    pred = predict_production(sample_input)
    print("\nPrediction Test:")
    print("Target:", pred["target_production"])
    print("Predicted:", pred["predicted_production"])
    print("Shortfall:", pred["shortfall"])
    print("Risk:", pred["risk"])
    print("Top Explanatory Factors (SHAP):")
    for kf in pred["key_factors"]:
        print(f" - {kf['factor']}: {kf['impact_tons']} tons ({kf['description']})")
