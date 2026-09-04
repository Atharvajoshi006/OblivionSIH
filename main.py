"""FastAPI Backend Application for GeoOre-AI (OblivionSIH).
Exposes REST APIs for:
- Live environmental data ingestion (Open-Meteo)
- Sentinel-2 multi-spectral surface alteration & mineral indices
- Multi-criteria manganese prospectivity intelligence (Sausar belt)
- Production shortfall forecasting & SHAP explainability
- Contextual operational dispatch recommendations
"""

import os
import sys
import json
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services.environmental_service import get_environmental_data
from services.satellite_service import process_satellite_data
from services.prospectivity_service import (
    generate_prospectivity_map,
    validate_prospectivity_against_known_deposits,
    PROSPECTIVITY_GEOJSON,
    LEGACY_ANOMALIES_JSON
)
from services.production_service import predict_production, train_production_model
from services.recommendation_service import generate_recommendations

app = FastAPI(
    title="GeoOre-AI API",
    description="AI/ML & Space Intelligence Platform for Manganese Prospectivity & Production Shortfall Mitigation",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class OperationalInput(BaseModel):
    planned_production_tons: Optional[float] = 1200.0
    active_shovels: int = 5
    shovel_availability_pct: Optional[float] = 85.0
    haul_trucks: Optional[int] = None
    rainfall_mm: Optional[float] = None
    soil_moisture_pct: Optional[float] = None
    temperature_c: Optional[float] = None
    blasting_delayed: Optional[int] = 0


@app.get("/")
def serve_home():
    """Serves the main operations center dashboard."""
    index_path = os.path.join(PROJECT_ROOT, "templates", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "GeoOre-AI API is active. Template not found."}


@app.get("/api/anomalies")
def get_anomalies():
    """Serves map-ready prospectivity anomaly polygons.
    
    Compatible with existing Leaflet layer renderer in index.html.
    """
    if os.path.exists(LEGACY_ANOMALIES_JSON):
        with open(LEGACY_ANOMALIES_JSON, "r") as f:
            return json.load(f)
    
    # Generate prospectivity if not yet built
    result = generate_prospectivity_map()
    with open(LEGACY_ANOMALIES_JSON, "r") as f:
        return json.load(f)


@app.get("/api/prospectivity/geojson")
def get_prospectivity_geojson():
    """Serves standard GeoJSON FeatureCollection of multi-criteria prospectivity cells."""
    if os.path.exists(PROSPECTIVITY_GEOJSON):
        with open(PROSPECTIVITY_GEOJSON, "r") as f:
            return json.load(f)
    
    result = generate_prospectivity_map()
    with open(PROSPECTIVITY_GEOJSON, "r") as f:
        return json.load(f)


@app.get("/api/environmental-current")
def get_current_environmental_metrics(lat: float = 21.805, lon: float = 80.185):
    """Retrieves live atmospheric and soil moisture conditions from Open-Meteo."""
    return get_environmental_data(latitude=lat, longitude=lon)


@app.post("/api/predict-shortfall")
def predict_shortfall(data: OperationalInput):
    """Predicts daily production shortfall, risk level, SHAP root causes, and recommendations.
    
    If weather parameters are not provided, automatically incorporates live Open-Meteo data.
    """
    # 1. Fetch live environmental context for Balaghat if not explicitly overridden
    env_data = get_environmental_data()
    curr_env = env_data.get("current_conditions", {})

    actual_rainfall = data.rainfall_mm if data.rainfall_mm is not None else float(curr_env.get("rainfall_mm", 0.0))
    actual_soil_moisture = data.soil_moisture_pct if data.soil_moisture_pct is not None else float(curr_env.get("soil_moisture_pct", 35.0))
    actual_temp = data.temperature_c if data.temperature_c is not None else float(curr_env.get("temperature_c", 28.0))
    actual_trucks = data.haul_trucks if data.haul_trucks is not None else data.active_shovels * 2

    # 2. Execute ML Production Prediction
    operational_dict = {
        "planned_production_tons": data.planned_production_tons or 1200.0,
        "active_shovels": data.active_shovels,
        "shovel_availability_pct": data.shovel_availability_pct or 85.0,
        "haul_trucks": actual_trucks,
        "rainfall_mm": actual_rainfall,
        "soil_moisture_pct": actual_soil_moisture,
        "temperature_c": actual_temp,
        "blasting_delayed": data.blasting_delayed or 0
    }

    prediction = predict_production(operational_dict)

    # 3. Generate dynamic decision-support recommendations
    # Overlay the evaluated environmental data
    effective_env = {
        "current_conditions": {
            "rainfall_mm": actual_rainfall,
            "soil_moisture_pct": actual_soil_moisture,
            "temperature_c": actual_temp,
            "haul_road_trafficability": "SLUSH_HAZARD" if actual_soil_moisture > 65.0 else "NORMAL"
        },
        "forecast_summary": env_data.get("forecast_summary", {})
    }
    recs = generate_recommendations(prediction, effective_env)

    # 4. Return unified response preserving backward compatibility with templates/index.html
    return {
        "predicted_shortfall_tons": prediction["shortfall"],
        "predicted_production_tons": prediction["predicted_production"],
        "target_production_tons": prediction["target_production"],
        "risk_level": prediction["risk"],
        "shortfall_pct": prediction["shortfall_pct"],
        "action": recs["primary_action"],
        "key_factors": prediction["key_factors"],
        "all_attributions": prediction["all_factor_attributions"],
        "action_items": recs["action_items"],
        "mitigation_categories": recs["mitigation_categories"],
        "environmental_context": effective_env["current_conditions"],
        "forecast_trajectory": prediction.get("forecast_trajectory", {})
    }


@app.post("/api/simulate")
def run_operational_simulation(data: OperationalInput):
    """Alias endpoint for scenario planning simulations."""
    return predict_shortfall(data)


@app.get("/api/satellite/spectral-summary")
def get_satellite_spectral_summary():
    """Returns metadata and statistical summary from Sentinel-2 processing."""
    return process_satellite_data()


@app.get("/api/prospectivity/validation")
def get_prospectivity_validation():
    """Returns empirical validation report verifying spatial concordance against known deposits."""
    return validate_prospectivity_against_known_deposits()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)