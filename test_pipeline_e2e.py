"""Comprehensive End-to-End Verification Test for GeoOre-AI Pipeline.
Tests:
1. Environmental Data Ingestion (Open-Meteo live API / calibrated cache)
2. Sentinel-2 Multi-Spectral Indices & GeoTIFF verification
3. Sausar Fold Belt Geological Context & Structural Perpendicular Lineament Distance
4. Multi-Criteria Mineral Prospectivity Mapping (MPM) with Known Deposit Validation (Zero circular leakage)
5. Production Shortfall XGBoost Regressor & SHAP Explainer
6. Dynamic Multi-Day Trajectory Forecast Generation
7. Decision-Support Operational Dispatch Recommendations
8. FastAPI REST Endpoints & HTML Dashboard Template Integration
"""

import os
import sys
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from main import app
from services.environmental_service import get_environmental_data
from services.satellite_service import process_satellite_data
from services.geology_service import get_geological_context, _distance_to_nearest_lineament
from services.prospectivity_service import (
    generate_prospectivity_map,
    validate_prospectivity_against_known_deposits
)
from services.production_service import predict_production, predict_forecast_trajectory
from services.recommendation_service import generate_recommendations


def run_all_tests():
    print("=================================================================")
    print("GEOORE-AI SYSTEM VERIFICATION & HARDENING TEST SUITE")
    print("=================================================================\n")

    # TEST 1: Environmental Service
    print("[TEST 1/8] Environmental Service (Open-Meteo)...")
    env = get_environmental_data()
    assert "current_conditions" in env, "current_conditions missing"
    assert "forecast_summary" in env, "forecast_summary missing"
    curr = env["current_conditions"]
    print(f"  * Source: {env.get('source')}")
    print(f"  * Temperature: {curr['temperature_c']} deg C")
    print(f"  * Rainfall: {curr['rainfall_mm']} mm")
    print(f"  * Soil Moisture: {curr['soil_moisture_pct']}% ({curr['haul_road_trafficability']})")
    print("  -> PASSED\n")

    # TEST 2: Sentinel-2 Multi-Spectral Pipeline
    print("[TEST 2/8] Sentinel-2 Satellite Multi-Spectral Processing...")
    sat = process_satellite_data()
    assert sat["status"] in ["success", "cached_success"], f"Unexpected sat status: {sat['status']}"
    assert "geotiff_path" in sat and os.path.exists(sat["geotiff_path"]), "GeoTIFF file not found"
    assert "geojson_path" in sat and os.path.exists(sat["geojson_path"]), "GeoJSON file not found"
    print(f"  * Status: {sat['status']}")
    print(f"  * Feature Count: {sat.get('feature_count')}")
    print(f"  * GeoTIFF: {sat['geotiff_path']}")
    print("  -> PASSED\n")

    # TEST 3: Geological Context & Lineaments
    print("[TEST 3/8] Geological Context & Sausar Belt Structural Lineaments...")
    # Test coordinates for Bharweli Mine
    geo_bharweli = get_geological_context(21.8125, 80.1902)
    assert geo_bharweli["inferred_formation"] == "Mansar Formation", "Bharweli should be inferred Mansar Formation"
    assert "deposit_vector_proximity" not in geo_bharweli["scores"], "Deposit proximity should NOT be an input score"
    print(f"  * Inferred Formation at Bharweli: {geo_bharweli['inferred_formation']}")
    print(f"  * Lineament Distance: {geo_bharweli['nearest_lineament_distance_km']} km")
    print(f"  * Lithology Score: {geo_bharweli['scores']['lithological_favorability']}")
    print(f"  * Structural Proximity Score: {geo_bharweli['scores']['structural_proximity']}")
    print(f"  * Inferred Magnetic Proxy: {geo_bharweli['scores']['geophysical_magnetic_proxy']}")
    print(f"  * Validation Reference: {geo_bharweli['validation_reference']['nearest_known_deposit']} ({geo_bharweli['validation_reference']['usage']})")
    print("  -> PASSED\n")

    # TEST 4: Prospectivity Map & Known Deposit Validation
    print("[TEST 4/8] Multi-Criteria Mineral Prospectivity & Deposit Validation...")
    pros_map = generate_prospectivity_map()
    assert pros_map["status"] == "success", "Prospectivity generation failed"
    assert pros_map["total_cells"] > 0, "No cells generated"
    
    val_report = validate_prospectivity_against_known_deposits()
    assert val_report["status"] == "validated", "Prospectivity validation failed"
    pilot = val_report["pilot_deposit"]
    assert pilot["prospectivity_score"] >= 0.65, "Pilot deposit should be High/Very-High"
    print(f"  * Total Cells: {pros_map['total_cells']}")
    print(f"  * High/Very-High Target Cells: {pros_map['high_prospect_cells']}")
    print(f"  * Pilot Deposit Validation: {pilot['name']}")
    print(f"    - Score: {pilot['prospectivity_score']} ({pilot['classification']})")
    print(f"    - Percentile Rank: {pilot['percentile_rank']}%")
    print(f"    - Contrast vs Regional Mean: {pilot['contrast_ratio_vs_background']}x")
    print(f"  * Integrity Audit: {val_report['scientific_integrity_audit']['validation_finding']}")
    print("  -> PASSED\n")

    # TEST 5: Production Shortfall Model & SHAP Attributions
    print("[TEST 5/8] Production Shortfall ML & SHAP Attributions...")
    pred = predict_production({
        "planned_production_tons": 1200.0,
        "active_shovels": 4,
        "haul_trucks": 8,
        "shovel_availability_pct": 75.0,
        "rainfall_mm": 20.0,
        "soil_moisture_pct": 65.0
    })
    assert "predicted_production" in pred, "predicted_production missing"
    assert "shortfall" in pred, "shortfall missing"
    assert "key_factors" in pred and len(pred["key_factors"]) > 0, "key_factors missing"
    assert "forecast_trajectory" in pred, "forecast_trajectory missing"
    print(f"  * Target: {pred['target_production']} t | Predicted: {pred['predicted_production']} t | Shortfall: {pred['shortfall']} t")
    print(f"  * Risk Level: {pred['risk']} ({pred['shortfall_pct']}%)")
    print("  * Top SHAP Risk Drivers:")
    for kf in pred["key_factors"]:
        print(f"    - {kf['factor']}: {kf['impact_tons']:+0.1f} t ({kf['description']})")
    print("  -> PASSED\n")

    # TEST 6: Multi-Day Forecast Trajectory
    print("[TEST 6/8] Multi-Day Dynamic Forecast Trajectory...")
    traj = pred["forecast_trajectory"]
    assert len(traj["labels"]) >= 5, "Trajectory should have at least 5 days"
    assert len(traj["forecast_production"]) == len(traj["labels"]), "Trajectory lengths mismatch"
    print(f"  * Horizon: {len(traj['labels'])} days ({traj['labels']})")
    print(f"  * Targets: {traj['target_production']}")
    print(f"  * Forecasts: {traj['forecast_production']}")
    print(f"  * Shortfalls: {traj['predicted_shortfall']}")
    print(f"  * Risk Trajectory: {traj['risk_levels']}")
    print("  -> PASSED\n")

    # TEST 7: Dynamic Decision-Support Recommendations
    print("[TEST 7/8] Recommendation Engine Verification...")
    recs = generate_recommendations(pred, env)
    assert recs["action_count"] > 0, "Recommendations should not be empty for high-risk scenario"
    print(f"  * Action Count: {recs['action_count']}")
    print(f"  * Primary Directive: {recs['primary_action']}")
    print("  * Action Items:")
    for act in recs["action_items"]:
        print(f"    - [{act['priority']}] ({act['category']}): {act['directive']}")
    print("  -> PASSED\n")

    # TEST 8: Full FastAPI Endpoint Integration & HTML Verification
    print("[TEST 8/8] FastAPI Endpoints & UI Verification...")
    client = TestClient(app)

    # 8a. Test GET /
    r = client.get("/")
    assert r.status_code == 200, f"GET / returned {r.status_code}"
    html = r.text
    for tag in ["ctrl-shovels", "ctrl-trucks", "ctrl-rain", "Prospectivity Index", "Surface Alteration & Prospects", "btn-gen-report"]:
        assert tag in html, f"HTML missing tag: {tag}"
    print("  * Dashboard HTML: Verified interactive controls & updated labels")

    # 8b. Test GET /api/anomalies
    r = client.get("/api/anomalies")
    assert r.status_code == 200, f"GET /api/anomalies returned {r.status_code}"
    assert isinstance(r.json(), list) and len(r.json()) > 0, "Anomalies list invalid"
    print(f"  * GET /api/anomalies: 200 OK ({len(r.json())} features)")

    # 8c. Test GET /api/prospectivity/validation
    r = client.get("/api/prospectivity/validation")
    assert r.status_code == 200, f"GET /api/prospectivity/validation returned {r.status_code}"
    assert r.json().get("status") == "validated", "Validation endpoint failed"
    print("  * GET /api/prospectivity/validation: 200 OK")

    # 8d. Test POST /api/predict-shortfall
    r = client.post("/api/predict-shortfall", json={
        "active_shovels": 5,
        "haul_trucks": 10,
        "rainfall_mm": 0.0
    })
    assert r.status_code == 200, f"POST /api/predict-shortfall returned {r.status_code}"
    res_pred = r.json()
    assert "forecast_trajectory" in res_pred, "forecast_trajectory missing in API response"
    print(f"  * POST /api/predict-shortfall: 200 OK (Shortfall: {res_pred['predicted_shortfall_tons']}t, Risk: {res_pred['risk_level']})")

    # 8e. Test POST /api/simulate
    r = client.post("/api/simulate", json={"active_shovels": 3, "haul_trucks": 6})
    assert r.status_code == 200, f"POST /api/simulate returned {r.status_code}"
    print(f"  * POST /api/simulate: 200 OK (Shortfall: {r.json()['predicted_shortfall_tons']}t)")

    # 8f. Test GET /api/satellite/spectral-summary
    r = client.get("/api/satellite/spectral-summary")
    assert r.status_code == 200, f"GET /api/satellite/spectral-summary returned {r.status_code}"
    print(f"  * GET /api/satellite/spectral-summary: 200 OK (Status: {r.json().get('status')})")

    print("  -> PASSED\n")

    print("=================================================================")
    print("ALL 8 PIPELINE & SYSTEM INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=================================================================")


if __name__ == "__main__":
    run_all_tests()
