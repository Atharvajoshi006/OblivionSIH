"""Operational Recommendation Service for GeoOre-AI.
Generates dynamic, decision-support corrective action recommendations
based on predicted shortfalls, primary SHAP risk factors, and environmental conditions.

COMPLIANCE & SAFETY NOTICE:
Recommendations are decision-support directives for mine planning and fleet dispatch.
They do not replace the statutory DGMS (Directorate General of Mines Safety) protocols
or certified Mine Manager operational authority.
"""

import os
import sys
from typing import Dict, Any, List

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def generate_recommendations(
    production_prediction: Dict[str, Any],
    environmental_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluates prediction drivers and environmental forecasts to produce actionable mining directives."""
    shortfall = float(production_prediction.get("shortfall", 0.0))
    risk_level = production_prediction.get("risk", "LOW")
    key_factors = production_prediction.get("key_factors", [])
    
    current_env = environmental_data.get("current_conditions", {})
    rainfall_mm = float(current_env.get("rainfall_mm", 0.0))
    soil_moisture_pct = float(current_env.get("soil_moisture_pct", 35.0))
    road_status = current_env.get("haul_road_trafficability", "NORMAL")

    forecast_env = environmental_data.get("forecast_summary", {})
    rain_7d = float(forecast_env.get("total_rainfall_7d_mm", 0.0))

    actions: List[Dict[str, str]] = []
    category_summary: List[str] = []

    # Priority 1: Environmental & Pit Drainage Mitigations
    if soil_moisture_pct >= 70.0 or rainfall_mm >= 25.0:
        actions.append({
            "priority": "P0_CRITICAL",
            "category": "HAULAGE_&_RAMP_MANAGEMENT",
            "directive": "Divert heavy haul dumpers from steep lower pit ramps to crowned Bench-4 gravel haulway to avert wheel-slip and slush bogging.",
            "rationale": f"Soil moisture at {soil_moisture_pct}% and active rainfall ({rainfall_mm} mm) exceed safe traction threshold."
        })
        actions.append({
            "priority": "P0_CRITICAL",
            "category": "MINE_DEWATERING",
            "directive": "Activate auxiliary stage-2 dewatering pumps at the main pit sump before water accumulation submerges lower loading faces.",
            "rationale": f"High saturation with {rain_7d} mm 7-day precipitation forecast threatens pit floor accessibility."
        })
        category_summary.append("Severe Weather Pit Protocol")

    elif soil_moisture_pct >= 50.0 or rainfall_mm >= 10.0:
        actions.append({
            "priority": "P1_HIGH",
            "category": "HAULAGE_SAFETY",
            "directive": "Enforce 20 km/h speed governor on wet haul roads and dispatch motor grader for road surface reprofiling.",
            "rationale": f"Moderate moisture ({soil_moisture_pct}%) creates slick road conditions."
        })
        category_summary.append("Precautionary Haulage Speed Control")

    # Priority 2: Equipment Fleet Allocation & Shovel Balance
    factor_names = [f.get("factor") for f in key_factors]

    if "active_shovels" in factor_names or "shovel_availability_pct" in factor_names:
        actions.append({
            "priority": "P1_HIGH",
            "category": "FLEET_REDEPLOYMENT",
            "directive": "Redeploy 1 standby hydraulic excavator from low-priority waste/overburden stripping to the high-grade Mansar ore face.",
            "rationale": "Loading fleet deficit identified by model attribution as a primary contributor to shift tonnage shortfall."
        })
        category_summary.append("Ore-Face Shovel Capacity Augmentation")

    if "haul_trucks" in factor_names:
        actions.append({
            "priority": "P1_HIGH",
            "category": "DISPATCH_OPTIMIZATION",
            "directive": "Re-optimize dumper match factor: reassign 3 dumpers from long-haul external waste dumps to short-haul run-of-mine (ROM) stockpile.",
            "rationale": "Haulage fleet cycle turnaround time is constraining excavator loading rate."
        })
        category_summary.append("Hauler Cycle Distance Reduction")

    if "planned_production_tons" in factor_names:
        actions.append({
            "priority": "P1_HIGH",
            "category": "PLANNING_ALIGNMENT",
            "directive": "Shift production target exceeds mechanical fleet throughput capacity. Re-baseline shift commitment or schedule auxiliary loading shift.",
            "rationale": "High planned target identified by model attribution as a primary contributor to operational deficit."
        })
        category_summary.append("Target Throughput Re-baseline")

    # Priority 3: Blasting & Environmental Heat Coordination
    if "blasting_delayed" in factor_names or rainfall_mm > 20.0:
        actions.append({
            "priority": "P1_HIGH",
            "category": "BLASTING_SCHEDULE",
            "directive": "Reschedule secondary blast to pre-storm clearance window; switch to water-resistant emulsion explosives in wet blast-holes.",
            "rationale": "Prevents misfires, damp powder desensitization, and unfragmented toe formation."
        })
        category_summary.append("Emulsion Blasting Schedule Adjustment")

    curr_temp = float(current_env.get("temperature_c", 28.0))
    if curr_temp >= 38.0 or "temperature_c" in factor_names:
        actions.append({
            "priority": "P1_HIGH",
            "category": "HEAT_STRESS_MITIGATION",
            "directive": "Implement DGMS heat-stress protocol: stagger equipment operator rest intervals and inspect engine hydraulic cooling systems.",
            "rationale": f"Ambient temperature ({curr_temp}°C) increases thermal equipment shutdown risk and operator fatigue."
        })
        category_summary.append("DGMS Heat Stress Protocol")

    # Fallback for moderate/high risk when specific threshold wasn't met
    if risk_level in ["HIGH", "MODERATE"] and not actions:
        actions.append({
            "priority": "P1_HIGH",
            "category": "OPERATIONAL_AUDIT",
            "directive": "Execute comprehensive shift bottleneck audit across excavator face assignment, haul road cycle times, and crusher bin availability.",
            "rationale": f"Model forecasts a {risk_level.lower()} deficit of {shortfall} tons under current operational parameters."
        })
        category_summary.append("Shift Bottleneck Review")

    # If operations are normal / low risk and no intervention required
    if not actions:
        actions.append({
            "priority": "P2_ROUTINE",
            "category": "OPTIMAL_OPERATIONS",
            "directive": "Maintain standard shift dispatch schedule. Current shovel/truck match ratio and environmental conditions are within optimal bounds.",
            "rationale": f"Predicted shortfall is minimal ({shortfall} tons, <10% planned target)."
        })
        category_summary.append("Standard Operating Procedures")

    # Format single primary executive action for legacy UI compatibility
    primary_action_str = actions[0]["directive"] if actions else "Standard mining operations."

    return {
        "overall_risk": risk_level,
        "predicted_shortfall_tons": shortfall,
        "mitigation_categories": category_summary,
        "action_count": len(actions),
        "primary_action": primary_action_str,
        "action_items": actions,
        "decision_support_level": "Automated Dispatch & Planning Support (Decision-Support Only)"
    }


if __name__ == "__main__":
    test_prediction = {
        "target_production": 1200.0,
        "predicted_production": 750.0,
        "shortfall": 450.0,
        "risk": "HIGH",
        "key_factors": [
            {"factor": "soil_moisture_pct", "impact_tons": 190.0},
            {"factor": "active_shovels", "impact_tons": 130.0}
        ]
    }
    test_env = {
        "current_conditions": {
            "rainfall_mm": 32.0,
            "soil_moisture_pct": 78.5,
            "haul_road_trafficability": "SLUSH_HAZARD"
        },
        "forecast_summary": {
            "total_rainfall_7d_mm": 74.0
        }
    }
    recs = generate_recommendations(test_prediction, test_env)
    print("Recommendation Engine Test:")
    print("Risk:", recs["overall_risk"])
    print("Primary Action:", recs["primary_action"])
    print("Action Items:")
    for a in recs["action_items"]:
        print(f" [{a['priority']}] ({a['category']}): {a['directive']}")
