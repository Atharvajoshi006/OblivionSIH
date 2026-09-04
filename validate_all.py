"""SIH Step 3 - Full Validation & Stress Test Script"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── 1. PROSPECTIVITY VALIDATION ────────────────────────────────────────────
print("=" * 60)
print("1. PROSPECTIVITY VALIDATION (all known deposits)")
print("=" * 60)
from services.prospectivity_service import validate_prospectivity_against_known_deposits
result = validate_prospectivity_against_known_deposits()

bg = result["regional_background_statistics"]
print(f"  Grid cells: {bg['cell_count']}  | mean={bg['mean_score']}  median={bg['median_score']}  std={bg['std_deviation']}")
print()
high_vh = 0
for d in result["all_regional_deposits"]:
    cls = d["classification"]
    if cls in ("HIGH", "VERY_HIGH"):
        high_vh += 1
    print(f"  {d['name'][:35]:35s} | {cls:10s} | score={d['prospectivity_score']} | pct={d['percentile_rank']}% | contrast={d['contrast_ratio_vs_background']}x")

total = len(result["all_regional_deposits"])
print(f"\n  HIGH/VERY_HIGH: {high_vh}/{total} deposits")
audit = result["scientific_integrity_audit"]
print(f"  circular_leakage: {audit['circular_leakage_detected']}")
print(f"  deposits_as_inputs: {audit['known_deposits_used_as_inputs']}")
print(f"  Finding: {audit['validation_finding']}")

# ── 2. PRODUCTION STRESS TEST ──────────────────────────────────────────────
print()
print("=" * 60)
print("2. PRODUCTION MODEL STRESS TEST")
print("=" * 60)
from services.production_service import predict_production

scenarios = [
    ("Normal conditions",    {"planned_production_tons":1200,"active_shovels":5,"shovel_availability_pct":88,"haul_trucks":10,"rainfall_mm":2,"soil_moisture_pct":30,"temperature_c":28,"blasting_delayed":0}),
    ("Low shovel avail",     {"planned_production_tons":1200,"active_shovels":3,"shovel_availability_pct":62,"haul_trucks":8,"rainfall_mm":2,"soil_moisture_pct":30,"temperature_c":28,"blasting_delayed":0}),
    ("Low truck avail",      {"planned_production_tons":1200,"active_shovels":5,"shovel_availability_pct":88,"haul_trucks":5,"rainfall_mm":2,"soil_moisture_pct":30,"temperature_c":28,"blasting_delayed":0}),
    ("Heavy rainfall",       {"planned_production_tons":1200,"active_shovels":5,"shovel_availability_pct":85,"haul_trucks":10,"rainfall_mm":65,"soil_moisture_pct":80,"temperature_c":27,"blasting_delayed":1}),
    ("High soil moisture",   {"planned_production_tons":1200,"active_shovels":5,"shovel_availability_pct":85,"haul_trucks":10,"rainfall_mm":10,"soil_moisture_pct":78,"temperature_c":30,"blasting_delayed":0}),
    ("Blasting delay",       {"planned_production_tons":1200,"active_shovels":5,"shovel_availability_pct":85,"haul_trucks":10,"rainfall_mm":5,"soil_moisture_pct":38,"temperature_c":32,"blasting_delayed":1}),
    ("All adverse combined", {"planned_production_tons":1400,"active_shovels":3,"shovel_availability_pct":60,"haul_trucks":6,"rainfall_mm":70,"soil_moisture_pct":88,"temperature_c":38,"blasting_delayed":1}),
]

all_consistent = True
for name, inp in scenarios:
    p = predict_production(inp, include_trajectory=False)
    sf = p["shortfall"]
    prod = p["predicted_production"]
    risk = p["risk"]
    planned = inp["planned_production_tons"]
    top = [f["factor"] for f in p["key_factors"]]
    # Consistency check
    ok = True
    if prod < 0: ok = False
    if sf < 0: ok = False
    if abs(prod + sf - planned) > 1.5: ok = False  # prod + shortfall should ~ planned
    if risk == "HIGH" and sf < 100: ok = False
    if risk == "LOW" and sf > 400: ok = False
    status = "OK" if ok else "FAIL"
    if not ok: all_consistent = False
    print(f"  [{status}] {name:25s} | prod={prod:6.0f} | sf={sf:5.0f} | risk={risk:8s} | drivers={top}")

print(f"\n  Logic consistency: {'PASS' if all_consistent else 'FAIL'}")

# ── 3. SCIENTIFIC INTEGRITY SCAN ───────────────────────────────────────────
print()
print("=" * 60)
print("3. SCIENTIFIC INTEGRITY SCAN")
print("=" * 60)

files_to_scan = [
    "services/satellite_service.py",
    "services/prospectivity_service.py",
    "services/geology_service.py",
    "services/production_service.py",
    "services/recommendation_service.py",
    "main.py",
    "templates/index.html",
]
# Affirmative overclaim phrases only (negation/disclaimer forms are acceptable).
# e.g. "cannot detect underground" is a correct disclaimer and must NOT be flagged.
bad_phrases = [
    "satellites can detect underground",
    "detects underground ore",
    "direct detection of underground",
    "direct ore detection",
    "100% accurate",
    "guaranteed",
    "real drillhole data",
    "real moil production data",
    "proven accuracy",
]
good_phrases = {
    # Phrase -> [files where phrase (or equivalent concept) must be present]
    "exploration prox":      ["services/satellite_service.py"],       # "surface exploration proxies"
    "surface alteration":    ["services/prospectivity_service.py"],   # "surface alteration" in evidence weights
    "simulated":             ["services/production_service.py"],      # calibrated simulation label
    "geophysical magnetic proxy": ["services/geology_service.py"],    # inferred geophysical proxy
    "inferred proxy":        ["services/prospectivity_service.py"],   # "inferred proxy" in weights
    "requires drillhole":    ["services/prospectivity_service.py"],
}
issues = []
for fname in files_to_scan:
    try:
        text = open(fname, encoding="utf-8").read().lower()
        for bp in bad_phrases:
            if bp in text:
                issues.append(f"BAD PHRASE in {fname}: '{bp}'")
    except FileNotFoundError:
        issues.append(f"FILE NOT FOUND: {fname}")

if issues:
    for i in issues: print(f"  [ISSUE] {i}")
else:
    print("  No prohibited overclaims found.")

# Good phrase presence
for phrase, expected_files in good_phrases.items():
    for fname in expected_files:
        try:
            text = open(fname, encoding="utf-8").read().lower()
            status = "PRESENT" if phrase in text else "MISSING"
            print(f"  [{status}] '{phrase}' in {fname}")
        except FileNotFoundError:
            print(f"  [MISSING-FILE] {fname}")

# ── 4. END-TO-END PIPELINE TEST ────────────────────────────────────────────
print()
print("=" * 60)
print("4. END-TO-END PIPELINE")
print("=" * 60)

from services.environmental_service import get_environmental_data
env = get_environmental_data()
print(f"  [{'OK' if env.get('source') else 'FAIL'}] Environmental: source={env.get('source')} keys={list(env.get('current_conditions',{}).keys())[:4]}")

from services.satellite_service import process_satellite_data
sat = process_satellite_data()
print(f"  [{'OK' if sat.get('status') else 'FAIL'}] Satellite: status={sat.get('status')} features={sat.get('feature_count')}")

from services.prospectivity_service import generate_prospectivity_map
pm = generate_prospectivity_map()
print(f"  [{'OK' if pm.get('status')=='success' else 'FAIL'}] Prospectivity: cells={pm.get('total_cells')} high={pm.get('high_prospect_cells')}")

from services.production_service import predict_production
pred = predict_production({"planned_production_tons":1200,"active_shovels":5,"shovel_availability_pct":85,"haul_trucks":10,"rainfall_mm":3,"soil_moisture_pct":35,"temperature_c":29,"blasting_delayed":0})
print(f"  [{'OK' if pred.get('shortfall') is not None else 'FAIL'}] Production: shortfall={pred.get('shortfall')} risk={pred.get('risk')}")

from services.recommendation_service import generate_recommendations
rec = generate_recommendations(pred, env)
print(f"  [{'OK' if rec.get('action_items') else 'FAIL'}] Recommendations: {rec.get('action_count')} actions, risk={rec.get('overall_risk')}")

trajectory = pred.get("forecast_trajectory", {})
print(f"  [{'OK' if trajectory.get('labels') else 'FAIL'}] 7-day forecast: days={len(trajectory.get('labels',[]))}")

print()
print("  SIMULATED DATA NOTICE: All production results use calibrated engineering simulation.")
print("  Prospectivity uses real Sentinel-2 surface proxies + GSI/MOIL geological references.")
print("  No underground direct detection is claimed.")
print()
print("Validation complete.")
