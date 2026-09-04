"""Environmental Data Service for GeoOre-AI.
Retrieves real atmospheric and soil moisture data from Open-Meteo API
for the Balaghat Manganese Mining Complex (or any user-specified coordinates).
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)

# Default Pilot Mine Coordinates: Balaghat Open-Cast / Underground Complex, Madhya Pradesh
DEFAULT_LAT = 21.805
DEFAULT_LON = 80.185
CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "environmental_cache.json")


def get_environmental_data(
    latitude: float = DEFAULT_LAT,
    longitude: float = DEFAULT_LON,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    forecast_days: int = 7
) -> Dict[str, Any]:
    """Retrieve real environmental data (rainfall, temperature, soil moisture).
    
    Uses Open-Meteo API (ECMWF / ERA5-Land atmospheric models).
    Falls back gracefully to cached data if the network is unavailable.
    """
    params: Dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": "Asia/Kolkata",
    }
    
    # Determine whether to use archive API or forecast API
    if start_date and end_date:
        base_url = "https://archive-api.open-meteo.com/v1/archive"
        params["start_date"] = start_date
        params["end_date"] = end_date
        params["daily"] = "precipitation_sum,temperature_2m_max,temperature_2m_min"
        params["hourly"] = "temperature_2m,precipitation,soil_moisture_0_to_7cm"
    else:
        base_url = "https://api.open-meteo.com/v1/forecast"
        params["forecast_days"] = min(forecast_days, 16)
        params["daily"] = "precipitation_sum,temperature_2m_max,temperature_2m_min,wind_speed_10m_max"
        params["hourly"] = "temperature_2m,relative_humidity_2m,precipitation,soil_moisture_0_to_1cm,soil_moisture_1_to_3cm,soil_moisture_3_to_9cm"
        params["current"] = "temperature_2m,relative_humidity_2m,precipitation,surface_pressure,wind_speed_10m"

    try:
        response = requests.get(base_url, params=params, timeout=12)
        response.raise_for_status()
        raw_data = response.json()
        
        # Process and normalize data into clean operational metrics
        processed = _process_environmental_payload(raw_data, latitude, longitude)
        
        # Cache successful fetch
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(processed, f, indent=2)
            
        return processed

    except Exception as exc:
        logger.warning(f"Open-Meteo API query failed ({exc}). Attempting cache recovery.")
        cached = _load_cached_environmental_data()
        if cached:
            cached["source"] = "cached_fallback"
            cached["warning"] = f"Live API unreachable ({str(exc)}). Serving cached environmental snapshot."
            return cached
        
        # If no cache exists, return a calibrated safe fallback with explicit status
        return _generate_calibrated_fallback(latitude, longitude, str(exc))


def _process_environmental_payload(data: Dict[str, Any], lat: float, lon: float) -> Dict[str, Any]:
    """Extract key mining-relevant indicators from Open-Meteo response."""
    daily = data.get("daily", {})
    hourly = data.get("hourly", {})
    current = data.get("current", {})
    
    precip_daily = daily.get("precipitation_sum", [])
    temp_max_daily = daily.get("temperature_2m_max", [])
    temp_min_daily = daily.get("temperature_2m_min", [])
    dates = daily.get("time", [])
    
    # Soil moisture: convert volumetric m3/m3 to percentage (e.g. 0.35 m3/m3 -> 35.0%)
    soil_moisture_layers = []
    for layer_key in ["soil_moisture_0_to_1cm", "soil_moisture_1_to_3cm", "soil_moisture_3_to_9cm", "soil_moisture_0_to_7cm"]:
        if layer_key in hourly and hourly[layer_key]:
            # Take non-null values
            valid_vals = [v for v in hourly[layer_key] if v is not None]
            if valid_vals:
                soil_moisture_layers.extend(valid_vals[-24:])  # latest 24 hours
                
    if soil_moisture_layers:
        mean_volumetric = sum(soil_moisture_layers) / len(soil_moisture_layers)
        soil_moisture_pct = round(mean_volumetric * 100.0, 2)
    else:
        soil_moisture_pct = 42.0

    current_rainfall = float(current.get("precipitation", 0.0) or (precip_daily[0] if precip_daily else 0.0))
    current_temp = float(current.get("temperature_2m", 28.5) if current else (temp_max_daily[0] if temp_max_daily else 28.5))
    total_forecast_rainfall = round(sum(precip_daily), 2) if precip_daily else 0.0

    # Soil trafficability rating for open-cast haul roads
    if soil_moisture_pct > 75.0 or current_rainfall > 30.0:
        saturation_status = "CRITICAL_SATURATION"
        road_trafficability = "SLUSH_HAZARD"
    elif soil_moisture_pct > 55.0 or current_rainfall > 10.0:
        saturation_status = "ELEVATED_MOISTURE"
        road_trafficability = "CAUTION_REQUIRED"
    else:
        saturation_status = "OPTIMAL_DRY"
        road_trafficability = "NORMAL"

    return {
        "source": "open_meteo_live",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "location": {
            "latitude": lat,
            "longitude": lon,
            "region": "Balaghat Manganese Belt, MP"
        },
        "current_conditions": {
            "rainfall_mm": round(current_rainfall, 2),
            "temperature_c": round(current_temp, 1),
            "soil_moisture_pct": soil_moisture_pct,
            "soil_saturation_status": saturation_status,
            "haul_road_trafficability": road_trafficability
        },
        "forecast_summary": {
            "forecast_days": len(dates),
            "total_rainfall_7d_mm": total_forecast_rainfall,
            "daily_dates": dates,
            "daily_precipitation_mm": precip_daily,
            "daily_temp_max_c": temp_max_daily,
            "daily_temp_min_c": temp_min_daily
        }
    }


def _load_cached_environmental_data() -> Optional[Dict[str, Any]]:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _generate_calibrated_fallback(lat: float, lon: float, error_msg: str) -> Dict[str, Any]:
    """Safe fallback when neither API nor cache is accessible."""
    return {
        "source": "offline_calibrated_baseline",
        "warning": f"Network error: {error_msg}. Using regional seasonal baseline.",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "location": {"latitude": lat, "longitude": lon, "region": "Balaghat (Offline Mode)"},
        "current_conditions": {
            "rainfall_mm": 5.2,
            "temperature_c": 31.0,
            "soil_moisture_pct": 38.5,
            "soil_saturation_status": "OPTIMAL_DRY",
            "haul_road_trafficability": "NORMAL"
        },
        "forecast_summary": {
            "forecast_days": 7,
            "total_rainfall_7d_mm": 18.0,
            "daily_dates": [],
            "daily_precipitation_mm": [5.2, 3.1, 2.0, 0.5, 1.2, 2.8, 3.2],
            "daily_temp_max_c": [32, 33, 33, 34, 32, 31, 32],
            "daily_temp_min_c": [22, 22, 23, 23, 22, 21, 22]
        }
    }


if __name__ == "__main__":
    data = get_environmental_data()
    print("Environmental Service Test:")
    print(f"Source: {data['source']}")
    print(f"Current Temp: {data['current_conditions']['temperature_c']} C")
    print(f"Current Rainfall: {data['current_conditions']['rainfall_mm']} mm")
    print(f"Soil Moisture: {data['current_conditions']['soil_moisture_pct']}%")
    print(f"7-Day Rain Total: {data['forecast_summary']['total_rainfall_7d_mm']} mm")
