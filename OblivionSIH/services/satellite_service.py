"""Satellite Processing Service for GeoOre-AI.
Fetches real Sentinel-2 Level-2A multi-spectral imagery via Microsoft Planetary Computer STAC,
performs radiometric and spatial windowing for the Balaghat Manganese Belt,
computes mineral alteration indices (Hydrothermal, Ferrous Iron, NDVI),
and outputs georeferenced GeoTIFF and map-ready GeoJSON features.

SCIENTIFIC NOTE:
Optical satellite data cannot detect underground manganese reserves.
It maps surface/near-surface rock outcrops, hydrothermal alteration zones,
gossans, and bare ground indicators that serve as surface exploration proxies.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import requests
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds, transform
from skimage import measure

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RAW_SENTINEL_DIR = os.path.join(DATA_DIR, "raw", "sentinel")
OUTPUT_GEOJSON = os.path.join(PROCESSED_DIR, "manganese_spectral_anomalies.geojson")
OUTPUT_GEOTIFF = os.path.join(PROCESSED_DIR, "sentinel_balaghat_indices.tif")

# Default pilot bounding box: Balaghat - Bharweli Manganese belt [W, S, E, N]
DEFAULT_BBOX = [80.16, 21.79, 80.22, 21.84]


def process_satellite_data(
    bbox: List[float] = DEFAULT_BBOX,
    max_cloud_cover: int = 15,
    force_refresh: bool = False
) -> Dict[str, Any]:
    """Acquires and processes real Sentinel-2 bands to extract surface alteration indices.
    
    Returns structured spectral summary and GeoJSON feature collection.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(RAW_SENTINEL_DIR, exist_ok=True)

    # Check if processed output already exists and refresh is not forced
    if not force_refresh and os.path.exists(OUTPUT_GEOJSON) and os.path.exists(OUTPUT_GEOTIFF):
        try:
            with open(OUTPUT_GEOJSON, "r") as f:
                geojson_data = json.load(f)
            return {
                "status": "cached_success",
                "message": "Loaded pre-processed Sentinel-2 mineral indices",
                "geotiff_path": OUTPUT_GEOTIFF,
                "geojson_path": OUTPUT_GEOJSON,
                "feature_count": len(geojson_data.get("features", [])),
                "features": geojson_data
            }
        except Exception:
            pass

    try:
        # Step 1: Query Microsoft Planetary Computer STAC for cloud-free Sentinel-2 L2A scene
        stac_search_url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
        search_body = {
            "collections": ["sentinel-2-l2a"],
            "bbox": bbox,
            "query": {"eo:cloud_cover": {"lt": max_cloud_cover}},
            "limit": 1
        }
        res = requests.post(stac_search_url, json=search_body, timeout=15)
        res.raise_for_status()
        features = res.json().get("features", [])
        if not features:
            raise RuntimeError(f"No Sentinel-2 scene with cloud cover < {max_cloud_cover}% found for bbox {bbox}")
        
        scene = features[0]
        scene_id = scene["id"]
        scene_datetime = scene["properties"].get("datetime")
        cloud_pct = scene["properties"].get("eo:cloud_cover", 0.0)

        # Obtain read SAS token for assets
        token_res = requests.get("https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a", timeout=12)
        token = token_res.json().get("token", "") if token_res.status_code == 200 else ""

        # Step 2: Read windowed bands (B04: Red, B08: NIR, B11: SWIR1, B12: SWIR2)
        band_arrays, profile = _read_windowed_bands(scene["assets"], token, bbox)

        # Step 3: Compute Mineral Spectral Indices
        indices = _compute_spectral_indices(band_arrays)

        # Step 4: Save Multi-Band GeoTIFF (Band 1: Composite Anomaly, Band 2: Hydrothermal, Band 3: Ferrous Iron, Band 4: NDVI)
        _save_geotiff(indices, profile, OUTPUT_GEOTIFF)

        # Step 5: Extract Surface Anomaly Polygons into GeoJSON
        geojson_data = _extract_anomaly_polygons(indices, profile, scene_id, scene_datetime)
        with open(OUTPUT_GEOJSON, "w") as f:
            json.dump(geojson_data, f, indent=2)

        return {
            "status": "success",
            "scene_id": scene_id,
            "acquisition_date": scene_datetime,
            "cloud_cover_pct": cloud_pct,
            "geotiff_path": OUTPUT_GEOTIFF,
            "geojson_path": OUTPUT_GEOJSON,
            "feature_count": len(geojson_data.get("features", [])),
            "features": geojson_data,
            "spectral_summary": {
                "mean_ndvi": float(np.nanmean(indices["ndvi"])),
                "bare_ground_coverage_pct": float(np.mean(indices["bare_ground_mask"]) * 100.0),
                "max_anomaly_index": float(np.nanmax(indices["composite"]))
            }
        }

    except Exception as e:
        logger.error(f"Sentinel-2 processing pipeline failed: {e}", exc_info=True)
        # Generate clean fallback from available local bounds
        return _generate_fallback_spectral_data(bbox, str(e))


def _read_windowed_bands(assets: Dict[str, Any], token: str, bbox: List[float]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Reads spatial window for B04, B08, B11, B12 and resamples SWIR to 10m."""
    auth_suffix = f"?{token}" if token else ""
    b4_url = assets["B04"]["href"] + auth_suffix
    b8_url = assets["B08"]["href"] + auth_suffix
    b11_url = assets["B11"]["href"] + auth_suffix
    b12_url = assets["B12"]["href"] + auth_suffix

    with rasterio.open(b4_url) as src_b4:
        left, bottom, right, top = transform_bounds("EPSG:4326", src_b4.crs, bbox[0], bbox[1], bbox[2], bbox[3])
        win_10m = from_bounds(left, bottom, right, top, src_b4.transform)
        red = src_b4.read(1, window=win_10m).astype(np.float32)
        win_transform = rasterio.windows.transform(win_10m, src_b4.transform)
        crs = src_b4.crs

    with rasterio.open(b8_url) as src_b8:
        nir = src_b8.read(1, window=win_10m).astype(np.float32)

    # SWIR bands (B11, B12) are 20m resolution, read with destination shape matching 10m grid
    target_shape = red.shape
    with rasterio.open(b11_url) as src_b11:
        win_swir = from_bounds(left, bottom, right, top, src_b11.transform)
        swir1 = src_b11.read(1, window=win_swir, out_shape=target_shape, resampling=Resampling.bilinear).astype(np.float32)

    with rasterio.open(b12_url) as src_b12:
        win_swir = from_bounds(left, bottom, right, top, src_b12.transform)
        swir2 = src_b12.read(1, window=win_swir, out_shape=target_shape, resampling=Resampling.bilinear).astype(np.float32)

    profile = {
        "driver": "GTiff",
        "height": red.shape[0],
        "width": red.shape[1],
        "count": 4,
        "dtype": "float32",
        "crs": crs,
        "transform": win_transform,
        "compress": "lzw"
    }

    return {"red": red, "nir": nir, "swir1": swir1, "swir2": swir2}, profile


def _compute_spectral_indices(bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Computes standard remote-sensing mineral alteration ratios."""
    red = bands["red"]
    nir = bands["nir"]
    swir1 = bands["swir1"]
    swir2 = bands["swir2"]

    # NDVI = (NIR - Red) / (NIR + Red)
    denom_ndvi = nir + red
    ndvi = np.where(denom_ndvi > 0, (nir - red) / (denom_ndvi + 1e-6), 0.0)

    # Bare ground mask: suppress dense vegetation to avoid chlorophyll false-positives
    bare_ground_mask = ndvi < 0.35

    # Hydrothermal / Clay Alteration: SWIR1 / SWIR2
    hydrothermal = np.where(swir2 > 0, swir1 / (swir2 + 1e-6), 0.0)

    # Ferrous Iron / Gossan Proxy: SWIR1 / NIR
    ferrous = np.where(nir > 0, swir1 / (nir + 1e-6), 0.0)

    # Combined surface alteration score masked by bare ground
    composite = ((hydrothermal * 0.5) + (ferrous * 0.5)) * bare_ground_mask
    composite = np.clip(composite, 0.0, 5.0)

    return {
        "ndvi": ndvi,
        "bare_ground_mask": bare_ground_mask,
        "hydrothermal": hydrothermal,
        "ferrous": ferrous,
        "composite": composite
    }


def _save_geotiff(indices: Dict[str, np.ndarray], profile: Dict[str, Any], output_path: str):
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(indices["composite"].astype(np.float32), 1)
        dst.write(indices["hydrothermal"].astype(np.float32), 2)
        dst.write(indices["ferrous"].astype(np.float32), 3)
        dst.write(indices["ndvi"].astype(np.float32), 4)
        dst.set_band_description(1, "Composite_Surface_Alteration_Index")
        dst.set_band_description(2, "Hydrothermal_Alteration_Ratio_B11_B12")
        dst.set_band_description(3, "Ferrous_Iron_Ratio_B11_B08")
        dst.set_band_description(4, "NDVI_Vegetation_Index")


def _extract_anomaly_polygons(
    indices: Dict[str, np.ndarray],
    profile: Dict[str, Any],
    scene_id: str,
    scene_date: Optional[str]
) -> Dict[str, Any]:
    """Extracts significant anomaly contours and translates pixel coords to WGS84 GeoJSON."""
    composite = indices["composite"]
    bare_mask = indices["bare_ground_mask"]
    
    if np.sum(bare_mask) == 0:
        threshold = 1.2
    else:
        threshold = float(np.percentile(composite[bare_mask], 92))

    binary_mask = (composite >= threshold).astype(float)
    contours = measure.find_contours(binary_mask, 0.5)

    features = []
    anomaly_id = 1
    src_crs = profile["crs"]
    trans = profile["transform"]

    for contour in contours:
        if len(contour) < 6:  # Skip tiny noise specs
            continue
        
        coords = []
        for pt in contour:
            row, col = pt[0], pt[1]
            x, y = rasterio.transform.xy(trans, row, col)
            # Transform UTM EPSG:32644 to WGS84 EPSG:4326 (lon, lat)
            lon, lat = transform(src_crs, "EPSG:4326", [x], [y])
            coords.append([round(lon[0], 5), round(lat[0], 5)])

        if len(coords) >= 4:
            coords.append(coords[0])  # Close polygon
            confidence = "HIGH" if threshold > 1.6 else "MODERATE"
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords]
                },
                "properties": {
                    "id": f"S2-ANOM-{anomaly_id:03d}",
                    "confidence": confidence,
                    "spectral_index": round(float(threshold), 3),
                    "indicator_type": "Surface Hydrothermal & Iron Alteration Gossan",
                    "satellite_source": "Sentinel-2 L2A",
                    "scene_id": scene_id,
                    "acquisition_date": scene_date,
                    "scientific_note": "Surface alteration indicator; requires geological/structural verification"
                }
            })
            anomaly_id += 1
            if len(features) >= 25:
                break

    return {
        "type": "FeatureCollection",
        "metadata": {
            "source": "Sentinel-2 Level-2A Multi-Spectral Processing",
            "scene_id": scene_id,
            "crs": "EPSG:4326",
            "bands_used": ["B04", "B08", "B11", "B12"],
            "indices": ["NDVI", "SWIR1/SWIR2 (Hydrothermal)", "SWIR1/NIR (Ferrous Iron)"]
        },
        "features": features
    }


def _generate_fallback_spectral_data(bbox: List[float], error_msg: str) -> Dict[str, Any]:
    """Generates structured placeholder when live Planetary Computer API is unreachable."""
    # Centered on Balaghat mine
    center_lon = (bbox[0] + bbox[2]) / 2.0
    center_lat = (bbox[1] + bbox[3]) / 2.0
    
    fallback_features = [{
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [center_lon - 0.005, center_lat - 0.004],
                [center_lon + 0.006, center_lat - 0.003],
                [center_lon + 0.008, center_lat + 0.005],
                [center_lon - 0.003, center_lat + 0.006],
                [center_lon - 0.005, center_lat - 0.004]
            ]]
        },
        "properties": {
            "id": "S2-FALLBACK-001",
            "confidence": "MODERATE",
            "spectral_index": 1.45,
            "indicator_type": "Surface Alteration Anomaly (Offline Mode)",
            "warning": f"STAC API query error: {error_msg}"
        }
    }]
    return {
        "status": "offline_fallback",
        "features": {"type": "FeatureCollection", "features": fallback_features},
        "feature_count": 1
    }


if __name__ == "__main__":
    print("Running Sentinel-2 Satellite Service...")
    result = process_satellite_data()
    print("Satellite Processing Status:", result["status"])
    print("Features Extracted:", result.get("feature_count", 0))
    if "spectral_summary" in result:
        print("Spectral Summary:", result["spectral_summary"])
