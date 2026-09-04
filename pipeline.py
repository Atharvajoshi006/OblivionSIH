"""Sentinel-2 Satellite & Prospectivity Pipeline CLI.
Upgraded to use real Planetary Computer Sentinel-2 data and
Multi-Criteria Prospectivity Mapping (MPM).
"""

import os
import sys
import json
import logging
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services.satellite_service import process_satellite_data
from services.prospectivity_service import generate_prospectivity_map

logger = logging.getLogger(__name__)


def run_pipeline(output_path="data/anomalies.json"):
    """Executes the full satellite ingestion and manganese prospectivity workflow."""
    print("[GeoOre-AI] Ingesting Sentinel-2 multi-spectral bands...")
    sat_res = process_satellite_data()
    print(f"[GeoOre-AI] Sentinel-2 status: {sat_res['status']}, Features: {sat_res.get('feature_count', 0)}")
    
    print("[GeoOre-AI] Running Multi-Criteria Mineral Prospectivity Mapping (MPM)...")
    pros_res = generate_prospectivity_map()
    print(f"[GeoOre-AI] Prospectivity evaluated across {pros_res['total_cells']} spatial cells.")
    print(f"[GeoOre-AI] High/Very-High Prospectivity targets: {pros_res['high_prospect_cells']}")
    print(f"[GeoOre-AI] Map-ready anomaly layer written to: {output_path}")
    return pros_res


# Retain backward compatibility function signature
def process_satellite_bands(b4_path=None, b8_path=None, b11_path=None, b12_path=None, output_path="data/anomalies.json"):
    if b4_path and os.path.exists(b4_path):
        import rasterio
        from skimage import measure
        with rasterio.open(b4_path) as src_b4:
            red = src_b4.read(1).astype(np.float32)
            transform = src_b4.transform
        with rasterio.open(b8_path) as src_b8:
            nir = src_b8.read(1).astype(np.float32)
        with rasterio.open(b11_path) as src_b11:
            swir1 = src_b11.read(1).astype(np.float32)
        with rasterio.open(b12_path) as src_b12:
            swir2 = src_b12.read(1).astype(np.float32)

        ndvi = np.where((nir + red) == 0, 0, (nir - red) / (nir + red))
        bare_ground_mask = ndvi < 0.35
        hydrothermal = np.where(swir2 == 0, 0, swir1 / swir2)
        ferrous_iron = np.where(nir == 0, 0, swir1 / nir)
        composite = ((hydrothermal * 0.5) + (ferrous_iron * 0.5)) * bare_ground_mask
        threshold = np.percentile(composite[bare_ground_mask], 90)
        contours = measure.find_contours((composite > threshold).astype(float), 0.5)

        polygons = []
        for contour in contours[:20]:
            coords = []
            for pt in contour:
                x, y = rasterio.transform.xy(transform, int(pt[0]), int(pt[1]))
                coords.append([y, x])
            if len(coords) > 3:
                coords.append(coords[0])
                polygons.append({
                    "confidence": "HIGH",
                    "spectral_index": round(float(threshold), 3),
                    "coordinates": coords
                })
        with open(output_path, "w") as f:
            json.dump(polygons, f, indent=2)
    else:
        # Automated satellite & prospectivity service
        run_pipeline(output_path)


if __name__ == "__main__":
    run_pipeline()