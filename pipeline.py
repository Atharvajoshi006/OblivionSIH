import numpy as np
import rasterio
from skimage import measure
import json

def process_satellite_bands(b4_path, b8_path, b11_path, b12_path, output_path="anomalies.json"):
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
    bare_ground_mask = ndvi < 0.25

    hydrothermal = np.where(swir2 == 0, 0, swir1 / swir2)
    ferrous_iron = np.where(nir == 0, 0, swir1 / nir)

    composite_anomaly = ((hydrothermal * 0.5) + (ferrous_iron * 0.5)) * bare_ground_mask

    threshold = np.percentile(composite_anomaly[bare_ground_mask], 90)
    ore_mask = composite_anomaly > threshold

    contours = measure.find_contours(ore_mask.astype(float), 0.5)

    polygons = []
    for contour in contours[:20]:
        coords = []
        for point in contour:
            row, col = int(point[0]), int(point[1])
            x, y = rasterio.transform.xy(transform, row, col)
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

if __name__ == "__main__":
    b4 = "data/sentinel/B04.tif"
    b8 = "data/sentinel/B08.tif"
    b11 = "data/sentinel/B11.tif"
    b12 = "data/sentinel/B12.tif"
    process_satellite_bands(b4, b8, b11, b12)