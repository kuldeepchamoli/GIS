import osmnx as ox
import geopandas as gpd
import rasterio
import numpy as np
from shapely.geometry import Point
import requests

# ======================================================
# 1️⃣ LOAD OSM GRAPH
# ======================================================
print("Loading OSM graph...")
G = ox.load_graphml("data/processed/roads_base.graphml")

edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
edges = edges.to_crs("EPSG:3857")

# ======================================================
# 2️⃣ LOAD NDVI RASTER
# ======================================================
print("Loading NDVI raster...")
ndvi_path = "data/raw/sentinel_ndvi/ndvi_bengaluru_2024.tif"

with rasterio.open(ndvi_path) as raster:
    ndvi = raster.read(1)
    ndvi_transform = raster.transform
    ndvi_nodata = raster.nodata if raster.nodata is not None else -9999

# ======================================================
# 3️⃣ CENTROIDS
# ======================================================
print("Computing centroids...")
edges["centroid"] = edges.geometry.centroid

# ======================================================
# 4️⃣ SAMPLE NDVI
# ======================================================
print("Sampling NDVI...")

def sample_ndvi(point):
    x, y = point.x, point.y
    try:
        col, row = ~ndvi_transform * (x, y)
        row, col = int(row), int(col)
        if 0 <= row < ndvi.shape[0] and 0 <= col < ndvi.shape[1]:
            val = ndvi[row, col]
            return np.nan if val == ndvi_nodata else float(val)
    except:
        return np.nan
    return np.nan

edges["ndvi"] = edges["centroid"].apply(sample_ndvi)

# ======================================================
# 5️⃣ SUPER-FAST AQI (WORKING VERSION)
# ======================================================
print("Fetching AQI grid from Open-Meteo (bulk request)...")

edges_geo = edges.to_crs("EPSG:4326")
centroids = edges_geo.geometry.centroid

minx, miny, maxx, maxy = centroids.total_bounds

# 29x29 grid (841 points)
lats = np.linspace(miny, maxy, 29)
lons = np.linspace(minx, maxx, 29)

# Construct bulk query
lat_params = "&".join([f"latitude[]={lat}" for lat in lats])
lon_params = "&".join([f"longitude[]={lon}" for lon in lons])

url = (
    "https://air-quality-api.open-meteo.com/v1/air-quality?"
    + lat_params + "&" + lon_params +
    "&hourly=us_aqi"
)

print("Calling Open-Meteo once...")
resp = requests.get(url, timeout=30).json()

print("DEBUG TYPE:", type(resp))
print("DEBUG FIRST ENTRY:", resp[0])

# Build AQI grid from LIST response
aqi_grid = np.zeros((len(lats), len(lons)))

idx = 0
for i in range(len(lats)):
    for j in range(len(lons)):
        try:
            aqi_grid[i, j] = resp[idx]["hourly"]["us_aqi"][0]
        except:
            aqi_grid[i, j] = np.nan
        idx += 1

# ======================================================
# 6️⃣ BILINEAR INTERPOLATION OF AQI FOR EACH ROAD
# ======================================================
print("Interpolating AQI for road segments...")

def bilinear(lat, lon):
    # Normalize coordinates
    fx = (lon - minx) / (maxx - minx) * (len(lons) - 1)
    fy = (lat - miny) / (maxy - miny) * (len(lats) - 1)

    x0, y0 = int(fx), int(fy)
    x1, y1 = min(x0 + 1, len(lons)-1), min(y0 + 1, len(lats)-1)

    wx, wy = fx - x0, fy - y0

    v00 = aqi_grid[y0, x0]
    v10 = aqi_grid[y0, x1]
    v01 = aqi_grid[y1, x0]
    v11 = aqi_grid[y1, x1]

    return (
        v00 * (1-wx)*(1-wy) +
        v10 * wx * (1-wy) +
        v01 * (1-wx)*wy +
        v11 * wx * wy
    )

edges["aqi"] = [bilinear(c.y, c.x) for c in centroids]

# ======================================================
# 7️⃣ SAVE
# ======================================================

print("Restoring u, v, key columns...")

edges["u"] = edges.index.get_level_values(0)
edges["v"] = edges.index.get_level_values(1)
edges["key"] = edges.index.get_level_values(2)

print("Saving final CSV...")
edges[["u", "v", "key", "length", "ndvi", "aqi"]].to_csv(
    "data/processed/edges_ndvi_aqi.csv", index=False
)

print("🎉 SUCCESS: data/processed/edges_ndvi_aqi.csv created!")

# ======================================================
# 8️⃣ SAVE AS GEOJSON
# ======================================================
# ======================================================
# SAVE GEOJSON (Fix multiple geometry columns)
# ======================================================
print("Preparing GeoDataFrame for GeoJSON export...")

# Drop centroid column (second geometry)
if "centroid" in edges.columns:
    edges = edges.drop(columns=["centroid"])

# Remove index to avoid u,v,key conflicts
edges = edges.reset_index(drop=True)

print("Saving GeoJSON...")
edges.to_file(
    "data/processed/edges_ndvi_aqi.geojson",
    driver="GeoJSON"
)

print("🎉 SUCCESS: edges_ndvi_aqi.geojson created!")

