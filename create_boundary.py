import osmnx as ox

# get boundary of Bengaluru from OSM
gdf = ox.geocode_to_gdf("Bengaluru, India")

# save as geojson
gdf.to_file("data/raw/bengaluru_boundary.geojson", driver="GeoJSON")

print("Boundary created -> data/raw/bengaluru_boundary.geojson")
