import osmnx as ox

# Define place
place = "Bengaluru, India"

# Download drivable roads
G = ox.graph_from_place(place, network_type="drive")

# Save graph
ox.save_graphml(G, "data/processed/roads_base.graphml")

print("Saved: data/processed/roads_base.graphml")
