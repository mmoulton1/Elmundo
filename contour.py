import matplotlib.pyplot as plt
import numpy as np
import os
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPoint
from shapely.strtree import STRtree
from geopy.distance import geodesic
from shapely.affinity import scale

#This is a work in progress. Contour placement
def meters_to_degrees(meter_value, latitude):
    """Converts meters to degrees at a given latitude."""
    point1 = (latitude, 0)
    point2 = (latitude, 1)  # 1 degree of longitude
    distance = geodesic(point1, point2).meters
    degree_value = meter_value / distance
    return degree_value

def plot_and_count_polar_circles_perimeter_only(main_polygon, circle_diameter_m, offset_distance_m, obstacle_polygon):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot the main polygon and obstacle polygon
    gpd.GeoSeries([main_polygon]).plot(ax=ax, edgecolor='black', facecolor='none', label='Main Polygon')
    gpd.GeoSeries([obstacle_polygon]).plot(ax=ax, edgecolor='red', facecolor='none', label='Obstacle Polygon')
    
    minx, miny, maxx, maxy = main_polygon.bounds
    center_lat = (miny + maxy) / 2

    # Convert circle diameter and offset distance from meters to degrees
    circle_diameter = meters_to_degrees(circle_diameter_m, center_lat)
    offset_distance = meters_to_degrees(offset_distance_m, center_lat)
    radius = circle_diameter / 2
    circle_count = 0

    # Define buffer distance to shrink the polygon
    buffer_distance = radius + offset_distance
    
    # Create a smaller polygon by buffering the original polygon negatively
    smaller_polygon = main_polygon.buffer(-buffer_distance)
    
    # Plot the smaller buffered polygon
    gpd.GeoSeries([smaller_polygon]).plot(ax=ax, edgecolor='green', facecolor='none', linestyle='--', label='Smaller Polygon')
    
    # List to keep track of circle centers to ensure they are not too close to each other
    placed_circles = []

    def place_circles_on_edge():
        nonlocal circle_count
        # Generate points along the boundary of the buffered polygon
        boundary_points = list(smaller_polygon.exterior.coords)
        
        for point in boundary_points:
            # Create a point at the boundary of the buffered polygon
            boundary_point = Point(point[0], point[1])
            
            # Create a circle centered on this point
            circle = boundary_point.buffer(radius)
            
            # Check if circle is entirely within the original polygon and does not intersect with the obstacle
            if main_polygon.contains(circle) and not obstacle_polygon.intersects(circle):
                # Ensure the circle is at least offset_distance away from all previously placed circles
                too_close = False
                for placed_circle in placed_circles:
                    if placed_circle.distance(circle) < offset_distance:
                        too_close = True
                        break
                if not too_close:
                    gpd.GeoSeries([circle]).plot(ax=ax, edgecolor='blue', facecolor='none')
                    circle_count += 1
                    placed_circles.append(circle)
                else:
                    print(f"Circle at ({point[0]}, {point[1]}) is too close to another circle.")
            else:
                print(f"Circle at ({point[0]}, {point[1]}) is not valid: {circle.bounds}")
    
    # Place circles along the edge of the smaller buffered polygon
    place_circles_on_edge()

    # Adjust plot limits to ensure visibility
    ax.set_xlim(minx - 0.01, maxx + 0.01)
    ax.set_ylim(miny - 0.01, maxy + 0.01)
    

    
    plt.show()
    return circle_count


# Load geologic site data and set parameters
INPUT_DIR1 = "/Users/mmoulton/Documents/Projects/Elmundo/inputs/tl_2023_us_state"
INPUT_DIR2 = "/Users/mmoulton/Documents/Projects/Elmundo/inputs/geologic_storage"
shapefile_path1 = os.path.join(INPUT_DIR1, "tl_2023_us_state.shp")
shapefile_path2 = os.path.join(INPUT_DIR2, "salt_Shapefile_4326.shp")

state_data = gpd.read_file(shapefile_path1)
geologic_storage_data = gpd.read_file(shapefile_path2)

# Exclude non-continental states
non_continental = ['HI', 'VI', 'MP', 'GU', 'AK', 'AS', 'PR']
state_data = state_data[~state_data['STUSPS'].isin(non_continental)]

# Transform to a common CRS if needed
geologic_storage_data = geologic_storage_data.to_crs(epsg=4326)

# Select the first geologic site for demonstration
single_geologic_site = geologic_storage_data.iloc[0:1]
site_geometry = single_geologic_site.geometry.values[0]

# Get the bounds of the site geometry
minx, miny, maxx, maxy = site_geometry.bounds

# Create an obstacle polygon closer to the center of the site geometry
obstacle_polygon = Polygon([
    (minx + (maxx - minx) * 0.4, miny + (maxy - miny) * 0.4),
    (minx + (maxx - minx) * 0.5, miny + (maxy - miny) * 0.4),
    (minx + (maxx - minx) * 0.5, miny + (maxy - miny) * 0.5),
    (minx + (maxx - minx) * 0.4, miny + (maxy - miny) * 0.5)
])

# Print the coordinates of the obstacle polygon
obstacle_coords = list(obstacle_polygon.exterior.coords)
print("Coordinates of the obstacle polygon:")
for coord in obstacle_coords:
    print(coord)

# Check if the obstacle polygon is within the site geometry
is_within = site_geometry.contains(obstacle_polygon)
print(f"Is the obstacle polygon within the site geometry? {is_within}")

# Define circle parameters
circle_diameter_m = 10000
offset_distance_m = 5000

# Plot and count circles
circle_count = plot_and_count_polar_circles_perimeter_only(site_geometry, circle_diameter_m, offset_distance_m, obstacle_polygon)
print(f"The number of circles with diameter {circle_diameter_m} meters and offset {offset_distance_m} meters that can fit inside the geologic site, avoiding the obstacle polygon, is {circle_count}.")


