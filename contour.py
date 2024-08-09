import matplotlib.pyplot as plt
import numpy as np
import os
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon, LineString
from shapely.strtree import STRtree
from geopy.distance import geodesic
from shapely.affinity import scale
from shapely.ops import transform

# This is a work in progress. Contour placement
def meters_to_degrees(meter_value, latitude):
    """Converts meters to degrees at a given latitude."""
    point1 = (latitude, 0)
    point2 = (latitude, 1)  # 1 degree of longitude
    distance = geodesic(point1, point2).meters
    degree_value = meter_value / distance
    return degree_value

def densify(polygon, num_points=100):
    """Adds more points to the polygon's boundary to densify it."""
    if isinstance(polygon, MultiPolygon):
        return MultiPolygon([densify(poly, num_points) for poly in polygon.geoms])
    
    line = LineString(polygon.exterior.coords)
    distances = np.linspace(0, line.length, num_points)
    points = [line.interpolate(distance) for distance in distances]
    
    # Filter out any empty or invalid points
    filtered_points = []
    for point in points:
        if not point.is_empty:
            filtered_points.append((point.x, point.y))
    
    if len(filtered_points) > 2:  # Polygon must have at least 3 points
        return Polygon(filtered_points)
    else:
        return Polygon()  # Return an empty polygon if not enough points


def plot_and_count_polar_circles_filled(main_polygon, circle_diameter_m, offset_distance_m, obstacle_polygon):
    fig, ax = plt.subplots(figsize=(10, 10))
    gpd.GeoSeries([main_polygon]).plot(ax=ax, edgecolor='black', facecolor='none')
    gpd.GeoSeries([obstacle_polygon]).plot(ax=ax, edgecolor='red', facecolor='none')
    
    minx, miny, maxx, maxy = main_polygon.bounds
    center_lat = (miny + maxy) / 2

    # Convert circle diameter and offset distance from meters to degrees
    circle_diameter = meters_to_degrees(circle_diameter_m, center_lat)
    offset_distance = meters_to_degrees(offset_distance_m, center_lat)
    radius = circle_diameter / 2
    circle_count = 0

    # List to keep track of placed circles
    placed_circles = []

    def place_circles_within(polygon):
        nonlocal circle_count
        grid_step = radius/5  # Grid step size based on circle diameter
        x_min, y_min, x_max, y_max = polygon.bounds
        
        for x in np.arange(x_min + radius, x_max, grid_step):
            for y in np.arange(y_min + radius, y_max, grid_step):
                point = Point(x, y)
                circle = point.buffer(radius)
                
                if polygon.contains(circle) and not obstacle_polygon.intersects(circle):
                    too_close = False
                    for placed_circle in placed_circles:
                        if placed_circle.distance(circle) < offset_distance:
                            too_close = True
                            break
                    if not too_close:
                        gpd.GeoSeries([circle]).plot(ax=ax, edgecolor='blue', facecolor='none')
                        circle_count += 1
                        placed_circles.append(circle)
                   # else:
                        #print(f"Circle at ({x}, {y}) is too close to another circle.")
                #else:
                  #  print(f"Circle at ({x}, {y}) is not valid: {circle.bounds}")

    current_polygon = main_polygon

    while not current_polygon.is_empty:
        # Place circles within the current polygon
        if isinstance(current_polygon, MultiPolygon):
            for poly in current_polygon.geoms:
                place_circles_within(poly)
        else:
            place_circles_within(current_polygon)

        # Move inward by buffering
        buffer_distance = -radius/5
        current_polygon = current_polygon.buffer(buffer_distance)
        
        # Ensure the buffered polygon is densified for the next iteration
        if isinstance(current_polygon, MultiPolygon):
            current_polygon = MultiPolygon([densify(poly) for poly in current_polygon.geoms])
        else:
            current_polygon = densify(current_polygon)

    ax.set_xlim(minx - 0.01, maxx + 0.01)
    ax.set_ylim(miny - 0.01, maxy + 0.01)
    
    plt.show()
    # Print the number of circles placed
    print(f"The number of circles with diameter {circle_diameter_m} meters and offset {offset_distance_m} meters that can fit inside the geologic site, avoiding the obstacle polygon, is {circle_count}.")
    
    return placed_circles

def calculate_adjacent_distances(circles, latitude):
    distances = []
    for i in range(len(circles) - 1):
        current_center = circles[i].centroid
        next_center = circles[i + 1].centroid
        
        # Convert centroid coordinates from degrees to latitude/longitude tuples
        current_coord = (current_center.y, current_center.x)
        next_coord = (next_center.y, next_center.x)
        
        # Calculate the distance in meters
        distance_between_centers = geodesic(current_coord, next_coord).meters
        distances.append(distance_between_centers)
    
    for i, distance in enumerate(distances):
        print(f"Distance between circle {i+1} and circle {i+2} centers: {distance:.2f} meters")



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
placed_circles = plot_and_count_polar_circles_filled(site_geometry, circle_diameter_m, offset_distance_m, obstacle_polygon)

# Calculate and print distances between adjacent circles in meters
# Use the latitude of the circles' centroid for accurate distance calculation
center_lat = (site_geometry.bounds[1] + site_geometry.bounds[3]) / 2
calculate_adjacent_distances(placed_circles, center_lat)
