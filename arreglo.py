import matplotlib.pyplot as plt
import numpy as np
import os
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPoint
from shapely.strtree import STRtree
from geopy.distance import geodesic
from shapely.affinity import scale

def number_of_circles_in_square_with_offset(side_length, circle_diameter, offset_distance):
    center_distance = circle_diameter + offset_distance
    circles_per_side = int(side_length // center_distance)
    total_circles = circles_per_side ** 2
    return total_circles, circles_per_side

def circle_inside_triangle(center_x, center_y, radius, triangle_points):
    x1, y1 = triangle_points[0]
    x2, y2 = triangle_points[1]
    x3, y3 = triangle_points[2]
    
    def sign(px, py, x1, y1, x2, y2):
        return (px - x2) * (y1 - y2) - (x1 - x2) * (py - y2)
    
    d1 = sign(center_x, center_y, x1, y1, x2, y2)
    d2 = sign(center_x, center_y, x2, y2, x3, y3)
    d3 = sign(center_x, center_y, x3, y3, x1, y1)
    
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    
    return not (has_neg and has_pos)

def plot_circles_in_square_with_center_obstacle(side_length, circle_diameter, offset_distance, obstacle_side_length):
    total_circles, circles_per_side = number_of_circles_in_square_with_offset(side_length, circle_diameter, offset_distance)
    center_distance = circle_diameter + offset_distance
    radius = circle_diameter / 2
    
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    
    square = plt.Rectangle((0, 0), side_length, side_length, fill=None, edgecolor='r')
    ax.add_patch(square)
    
    obstacle_start = (side_length - obstacle_side_length) / 2
    obstacle = plt.Rectangle((obstacle_start, obstacle_start), obstacle_side_length, obstacle_side_length, fill=None, edgecolor='g')
    ax.add_patch(obstacle)
    
    triangle_points = [(2, 8), (1, 9), (3, 9)]
    triangle_polygon = plt.Polygon(triangle_points, closed=True, fill=None, edgecolor='g')
    ax.add_patch(triangle_polygon)
    
    circle_count = 0
    
    for i in range(circles_per_side):
        for j in range(circles_per_side):
            center_x = i * center_distance + radius
            center_y = j * center_distance + radius
            
            if (obstacle_start - radius) < center_x < (obstacle_start + obstacle_side_length + radius) and (obstacle_start - radius) < center_y < (obstacle_start + obstacle_side_length + radius):
                continue
            
            if circle_inside_triangle(center_x, center_y, radius, triangle_points):
                continue
            
            circle = plt.Circle((center_x, center_y), radius, fill=None, edgecolor='b')
            ax.add_patch(circle)
            circle_count += 1
    
    plt.xlim(-1, side_length + 1)
    plt.ylim(-1, side_length + 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    
    return circle_count

def generate_square_inside_site(site_geometry, side_length):
    bounds = site_geometry.bounds
    minx, miny, maxx, maxy = bounds
    
    for i in np.arange(minx, maxx - side_length, side_length / 10):
        for j in np.arange(miny, maxy - side_length, side_length / 10):
            square = Polygon([
                (i, j),
                (i, j + side_length),
                (i + side_length, j + side_length),
                (i + side_length, j),
                (i, j)
            ])
            if site_geometry.contains(square):
                return square
    return None

def number_of_circles_in_polygon_with_offset(polygon, circle_diameter, offset_distance):
    center_distance = circle_diameter + offset_distance
    radius = circle_diameter / 2
    
    minx, miny, maxx, maxy = polygon.bounds
    
    circle_count = 0
    circle_positions = []
    
    for x in np.arange(minx + radius, maxx, center_distance):
        for y in np.arange(miny + radius, maxy, center_distance):
            circle = Point(x, y).buffer(radius)
            if polygon.contains(circle):
                circle_positions.append((x, y))
                circle_count += 1
    
    return circle_count, circle_positions

def plot_circles_in_polygon_with_obstacles(polygon, diameter, spacing, obstacles):
    fig, ax = plt.subplots(figsize=(10, 10))
    gpd.GeoSeries([polygon]).plot(ax=ax, edgecolor='black', facecolor='none')
    
    for obstacle in obstacles:
        gpd.GeoSeries([obstacle]).plot(ax=ax, edgecolor='red', facecolor='none')
    
    minx, miny, maxx, maxy = polygon.bounds
    radius = diameter / 2
    circle_count = 0
    
    x = minx + radius
    while x < maxx - radius:
        y = miny + radius
        while y < maxy - radius:
            point = Point(x, y)
            circle = point.buffer(radius)
            if polygon.contains(circle) and all(not obstacle.intersects(circle) for obstacle in obstacles):
                gpd.GeoSeries([circle]).plot(ax=ax, edgecolor='blue', facecolor='none')
                circle_count += 1
            y += diameter + spacing
        x += diameter + spacing
    
    plt.show()
    return circle_count

def generate_irregular_obstacle_inside_site(site_polygon, buffer_distance=5):
    centroid = site_polygon.centroid
    obstacle = centroid.buffer(buffer_distance)
    
    # Check if the buffered shape fits entirely within the site polygon
    while not site_polygon.contains(obstacle):
        buffer_distance *= 0.9  # Reduce the buffer distance iteratively
        obstacle = centroid.buffer(buffer_distance)
    
    return obstacle, buffer_distance

# Example usage with square, triangle, and additional obstacle
L = 10
d = 1
s = 0.5
obstacle_side_length = 2

#circle_count = plot_circles_in_square_with_center_obstacle(L, d, s, obstacle_side_length)
#print(f"The number of circles with diameter {d} and offset {s} that can fit inside a square with side length {L}, avoiding multiple obstacles (square and triangle), is {circle_count}.")

# Example usage with geologic storage site and obstacles
INPUT_DIR1 = "/Users/mmoulton/Documents/Projects/Elmundo/inputs/tl_2023_us_state"
INPUT_DIR2 = "/Users/mmoulton/Documents/Projects/Elmundo/inputs/geologic_storage"
shapefile_path1 = os.path.join(INPUT_DIR1, "tl_2023_us_state.shp")
shapefile_path2 = os.path.join(INPUT_DIR2, "salt_Shapefile_4326.shp")

state_data = gpd.read_file(shapefile_path1)
geologic_storage_data = gpd.read_file(shapefile_path2)

non_continental = ['HI', 'VI', 'MP', 'GU', 'AK', 'AS', 'PR']
state_data = state_data[~state_data['STUSPS'].isin(non_continental)]

geologic_storage_data = geologic_storage_data.to_crs(epsg=4326)

single_geologic_site = geologic_storage_data.iloc[0:1]
site_geometry = single_geologic_site.geometry.values[0]

fig, ax = plt.subplots(figsize=(10, 10))

state_data.plot(ax=ax, edgecolor='black', color='white', linewidth=0.5)
single_geologic_site.plot(ax=ax, color='blue', alpha=0.5)

square_side_length = 2

square_inside_site = generate_square_inside_site(site_geometry, square_side_length)

if square_inside_site:
    gpd.GeoSeries([square_inside_site]).plot(ax=ax, edgecolor='red', facecolor='none', linewidth=1.5)
else:
    print("No suitable location for the square inside the site.")

minx, miny, maxx, maxy = single_geologic_site.total_bounds
ax.set_xlim(minx - 0.5, maxx + 0.5)
ax.set_ylim(miny - 0.5, maxy + 0.5)

ax.set_title('US States and Selected Geologic Storage Site with Square')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

plt.show()

d = 1
s = 0.1

square_obstacle = generate_square_inside_site(site_geometry, square_side_length)

if square_obstacle:
    circle_count = plot_circles_in_polygon_with_obstacles(site_geometry, d, s, [square_obstacle])
    print(f"The number of circles with diameter {d} and offset {s} that can fit inside the geologic site, avoiding the square obstacle, is {circle_count}.")
else:
    print("No suitable location for the square inside the site.")

d = 1
s = .1

irregular_obstacle, final_buffer_distance = generate_irregular_obstacle_inside_site(site_geometry)

circle_count = plot_circles_in_polygon_with_obstacles(site_geometry, d, s, [irregular_obstacle])
print(f"The number of circles with diameter {d} and offset {s} that can fit inside the geologic site, avoiding the irregular obstacle, is {circle_count}.")

# Plot the circles within the geologic site, avoiding both obstacles
square_obstacle = generate_square_inside_site(site_geometry, square_side_length)
if square_obstacle:
    circle_count = plot_circles_in_polygon_with_obstacles(site_geometry, d, s, [square_obstacle, irregular_obstacle])
    print(f"The number of circles with diameter {d} and offset {s} that can fit inside the geologic site, avoiding the square and irregular obstacles, is {circle_count}.")
else:
    print("No suitable location for the square inside the site.")

# Print the final buffer distance
print(f"The final buffer distance for the irregular obstacle is approximately {final_buffer_distance:.2f} units.")
# THis function actually works pretty well. The previous funtions were just for figuring stuff out

def meters_to_degrees(meter_value, latitude):
    """Converts meters to degrees at a given latitude."""
    point1 = (latitude, 0)
    point2 = (latitude, 1)  # 1 degree of longitude
    distance = geodesic(point1, point2).meters
    degree_value = meter_value / distance
    return degree_value

def plot_circles_in_polygon_with_obstacle(main_polygon, circle_diameter_m, offset_distance_m, obstacle_polygon):
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
    
    # Create spatial index for the obstacle
    obstacle_tree = STRtree([obstacle_polygon])
    
    x = minx + radius
    while x < maxx - radius:
        y = miny + radius
        while y < maxy - radius:
            point = Point(x, y)
            circle = point.buffer(radius)
            if main_polygon.contains(circle) and not obstacle_polygon.intersects(circle):
                gpd.GeoSeries([circle]).plot(ax=ax, edgecolor='blue', facecolor='none')
                circle_count += 1
            y += circle_diameter + offset_distance
        x += circle_diameter + offset_distance
    
    plt.show()
    return circle_count


# Load geologic site data
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

circle_diameter = 10000
offset_distance = 5000

# Plot and count circles within the geologic site while avoiding the obstacle polygon
circle_count = plot_circles_in_polygon_with_obstacle(site_geometry, circle_diameter, offset_distance, obstacle_polygon)
print(f"The number of circles with diameter {circle_diameter} and offset {offset_distance} that can fit inside the geologic site, avoiding the obstacle polygon, is {circle_count}.")


# This next part also uses square arrangement with additional checks arrangement


def plot_and_count_square_grid_circles_in_polygon_with_obstacle(main_polygon, circle_diameter_m, offset_distance_m, obstacle_polygon):
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

    # Create spatial index for the obstacle
    obstacle_tree = STRtree([obstacle_polygon])

    # Use a finer grid to improve accuracy
    step_size = circle_diameter + offset_distance
    
    # List to keep track of circle centers to ensure they are not too close to each other
    placed_circles = []

    x = minx + radius
    while x < maxx - radius:
        y = miny + radius
        while y < maxy - radius:
            point = Point(x, y)
            circle = point.buffer(radius)

            # Check if circle is within the main polygon and does not intersect with the obstacle
            if main_polygon.contains(circle) and not obstacle_polygon.intersects(circle):
                # Check if circle is at least offset_distance away from all previously placed circles
                too_close = False
                for placed_circle in placed_circles:
                    if placed_circle.distance(circle) < offset_distance:
                        too_close = True
                        break
                
                if not too_close:
                    gpd.GeoSeries([circle]).plot(ax=ax, edgecolor='blue', facecolor='none')
                    circle_count += 1
                    placed_circles.append(circle)
            
            y += step_size
        x += step_size
    
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

circle_diameter_m = 10000
offset_distance_m = 5000

# Plot and count circles within the geologic site while avoiding the obstacle polygon
#circle_count = plot_and_count_square_grid_circles_in_polygon_with_obstacle(site_geometry, circle_diameter_m, offset_distance_m, obstacle_polygon)
#print(f"The number of circles with diameter {circle_diameter_m} meters and offset {offset_distance_m} meters that can fit inside the geologic site, avoiding the obstacle polygon, is {circle_count}.")



def plot_and_count_hexagonal_circles_in_polygon_with_obstacle(main_polygon, circle_diameter_m, offset_distance_m, obstacle_polygon):
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

    # Create spatial index for the obstacle
    obstacle_tree = STRtree([obstacle_polygon])

    # Calculate the vertical and horizontal steps for a hexagonal grid
    vertical_step = (circle_diameter + offset_distance) * np.sqrt(3) / 2
    horizontal_step = circle_diameter + offset_distance
    
    placed_circles = []

    x = minx + radius
    row = 0
    while x < maxx - radius:
        y = miny + radius if row % 2 == 0 else miny + radius + vertical_step / 2
        while y < maxy - radius:
            point = Point(x, y)
            circle = point.buffer(radius)

            # Check if circle is within the main polygon and does not intersect with the obstacle
            if main_polygon.contains(circle) and not obstacle_polygon.intersects(circle):
                # Check if circle is at least offset_distance away from all previously placed circles
                too_close = False
                for placed_circle in placed_circles:
                    if placed_circle.distance(circle) < offset_distance:
                        too_close = True
                        break
                
                if not too_close:
                    gpd.GeoSeries([circle]).plot(ax=ax, edgecolor='blue', facecolor='none')
                    circle_count += 1
                    placed_circles.append(circle)
            
            y += vertical_step
        x += horizontal_step
        row += 1
    
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

circle_diameter_m = 10000
offset_distance_m = 5000

# Plot and count circles within the geologic site while avoiding the obstacle polygon
#circle_count = plot_and_count_hexagonal_circles_in_polygon_with_obstacle(site_geometry, circle_diameter_m, offset_distance_m, obstacle_polygon)
#print(f"The number of circles with diameter {circle_diameter_m} meters and offset {offset_distance_m} meters that can fit inside the geologic site, avoiding the obstacle polygon, is {circle_count}.")

#This uses a polar arrangement

def plot_and_count_polar_circles_in_polygon_with_obstacle(main_polygon, circle_diameter_m, offset_distance_m, obstacle_polygon):
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

    # Center of the polygon
    center_x, center_y = main_polygon.centroid.x, main_polygon.centroid.y
    
    # Create spatial index for the obstacle
    obstacle_tree = STRtree([obstacle_polygon])
    
    # Place circles in concentric rings
    current_radius = radius
    while current_radius < max(maxx - minx, maxy - miny) / 2:
        circumference = 2 * np.pi * current_radius
        num_circles = max(1, int(circumference / (circle_diameter + offset_distance)))  # Ensure num_circles is at least 1
        angle_step = 2 * np.pi / num_circles

        for i in range(num_circles):
            angle = i * angle_step
            x = center_x + current_radius * np.cos(angle)
            y = center_y + current_radius * np.sin(angle)
            point = Point(x, y)
            circle = point.buffer(radius)
            
            if main_polygon.contains(circle) and not obstacle_polygon.intersects(circle):
                gpd.GeoSeries([circle]).plot(ax=ax, edgecolor='blue', facecolor='none')
                circle_count += 1
        
        current_radius += circle_diameter + offset_distance
    
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

circle_diameter_m = 10000
offset_distance_m = 5000

# Plot and count circles within the geologic site while avoiding the obstacle polygon
circle_count = plot_and_count_polar_circles_in_polygon_with_obstacle(site_geometry, circle_diameter_m, offset_distance_m, obstacle_polygon)
print(f"The number of circles with diameter {circle_diameter_m} meters and offset {offset_distance_m} meters that can fit inside the geologic site, avoiding the obstacle polygon, is {circle_count}.")


# This function uses edge filling first then polar arrangement
def plot_and_count_polar_circles_with_edge_filling(main_polygon, circle_diameter_m, offset_distance_m, obstacle_polygon):
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

    # Create spatial index for the obstacle
    obstacle_tree = STRtree([obstacle_polygon])

    # Define edge region to be filled first
    edge_buffer = radius + offset_distance
    edge_polygon = main_polygon.buffer(edge_buffer)

    # Function to place circles in polar arrangement
    def place_polar_circles(central_point, max_radius):
        nonlocal circle_count
        angle_step = np.pi / 6  # Step angle (30 degrees)
        current_radius = radius
        while current_radius < max_radius:
            angle = 0
            while angle < 2 * np.pi:
                x = central_point.x + current_radius * np.cos(angle)
                y = central_point.y + current_radius * np.sin(angle)
                point = Point(x, y)
                circle = point.buffer(radius)
                if main_polygon.contains(circle) and not obstacle_polygon.intersects(circle):
                    # Check if circle is at least offset_distance away from all previously placed circles
                    too_close = False
                    for placed_circle in placed_circles:
                        if placed_circle.distance(circle) < offset_distance:
                            too_close = True
                            break
                    if not too_close:
                        gpd.GeoSeries([circle]).plot(ax=ax, edgecolor='blue', facecolor='none')
                        circle_count += 1
                        placed_circles.append(circle)
                angle += angle_step
            current_radius += circle_diameter + offset_distance

    # List to keep track of circle centers to ensure they are not too close to each other
    placed_circles = []

    # Fill edges first
    edge_step_size = circle_diameter + offset_distance
    x = minx + radius
    while x < maxx - radius:
        y = miny + radius
        while y < maxy - radius:
            point = Point(x, y)
            if edge_polygon.contains(point):
                place_polar_circles(point, radius + edge_buffer)
            y += edge_step_size
        x += edge_step_size

    # Fill the interior with polar arrangement
    place_polar_circles(Point((minx + maxx) / 2, (miny + maxy) / 2), min((maxx - minx) / 2, (maxy - miny) / 2))
    
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

circle_diameter_m = 10000
offset_distance_m = 5000

# Plot and count circles within the geologic site while avoiding the obstacle polygon
#circle_count = plot_and_count_polar_circles_with_edge_filling(site_geometry, circle_diameter_m, offset_distance_m, obstacle_polygon)
#print(f"The number of circles with diameter {circle_diameter_m} meters and offset {offset_distance_m} meters that can fit inside the geologic site, avoiding the obstacle polygon, is {circle_count}.")

# This next function makes arranges the circles in a radial format then fills in the edges
def plot_and_count_polar_circles_with_edge_filling2(main_polygon, circle_diameter_m, offset_distance_m, obstacle_polygon):
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

    # Center of the polygon
    center_x, center_y = main_polygon.centroid.x, main_polygon.centroid.y
    
    # Create spatial index for the obstacle
    obstacle_tree = STRtree([obstacle_polygon])
    
    # List to keep track of circle centers to ensure they are not too close to each other
    placed_circles = []

    # Function to place circles in concentric rings
    def place_circles_in_concentric_rings():
        nonlocal circle_count
        current_radius = radius
        while current_radius < max(maxx - minx, maxy - miny) / 2:
            circumference = 2 * np.pi * current_radius
            num_circles = max(1, int(circumference / (circle_diameter + offset_distance)))  # Ensure num_circles is at least 1
            angle_step = 2 * np.pi / num_circles

            for i in range(num_circles):
                angle = i * angle_step
                x = center_x + current_radius * np.cos(angle)
                y = center_y + current_radius * np.sin(angle)
                point = Point(x, y)
                circle = point.buffer(radius)
                
                if main_polygon.contains(circle) and not obstacle_polygon.intersects(circle):
                    # Check if circle is at least offset_distance away from all previously placed circles
                    too_close = False
                    for placed_circle in placed_circles:
                        if placed_circle.distance(circle) < offset_distance:
                            too_close = True
                            break
                    if not too_close:
                        gpd.GeoSeries([circle]).plot(ax=ax, edgecolor='blue', facecolor='none')
                        circle_count += 1
                        placed_circles.append(circle)
            
            current_radius += circle_diameter + offset_distance

    # Fill interior with concentric rings
    place_circles_in_concentric_rings()

    # Fill edges
    edge_buffer = radius + offset_distance
    edge_polygon = main_polygon.buffer(edge_buffer)
    
    edge_step_size = circle_diameter + offset_distance
    x = minx + radius
    while x < maxx - radius:
        y = miny + radius
        while y < maxy - radius:
            point = Point(x, y)
            if edge_polygon.contains(point):
                # Place circles on the edge
                for angle in np.linspace(0, 2 * np.pi, num=12, endpoint=False):  # Place circles around the edge
                    edge_x = point.x + edge_buffer * np.cos(angle)
                    edge_y = point.y + edge_buffer * np.sin(angle)
                    edge_point = Point(edge_x, edge_y)
                    edge_circle = edge_point.buffer(radius)
                    
                    if main_polygon.contains(edge_circle) and not obstacle_polygon.intersects(edge_circle):
                        # Check if circle is at least offset_distance away from all previously placed circles
                        too_close = False
                        for placed_circle in placed_circles:
                            if placed_circle.distance(edge_circle) < offset_distance:
                                too_close = True
                                break
                        if not too_close:
                            gpd.GeoSeries([edge_circle]).plot(ax=ax, edgecolor='blue', facecolor='none')
                            circle_count += 1
                            placed_circles.append(edge_circle)
            
            y += edge_step_size
        x += edge_step_size
    
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

circle_diameter_m = 10000
offset_distance_m = 5000

# Plot and count circles within the geologic site while avoiding the obstacle polygon
#circle_count = plot_and_count_polar_circles_with_edge_filling2(site_geometry, circle_diameter_m, offset_distance_m, obstacle_polygon)
#print(f"The number of circles with diameter {circle_diameter_m} meters and offset {offset_distance_m} meters that can fit inside the geologic site, avoiding the obstacle polygon, is {circle_count}.")


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


def plot_and_count_triangular_circles_in_polygon_with_obstacle(main_polygon, circle_diameter_m, offset_distance_m, obstacle_polygon):
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

    # Calculate steps for the triangular grid
    vertical_step = circle_diameter * np.sqrt(3) / 2
    horizontal_step = circle_diameter

    placed_circles = []
    
    x = minx + radius
    row = 0
    while x < maxx - radius:
        y = miny + radius + (vertical_step / 2 * (row % 2))
        while y < maxy - radius:
            point = Point(x, y)
            circle = point.buffer(radius)

            # Check if circle is within the main polygon and does not intersect with the obstacle
            if main_polygon.contains(circle) and not obstacle_polygon.intersects(circle):
                # Check if circle is at least offset_distance away from all previously placed circles
                too_close = False
                for placed_circle in placed_circles:
                    if placed_circle.distance(circle) < offset_distance:
                        too_close = True
                        break
                
                if not too_close:
                    gpd.GeoSeries([circle]).plot(ax=ax, edgecolor='blue', facecolor='none')
                    circle_count += 1
                    placed_circles.append(circle)
            
            y += vertical_step
        x += horizontal_step
        row += 1

    plt.show()
    return circle_count

# Example usage with hypothetical data
# Define the main_polygon and obstacle_polygon as before, and use them in the function

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
#circle_count = plot_and_count_triangular_circles_in_polygon_with_obstacle(site_geometry, circle_diameter_m, offset_distance_m, obstacle_polygon)
#print(f"The number of circles with diameter {circle_diameter_m} meters and offset {offset_distance_m} meters that can fit inside the geologic site, avoiding the obstacle polygon, is {circle_count}.")