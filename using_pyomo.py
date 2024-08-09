import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from geopy.distance import geodesic
import os
import pyomo.environ as pyo
from shapely.prepared import prep

# This is a work in progress. Contour placement with pyomo
def meters_to_degrees(meter_value, latitude):
    """Converts meters to degrees at a given latitude."""
    point1 = (latitude, 0)
    point2 = (latitude, 1)  # 1 degree of longitude
    distance = geodesic(point1, point2).meters
    degree_value = meter_value / distance
    return degree_value

def plot_polygon_and_circles(main_polygon, obstacle_polygon, circle_centers, circle_diameter_m):
    fig, ax = plt.subplots(figsize=(10, 10))
    gpd.GeoSeries([main_polygon]).plot(ax=ax, edgecolor='black', facecolor='none')
    gpd.GeoSeries([obstacle_polygon]).plot(ax=ax, edgecolor='red', facecolor='none')

    for (x, y) in circle_centers:
        circle = Point(x, y).buffer(meters_to_degrees(circle_diameter_m / 2, y))
        gpd.GeoSeries([circle]).plot(ax=ax, edgecolor='blue', facecolor='none')

    ax.set_xlim(main_polygon.bounds[0] - 0.01, main_polygon.bounds[2] + 0.01)
    ax.set_ylim(main_polygon.bounds[1] - 0.01, main_polygon.bounds[3] + 0.01)
    plt.show()

def optimize_circle_placement(main_polygon, circle_diameter_m, offset_distance_m, obstacle_polygon):
    # Convert circle diameter and offset distance from meters to degrees
    radius = meters_to_degrees(circle_diameter_m / 2, main_polygon.centroid.y)
    offset_distance = meters_to_degrees(offset_distance_m, main_polygon.centroid.y)

    N = 100  # Estimate maximum number of circles
    model = pyo.ConcreteModel()

    # Variables: Coordinates of circle centers and a binary variable to indicate if a circle is placed
    model.x = pyo.Var(range(N), domain=pyo.Reals, bounds=(main_polygon.bounds[0], main_polygon.bounds[2]))
    model.y = pyo.Var(range(N), domain=pyo.Reals, bounds=(main_polygon.bounds[1], main_polygon.bounds[3]))
    model.z = pyo.Var(range(N), domain=pyo.Binary)

    # Objective: Maximize the number of placed circles
    model.obj = pyo.Objective(expr=sum(model.z[i] for i in range(N)), sense=pyo.maximize)

    # Constraints: Circles must be within the polygon bounds
    def inside_bounds_x(model, i):
        return model.x[i] >= main_polygon.bounds[0] and model.x[i] <= main_polygon.bounds[2]

    def inside_bounds_y(model, i):
        return model.y[i] >= main_polygon.bounds[1] and model.y[i] <= main_polygon.bounds[3]

    model.inside_bounds_x = pyo.Constraint(range(N), rule=inside_bounds_x)
    model.inside_bounds_y = pyo.Constraint(range(N), rule=inside_bounds_y)

    # Constraints: Circles must not overlap with the obstacle polygon
    def no_intersection_with_obstacle(model, i):
        prepared_obstacle = prep(obstacle_polygon)
        circle_center = Point(model.x[i], model.y[i])
        circle = circle_center.buffer(radius)
        return prepared_obstacle.disjoint(circle)
    model.no_intersection_with_obstacle = pyo.Constraint(range(N), rule=no_intersection_with_obstacle)

    # Constraint: Circles must not overlap with each other
    def no_overlap(model, i, j):
        if i < j:
            distance_between_centers = ((model.x[i] - model.x[j])**2 + (model.y[i] - model.y[j])**2)**0.5
            return distance_between_centers >= 2 * radius + offset_distance
        return pyo.Constraint.Skip
    model.no_overlap = pyo.Constraint(range(N), range(N), rule=no_overlap)

    # Solve the model
    solver = pyo.SolverFactory('glpk')
    solver.solve(model)

    # Extract the placed circle centers
    circle_centers = [(pyo.value(model.x[i]), pyo.value(model.y[i])) for i in range(N) if pyo.value(model.z[i]) > 0.5]

    return circle_centers

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

# Create an obstacle polygon closer to the center of the site geometry
minx, miny, maxx, maxy = site_geometry.bounds
obstacle_polygon = Polygon([
    (minx + (maxx - minx) * 0.4, miny + (maxy - miny) * 0.4),
    (minx + (maxx - minx) * 0.5, miny + (maxy - miny) * 0.4),
    (minx + (maxx - minx) * 0.5, miny + (maxy - miny) * 0.5),
    (minx + (maxx - minx) * 0.4, miny + (maxy - miny) * 0.5)
])

# Define circle parameters
circle_diameter_m = 100000
offset_distance_m = 50000

# Run the optimization to place circles
circle_centers = optimize_circle_placement(site_geometry, circle_diameter_m, offset_distance_m, obstacle_polygon)

# Plot the results
plot_polygon_and_circles(site_geometry, obstacle_polygon, circle_centers, circle_diameter_m)
