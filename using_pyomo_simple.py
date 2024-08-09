import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from geopy.distance import geodesic
import pyomo.environ as pyo

# Convert meters to degrees at a given latitude
def meters_to_degrees(meter_value, latitude):
    """Converts meters to degrees at a given latitude."""
    point1 = (latitude, 0)
    point2 = (latitude, 1)  # 1 degree of longitude
    distance = geodesic(point1, point2).meters
    degree_value = meter_value / distance
    return degree_value

def plot_polygon_and_circles(main_polygon, circle_centers, circle_diameter_m):
    fig, ax = plt.subplots(figsize=(10, 10))
    gpd.GeoSeries([main_polygon]).plot(ax=ax, edgecolor='black', facecolor='none')

    for (x, y) in circle_centers:
        circle = Point(x, y).buffer(meters_to_degrees(circle_diameter_m / 2, y))
        gpd.GeoSeries([circle]).plot(ax=ax, edgecolor='blue', facecolor='none')

    ax.set_xlim(main_polygon.bounds[0] - 0.01, main_polygon.bounds[2] + 0.01)
    ax.set_ylim(main_polygon.bounds[1] - 0.01, main_polygon.bounds[3] + 0.01)
    plt.show()

def optimize_circle_placement(main_polygon, circle_diameter_m, offset_distance_m):
    # Convert circle diameter and offset distance from meters to degrees
    radius = meters_to_degrees(circle_diameter_m / 2, main_polygon.centroid.y)
    offset_distance = meters_to_degrees(offset_distance_m, main_polygon.centroid.y)

    N = 100  # Estimate maximum number of circles
    model = pyo.ConcreteModel("CirclePlacement")

    # Define sets
    model.Circles = pyo.RangeSet(1, N)

    # Parameters
    model.BoundsX = pyo.Param(initialize=(main_polygon.bounds[0], main_polygon.bounds[2]))
    model.BoundsY = pyo.Param(initialize=(main_polygon.bounds[1], main_polygon.bounds[3]))
    model.Radius = pyo.Param(initialize=radius)
    model.OffsetDistance = pyo.Param(initialize=offset_distance)

    # Variables
    model.x = pyo.Var(model.Circles, domain=pyo.Reals, bounds=(model.BoundsX[0], model.BoundsX[1]))
    model.y = pyo.Var(model.Circles, domain=pyo.Reals, bounds=(model.BoundsY[0], model.BoundsY[1]))
    model.z = pyo.Var(model.Circles, domain=pyo.Binary)

    # Objective: Maximize the number of placed circles
    def objective_rule(model):
        return sum(model.z[i] for i in model.Circles)
    model.Objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    # Constraints
    # Ensure x-coordinates are within the polygon bounds
    def inside_bounds_x_rule(model, i):
        return model.x[i] >= model.BoundsX[0]
    model.inside_bounds_x_low = pyo.Constraint(model.Circles, rule=inside_bounds_x_rule)

    def inside_bounds_x_high_rule(model, i):
        return model.x[i] <= model.BoundsX[1]
    model.inside_bounds_x_high = pyo.Constraint(model.Circles, rule=inside_bounds_x_high_rule)

    # Ensure y-coordinates are within the polygon bounds
    def inside_bounds_y_rule(model, i):
        return model.y[i] >= model.BoundsY[0]
    model.inside_bounds_y_low = pyo.Constraint(model.Circles, rule=inside_bounds_y_rule)

    def inside_bounds_y_high_rule(model, i):
        return model.y[i] <= model.BoundsY[1]
    model.inside_bounds_y_high = pyo.Constraint(model.Circles, rule=inside_bounds_y_high_rule)

    # No overlap constraints
    def no_overlap_rule(model, i, j):
        if i < j:
            distance_between_centers = ((model.x[i] - model.x[j])**2 + (model.y[i] - model.y[j])**2)**0.5
            return distance_between_centers >= 2 * model.Radius + model.OffsetDistance
        return pyo.Constraint.Skip
    model.no_overlap = pyo.Constraint(model.Circles, model.Circles, rule=no_overlap_rule)

    # Solve the model
    solver = pyo.SolverFactory('glpk')
    results = solver.solve(model, tee=True)

    # Extract the placed circle centers
    circle_centers = [(pyo.value(model.x[i]), pyo.value(model.y[i])) for i in model.Circles if pyo.value(model.z[i]) > 0.5]

    return circle_centers

# Define a simple square polygon with lat/lon coordinates
latitude = 37.7749  # Example latitude
longitude = -122.4194  # Example longitude

# Define a simple square polygon centered around (latitude, longitude)
main_polygon = Polygon([
    (longitude - 0.01, latitude - 0.01),
    (longitude - 0.01, latitude + 0.01),
    (longitude + 0.01, latitude + 0.01),
    (longitude + 0.01, latitude - 0.01)
])

# Define circle parameters
circle_diameter_m = 100  # Diameter of circles in meters
offset_distance_m = 20  # Offset distance between circles in meters

# Run the optimization to place circles
circle_centers = optimize_circle_placement(main_polygon, circle_diameter_m, offset_distance_m)

# Plot the results
plot_polygon_and_circles(main_polygon, circle_centers, circle_diameter_m)
