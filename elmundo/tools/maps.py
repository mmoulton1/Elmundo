import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
from geopy.distance import geodesic 
from geopy.point import Point
import heapq
from shapely.geometry import Polygon, MultiPolygon, Point
# Define input directories
INPUT_DIR1 = "/Users/mmoulton/Documents/Projects/Elmundo/inputs/tl_2023_us_state"
INPUT_DIR2 = "/Users/mmoulton/Documents/Projects/Elmundo/inputs/geologic_storage"
SITE_LIST_FILE = "/Users/mmoulton/Documents/Projects/Elmundo/inputs/sitelist_50082locs.csv"

# Define the shapefile paths
shapefile_path1 = os.path.join(INPUT_DIR1, "tl_2023_us_state.shp")
shapefile_path2 = os.path.join(INPUT_DIR2, "salt_Shapefile_4326.shp")

# Read the shapefiles using GeoPandas
state_data = gpd.read_file(shapefile_path1)
geologic_storage_data = gpd.read_file(shapefile_path2)

# Exclude non-continental states
non_continental = ['HI', 'VI', 'MP', 'GU', 'AK', 'AS', 'PR']
state_data = state_data[~state_data['STUSPS'].isin(non_continental)]

# Read the site list
site_list = pd.read_csv(SITE_LIST_FILE, index_col="Unnamed: 0")

# Convert the site list to a GeoDataFrame
geometry = gpd.points_from_xy(site_list['longitude'], site_list['latitude'])
site_gdf = gpd.GeoDataFrame(site_list, geometry=geometry, crs="EPSG:4326")

# Reproject geologic storage data to a projected CRS (e.g., UTM)
geologic_storage_data = geologic_storage_data.to_crs(epsg=4326)  # Use appropriate UTM zone

# Calculate the centroids of the geologic storage shapes
geologic_storage_data['centroid'] = geologic_storage_data.geometry.centroid
print(geologic_storage_data.crs)

# Function to find the 3 closest centroids
def find_closest_centroids(site_point, centroids, k=3):
    distances = []
    for idx, centroid in centroids.iterrows():
        # Extract centroid point from the row
        centroid_point = centroid['centroid']
        
        # Calculate distance using geodesic
        distance = geodesic((site_point.y, site_point.x), (centroid_point.y, centroid_point.x)).kilometers
        
        # Append distance along with centroid index and point to distances list
        distances.append((distance, idx, centroid_point))
    
    # Sort distances list based on distance and return top k closest centroids
    closest_centroids = sorted(distances)[:k]
    
    return closest_centroids


# Extract the first point from site_gdf
first_site_point = site_gdf.iloc[0].geometry

# Find the 3 closest centroids to the first site point
closest_centroids = find_closest_centroids(first_site_point, geologic_storage_data[['centroid']])

# Print the 3 closest centroids and their distances
for distance, idx, centroid_point in closest_centroids:
    print(f"Distance: {distance} km, Centroid Index: {idx}, Centroid Point: {centroid_point}")



# Function to find the closest point on the polygon boundary
def find_closest_point_on_polygon(site_point, polygon):
    site_coords = (site_point.y, site_point.x)  # Assuming site_point is a Point object
    min_distance = float('inf')
    closest_point = None
    
    if polygon.geom_type == 'Polygon':
        # Handle Polygon
        for coords in polygon.exterior.coords:
            poly_point = Point(coords[0], coords[1])
            distance = geodesic(site_coords, (poly_point.y, poly_point.x)).kilometers
            if distance < min_distance:
                min_distance = distance
                closest_point = poly_point
    
    elif polygon.geom_type == 'MultiPolygon':
        # Handle MultiPolygon
        for poly in polygon.geoms:
            for coords in poly.exterior.coords:
                poly_point = Point(coords[0], coords[1])
                distance = geodesic(site_coords, (poly_point.y, poly_point.x)).kilometers
                if distance < min_distance:
                    min_distance = distance
                    closest_point = poly_point
    
    return closest_point, min_distance

# Find the polygons that contain these centroids and calculate the closest point on the polygon boundary
closest_points = []
for distance, idx, centroid in closest_centroids:
    polygon = geologic_storage_data.loc[idx].geometry
    closest_point, min_distance = find_closest_point_on_polygon(first_site_point, polygon)
    closest_points.append((min_distance, closest_point, polygon))

# Print the closest points and their distances
for distance, closest_point, polygon in closest_points:
    print(f"Distance: {distance} km, Closest Point: {closest_point}")


# Function to check if a point is within any polygon
def is_within_any_polygon(point, polygons):
    for polygon in polygons:
        if point.within(polygon):
            return True
    return False


# Check if each site point is within any geologic storage polygon
within_polygon_mask = site_gdf['geometry'].apply(lambda point: is_within_any_polygon(point, geologic_storage_data.geometry))

# Filter out the site points that are within any polygon
excluded_sites_gdf = site_gdf[~within_polygon_mask]

# Create a DataFrame with the remaining site points
excluded_sites_df = excluded_sites_gdf.drop(columns='geometry')

# Plot the state data, geologic storage data, and the remaining site points
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
state_data.plot(ax=ax, color='white', edgecolor='black')
geologic_storage_data.plot(ax=ax, color='blue', alpha=0.5)
excluded_sites_gdf.plot(ax=ax, color='red', markersize=0.1)

# Step 1: Collect closest centroids and their indices for each site point
closest_centroids_list = []
for _, site_point in excluded_sites_gdf.head(5).iterrows():   # Process only the first 5 sites
    closest_centroids = find_closest_centroids(site_point.geometry, geologic_storage_data[['centroid']])
    closest_centroids_list.append((site_point.name, closest_centroids))



# Step 2: Collect closest polygons corresponding to the centroids
closest_polygons_list = []
for site_name, closest_centroids in closest_centroids_list:
    for distance, idx, centroid_point in closest_centroids:
        polygon = geologic_storage_data.loc[idx].geometry
        closest_polygons_list.append((site_name, polygon, centroid_point, distance))


# Step 3: Find closest points on polygons
closest_points_list = []
for site_name, polygon, centroid_point, centroid_distance in closest_polygons_list:
    closest_point, min_distance = find_closest_point_on_polygon(site_gdf.loc[site_name].geometry, polygon)
    closest_points_list.append((site_name, centroid_distance, centroid_point, min_distance, closest_point, polygon))

# Step 4: Select the closest point among the 3 for each site
final_closest_points_list = []
df_closest_points = pd.DataFrame(closest_points_list, columns=["site_name", "centroid_distance", "centroid_point", "polygon_distance", "closest_point", "polygon"])
for site_name, group in df_closest_points.groupby('site_name'):
    closest_point = min(group.values.tolist(), key=lambda x: x[3])  # Select the one with the shortest polygon distance
    final_closest_points_list.append(closest_point)




# Print the site name, distances to the closest centroids, and closest points on the polygons
for site_name, closest_centroids in closest_centroids_list:
    print(f"Site: {site_name}")
    for distance, idx, centroid_point in closest_centroids:
        print(f"  Distance to Centroid: {distance:.2f} km, Centroid Point: {centroid_point}")

for site_name, centroid_distance, centroid_point, polygon_distance, closest_point, polygon in final_closest_points_list:
    print(f"Site: {site_name}, Distance to Centroid: {centroid_distance:.2f} km, Distance to Closest Point on Polygon: {polygon_distance:.2f} km, Closest Point: {closest_point}")


# Step to find and print closest points among the 3 polygons for each site
for site_name, group in pd.DataFrame(closest_points_list, columns=["site_name", "centroid_distance", "centroid_point", "polygon_distance", "closest_point", "polygon"]).groupby('site_name'):
    closest_points = []
    for i, row in group.iterrows():
        closest_points.append((row['polygon_distance'], row['closest_point']))
    
    # Sort by polygon distance and get the closest points
    closest_points = sorted(closest_points)[:3]
    
    # Print closest points for the current site
    print(f"Site: {site_name}, Closest Points on All Polygons:")
    for j, (distance, point) in enumerate(closest_points, start=1):
        print(f"  Closest Point {j}: Distance: {distance:.2f} km, Point: ({point.x}, {point.y})")
    print()  # Print an empty line for separation between sites

print('elmundo')


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from shapely.geometry import LineString

# Convert final_closest_points_list to DataFrame
final_df = pd.DataFrame(final_closest_points_list, columns=["site_name", "centroid_distance", "centroid_point", "polygon_distance", "closest_point", "polygon"])
# Normalize the distances for color mapping
norm = plt.Normalize(final_df['polygon_distance'].min(), final_df['polygon_distance'].max())
cmap = plt.get_cmap('coolwarm')  # Use any colormap of your choice


# Plot the state data, geologic storage data, and only the first 5 site points with colors based on distance
fig1, ax1 = plt.subplots(figsize=(12, 8))

state_data.plot(ax=ax1, color='white', edgecolor='black', linewidth=1.5, label='State Data')
geologic_storage_data.plot(ax=ax1, color='blue', alpha=0.3, label='Geologic Storage')

# Initialize handles and labels for legend
handles = []
labels = []


# Plot each site point with the corresponding color and draw lines to the closest points
for _, row in final_df.iterrows():
    color = cmap(norm(row['polygon_distance']))
    site_point = site_gdf.loc[row['site_name']].geometry
    closest_point = row['closest_point']
    
    # Plot the site point
    site_handle = ax1.plot(site_point.x, site_point.y, marker='o', color=color, markersize=3)[0]
    handles.append(site_handle)
    labels.append(f'Site {row["site_name"]}')
    
    # Plot the closest point
    closest_handle = ax1.plot(closest_point.x, closest_point.y, marker='x', color=color, markersize=3)[0]
    handles.append(closest_handle)
    labels.append(f'Closest Point {row["site_name"]}')
    
    # Draw a line between the site point and the closest point
    line = LineString([site_point, closest_point])
    ax1.plot(*line.xy, color=color, linewidth=1)

# Add a color bar to indicate distances
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Only needed for older versions of matplotlib
fig1.colorbar(sm, ax=ax1, label='Distance to Closest Point (km)')

# Set labels for axes
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.set_title('Site Points and Closest Points')

# Create legend with custom handles and labels
ax1.legend(handles, labels, loc='upper left', bbox_to_anchor=(.12, 1.52))
plt.show()

# Plot 2: Site Points with Centroids
fig2, ax2 = plt.subplots(figsize=(12, 8))

# Plot the state data
state_data.plot(ax=ax2, color='white', edgecolor='black', linewidth=1.5, label='State Data')
geologic_storage_data.plot(ax=ax2, color='blue', alpha=0.3, label='Geologic Storage')

# Initialize handles and labels for legend
handles = []
labels = []

# Plot each site point, closest point, and centroid with the corresponding color
for _, row in final_df.iterrows():
    color = cmap(norm(row['polygon_distance']))
    site_point = site_gdf.loc[row['site_name']].geometry
    closest_point = row['closest_point']
    centroid_point = row['centroid_point']
    
    # Plot the site point
    site_handle = ax2.plot(site_point.x, site_point.y, marker='o', color=color, markersize=3)[0]
    handles.append(site_handle)
    labels.append(f'Site {row["site_name"]}')
    
    # Plot the closest point
    closest_handle = ax2.plot(closest_point.x, closest_point.y, marker='x', color=color, markersize=3)[0]
    handles.append(closest_handle)
    labels.append(f'Closest Point {row["site_name"]}')
    
    # Plot the centroid point
    centroid_handle = ax2.plot(centroid_point.x, centroid_point.y, marker='^', color=color, markersize=3)[0]
    handles.append(centroid_handle)
    labels.append(f'Centroid {row["site_name"]}')

# Add a color bar to indicate distances
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig2.colorbar(sm, ax=ax2, label='Distance to Closest Point (km)')

# Set labels for axes
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
ax2.set_title('Site Points, Closest Points, and Centroids')

# Create legend with custom handles and labels
ax2.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.5, 1))

plt.show()

#define a function for the capital cost of H2 storage in salt caverns. I am using the model from Papadias
def capital_cost_salt_cavern(storage_capacity): 
    """
    Calculate the capital cost of H2 storage in salt caverns using the Papadias model.

    Parameters:
    storage_capacity (float): Storage capacity of the cavern in kg.

    Returns:
    float: Adjusted installed capital cost in 2018 USD.
    """
    # Installed capital cost
    a = 0.092548
    b = 1.6432
    c = 10.161
    cost_per_kg_H2 = np.exp(a*(np.log(storage_capacity/1000))**2 - b*np.log(storage_capacity/1000) + c)  # 2019 [USD] from Papadias
    installed_capex = cost_per_kg_H2 * storage_capacity   
    cepci_overall = 1.29/1.30 # Convert from $2019 to $2018
    adjusted_installed_capex = installed_capex * cepci_overall
    return adjusted_installed_capex, cost_per_kg_H2

c = capital_cost_salt_cavern(1000000)

def capital_cost_underground_pipes(storage_capacity):
    """
    Calculate the capital cost of H2 storage in underground pipes using a similar model to Papadias.

    Parameters:
    storage_capacity (float): Storage capacity of the pipes in kg.

    Returns:
    float: Adjusted installed capital cost in 2018 USD.
    float: Cost per kg H2 in 2019 USD.
    """
    # Installed capital cost constants for underground pipes
    a = 0.001559
    b = 0.035313
    c = 4.5183

    # Calculate cost per kg H2 in 2019 USD
    cost_per_kg_H2 = np.exp(a * (np.log(storage_capacity / 1000)) ** 2 - b * np.log(storage_capacity / 1000) + c)
    
    # Calculate installed capital expenditure (CAPEX) in 2019 USD
    installed_capex = cost_per_kg_H2 * storage_capacity
    
    # Cost adjustment from 2019 to 2018 USD
    cepci_overall = 1.29 / 1.30
    adjusted_installed_capex = installed_capex * cepci_overall
    
    return adjusted_installed_capex, cost_per_kg_H2

# Example usage
storage_capacity_example = 1000000  # Example storage capacity in kg
capex_salt_cavern, cost_per_kg_H2_salt_cavern = capital_cost_salt_cavern(storage_capacity_example)
capex_underground_pipes, cost_per_kg_H2_underground_pipes = capital_cost_underground_pipes(storage_capacity_example)

print(f"Salt Cavern - CAPEX: {capex_salt_cavern:.2f} USD, Cost per kg H2: {cost_per_kg_H2_salt_cavern:.2f} USD")
print(f"Underground Pipes - CAPEX: {capex_underground_pipes:.2f} USD, Cost per kg H2: {cost_per_kg_H2_underground_pipes:.2f} USD")
print('elmundo')
print(c)
'''
    Author: Jamie Kee
    Feb 7, 2023
    Source for most equations is HDSAM3.1, H2 Compressor sheet
    Output is in 2016 USD
'''

from numpy import interp, mean, dot
from math import log10, ceil, log

class Compressor:
    def __init__(self, p_outlet, flow_rate_kg_d, p_inlet=20, n_compressors=2, sizing_safety_factor=1.1):
        '''
            Parameters:
            ---------------
            p_outlet: oulet pressure (bar)
            flow_Rate_kg_d: mass flow rate in kg/day
        '''
        self.p_inlet = p_inlet # bar
        self.p_outlet = p_outlet # bar
        self.flow_rate_kg_d = flow_rate_kg_d # kg/day

        self.n_compressors = n_compressors # At least 2 compressors are recommended for operation at any given time
        self.n_comp_back_up = 1 # Often times, an extra compressor is purchased and installed so that the system can operate at a higher availability.
        self.sizing_safety_factor = sizing_safety_factor # typically oversized. Default to oversize by 10%

        if flow_rate_kg_d*(1/24)*(1/60**2)/n_compressors > 5.4:
            # largest compressors can only do up to about 5.4 kg/s
            """
            H2A Hydrogen Delivery Infrastructure Analysis Models and Conventional Pathway Options Analysis Results
            DE-FG36-05GO15032
            Interim Report
            Nexant, Inc., Air Liquide, Argonne National Laboratory, Chevron Technology Venture, Gas Technology Institute, National Renewable Energy Laboratory, Pacific Northwest National Laboratory, and TIAX LLC
            May 2008
            """
            raise ValueError("Invalid compressor design. Flow rate must be less than 5.4 kg/s per compressor")
            
    def compressor_power(self):
        R = 8.314 # Universal gas constant in J/mol-K
        T = 25 + 273.15 # Temperature in Kelvin (25°C)
    
        cpcv = 1.41 # Specific heat ratio (Cp/Cv) for hydrogen
        sizing = self.sizing_safety_factor # Sizing safety factor, typically 110%
        isentropic_efficiency = 0.88 # Isentropic efficiency for a reciprocating compressor

        # Hydrogen compressibility factor (Z) at different pressures (bar) and 25°C
        Z_pressures = [1, 10, 50, 100, 300, 500, 1000]
        Z_z = [1.0006, 1.0059, 1.0297, 1.0601, 1.1879, 1.3197, 1.6454]
        Z = np.mean(np.interp([self.p_inlet, self.p_outlet], Z_pressures, Z_z)) # Average Z factor between inlet and outlet pressures

        # Compression ratio per stage
        c_ratio_per_stage = 2.1
        # Number of stages required
        self.stages = np.ceil((np.log10(self.p_outlet) - np.log10(self.p_inlet)) / np.log10(c_ratio_per_stage))
        
        # Flow rate per compressor (kg/day) converted to kg-mols/sec
        flow_per_compressor = self.flow_rate_kg_d / self.n_compressors # kg/day
        flow_per_compressor_kg_mols_sec = flow_per_compressor / (24 * 60 * 60) / 2.0158 # kg-mols/sec

        # Pressure ratio
        p_ratio = self.p_outlet / self.p_inlet

        # Theoretical power required for compression (kW)
        theoretical_power = Z * flow_per_compressor_kg_mols_sec * R * T * self.stages * (cpcv / (cpcv - 1)) * ((p_ratio) ** ((cpcv - 1) / (self.stages * cpcv)) - 1)
        # Actual power required considering isentropic efficiency (kW)
        actual_power = theoretical_power / isentropic_efficiency

        # Motor efficiency based on empirical formula
        motor_efficiency = np.dot(
            [0.00008, -0.0015, 0.0061, 0.0311, 0.7617],
            [np.log(actual_power) ** x for x in [4, 3, 2, 1, 0]]
        )
        # Motor rating with sizing safety factor (kW)
        self.motor_rating = sizing * actual_power / motor_efficiency
    def compressor_system_power(self):
            return self.motor_rating, self.motor_rating*self.n_compressors # [kW] total system power
    
    def compressor_costs(self):
        n_comp_total = self.n_compressors + self.n_comp_back_up # 2 compressors + 1 backup for reliability
        production_volume_factor = 0.55 # Assume high production volume
        CEPCI = 1.29/1.1 #Convert from 2007 to 2016$

        cost_per_unit = 1962.2*self.motor_rating**0.8225*production_volume_factor*CEPCI
        if self.stages>2:
            cost_per_unit = cost_per_unit * (1+0.2*(self.stages-2))

        install_cost_factor = 2

        direct_capex = cost_per_unit*n_comp_total*install_cost_factor

        land_required = 10000 #m^2 This doesn't change at all in HDSAM...?
        land_cost = 12.35 #$/m2
        land = land_required*land_cost

        other_capital_pct = [0.05,0.1,0.1,0,0.03,0.12] # These are all percentages of direct capex (site,E&D,contingency,licensing,permitting,owners cost)
        other_capital = dot(other_capital_pct,[direct_capex]*len(other_capital_pct)) + land
        
        total_capex = direct_capex + other_capital
        return total_capex

## Now an exaplle site that has 1000000kg of storage. I want to calulate the cost of the cavern and the cost of the compressors

# Example usage:
p_outlet = 50  # Example outlet pressure in bar
flow_rate_kg_d = 5000  # Example flow rate in kg/day

# Create an instance of Compressor
compressor = Compressor(p_outlet, flow_rate_kg_d)
compressor.compressor_power()  # Calculate compressor power
total_capex_compressor = compressor.compressor_costs()  # Calculate compressor system cost

# Example salt cavern storage cost calculation
storage_capacity = 1000000  # Example storage capacity in kg
capex_salt_cavern, _ = capital_cost_salt_cavern(storage_capacity)  # Calculate salt cavern storage cost

# Total capital expenditure including both costs
total_capex = capex_salt_cavern + total_capex_compressor

# Print results
print(f"Motor Rating: {compressor.motor_rating:.2f} kW")
print(f"Total Power: {compressor.motor_rating * compressor.n_compressors:.2f} kW")
print(f"Total Capital Expenditure (Compressor System): ${total_capex_compressor:.2f}")
print(f"Total Capital Expenditure (Salt Cavern Storage): ${capex_salt_cavern:.2f}")
print(f"Total Capital Expenditure: ${total_capex:.2f}")

# Initialize the plot
fig, ax = plt.subplots(figsize=(10, 10))

# Plot state boundaries
state_data.plot(ax=ax, color='white', edgecolor='black')

# Plot geologic storage polygons
geologic_storage_data.plot(ax=ax, color='blue', alpha=0.5, markersize=2)

# Plot sites with 'within_polygon_mask' as True (excluded sites)
site_gdf[within_polygon_mask].plot(ax=ax, color='red', marker='x', label='Excluded Sites',markersize=2)

# Optionally plot the non-excluded sites for contrast
site_gdf[~within_polygon_mask].plot(ax=ax, color='green', marker='o', label='Included Sites')


# Add legend and labels
plt.legend()
plt.title('Excluded and Included Sites with Geologic Storage Polygons')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()