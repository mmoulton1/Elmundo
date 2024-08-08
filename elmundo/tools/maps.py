import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
from geopy.distance import geodesic 
from shapely.geometry import Polygon, MultiPolygon, Point
from geodistance import find_closest_centroids, find_closest_point_on_polygon
from capital_costs import capital_cost_salt_cavern, capital_cost_underground_pipes, Compressor
import heapq
# Define input directories
INPUT_DIR1 = "/Users/mmoulton/Documents/Projects/Elmundo/inputs/tl_2023_us_state"
INPUT_DIR2 = "/Users/mmoulton/Documents/Projects/Elmundo/inputs/geologic_storage"
SITE_LIST_FILE = "/Users/mmoulton/Documents/Projects/Elmundo/inputs/sitelist_50082locs.csv"
storage_file = "/Users/mmoulton/Documents/Projects/Elmundo/inputs/sitelist_50082sites_forMatthew.xlsx"
storage_info_df = pd.read_excel(storage_file)

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
# Merge the site list with the storage information based on latitude and longitude
site_list = pd.merge(site_list, storage_info_df, on=['latitude', 'longitude'], how='left')

# Convert the site list to a GeoDataFrame
geometry = gpd.points_from_xy(site_list['longitude'], site_list['latitude'])
site_gdf = gpd.GeoDataFrame(site_list, geometry=geometry, crs="EPSG:4326")

# Reproject geologic storage data to a projected CRS (e.g., UTM)
geologic_storage_data = geologic_storage_data.to_crs(epsg=4326)  # Use appropriate UTM zone

# Calculate the centroids of the geologic storage shapes
geologic_storage_data['centroid'] = geologic_storage_data.geometry.centroid

# Extract the first point from site_gdf
first_site_point = site_gdf.iloc[0].geometry

# Find the 3 closest centroids to the first site point
closest_centroids = find_closest_centroids(first_site_point, geologic_storage_data[['centroid']])

# Print the 3 closest centroids and their distances
for distance, idx, centroid_point in closest_centroids:
    print(f"Distance: {distance} km, Centroid Index: {idx}, Centroid Point: {centroid_point}")

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

# Step 1: Collect closest centroids and their indices for each site point
closest_centroids_list = []
for _, site_point in excluded_sites_gdf.iterrows():   # Process only the first 10 sites
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

# Add latitude and longitude of the site and the closest point to the DataFrame
final_closest_points_df = pd.DataFrame(final_closest_points_list, columns=["site_name", "centroid_distance", "centroid_point", "polygon_distance", "closest_point", "polygon"])
final_closest_points_df['latitude'] = final_closest_points_df['site_name'].apply(lambda site: site_gdf.loc[site].geometry.y)
final_closest_points_df['longitude'] = final_closest_points_df['site_name'].apply(lambda site: site_gdf.loc[site].geometry.x)
final_closest_points_df['closest_point_latitude'] = final_closest_points_df['closest_point'].apply(lambda point: point.y)
final_closest_points_df['closest_point_longitude'] = final_closest_points_df['closest_point'].apply(lambda point: point.x)
print(excluded_sites_df.head())
# Merge final_closest_points_df with excluded_sites_df based on latitude and longitude so that the final data frame has the hydrogen storage data
final_closest_points_df = pd.merge(final_closest_points_df, excluded_sites_df, on=['latitude', 'longitude'], how='inner')

# Display the merged DataFrame
print(final_closest_points_df.head())

# Convert final_closest_points_list to DataFrame
final_df = pd.DataFrame(final_closest_points_list, columns=["site_name", "centroid_distance", "centroid_point", "polygon_distance", "closest_point", "polygon"])

import matplotlib.colors as mcolors
from shapely.geometry import LineString


def plot_sites(ax, state_data, geologic_storage_data, site_gdf, mask, color, label, marker):
    state_data.plot(ax=ax, color='white', edgecolor='black')
    geologic_storage_data.plot(ax=ax, color='blue', alpha=0.5)
    site_gdf[mask].plot(ax=ax, color=color, marker=marker, label=label, markersize=2)
    ax.legend()
    ax.set_title(label)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True)

def plot_closest_points(df, state_data, geologic_storage_data, site_gdf, cmap, norm):
    fig, ax = plt.subplots(figsize=(12, 8))
    state_data.plot(ax=ax, color='white', edgecolor='black', linewidth=1.5)
    geologic_storage_data.plot(ax=ax, color='blue', alpha=0.3)

    for _, row in df.iterrows():
        color = cmap(norm(row['polygon_distance']))
        site_point = site_gdf.loc[row['site_name']].geometry
        closest_point = row['closest_point']

        # Plot the site point
        ax.plot(site_point.x, site_point.y, marker='o', color=color, markersize=.125)

        # Plot the closest point
        ax.plot(closest_point.x, closest_point.y, marker='x', color=color, markersize=.125)

        # Draw a line between the site point and the closest point
        #line = LineString([site_point, closest_point])
        #ax.plot(*line.xy, color=color, linewidth=.25)

    # Add a color bar
   # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
   # sm.set_array([])
  #  fig.colorbar(sm, ax=ax, label='Distance to Closest Point (km)')
   # ax.set_xlabel('Longitude')
   # ax.set_ylabel('Latitude')
   # ax.set_title('Site Points and Closest Points')
   # plt.show()

# Plot Excluded and Included Sites
#fig, axes = plt.subplots(2, 1, figsize=(10, 20))
#plot_sites(axes[0], state_data, geologic_storage_data, site_gdf, within_polygon_mask, 'red', 'Excluded Sites', 'x')
#plot_sites(axes[1], state_data, geologic_storage_data, site_gdf, ~within_polygon_mask, 'green', 'Included Sites', 'o')
#plt.show()

# Normalize distances for color mapping
norm = plt.Normalize(final_closest_points_df['polygon_distance'].min(), final_closest_points_df['polygon_distance'].max())
cmap = plt.get_cmap('coolwarm')

# Plot Closest Points
#plot_closest_points(final_closest_points_df, state_data, geologic_storage_data, site_gdf, cmap, norm)

# Plot Site Points and Closest Points
#fig1, ax1 = plt.subplots(figsize=(12, 8))
#state_data.plot(ax=ax1, color='white', edgecolor='black', linewidth=1.5, label='State Data')
#geologic_storage_data.plot(ax=ax1, color='blue', alpha=0.3, label='Geologic Storage')

#for _, row in final_df.iterrows():
   # color = cmap(norm(row['polygon_distance']))
   # site_point = site_gdf.loc[row['site_name']].geometry
   # closest_point = row['closest_point']

    # Plot the site point
   # ax1.plot(site_point.x, site_point.y, marker='o', color=color, markersize=.5)
    
    # Plot the closest point
   # ax1.plot(closest_point.x, closest_point.y, marker='x', color=color, markersize=.25)
    
    # Draw a line between the site point and the closest point
    #line = LineString([site_point, closest_point])
  #  ax1.plot(*line.xy, color=color, linewidth=1)

# Add a color bar to indicate distances
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
#fig1.colorbar(sm, ax=ax1, label='Distance to Closest Point (km)')
#ax1.set_xlabel('Longitude')
#ax1.set_ylabel('Latitude')
#ax1.set_title('Site Points and Closest Points')
#plt.show()

# Plot Site Points, Closest Points, and Centroids
fig2, ax2 = plt.subplots(figsize=(12, 8))
state_data.plot(ax=ax2, color='white', edgecolor='black', linewidth=1.5, label='State Data')
geologic_storage_data.plot(ax=ax2, color='blue', alpha=0.3, label='Geologic Storage')

#for _, row in final_df.iterrows():
 #   color = cmap(norm(row['polygon_distance']))
 #   site_point = site_gdf.loc[row['site_name']].geometry
  #  closest_point = row['closest_point']
  #  centroid_point = row['centroid_point']

    # Plot the site point
  #  ax2.plot(site_point.x, site_point.y, marker='o', color=color, markersize=3)
    
    # Plot the closest point
   # ax2.plot(closest_point.x, closest_point.y, marker='x', color=color, markersize=3)
    
    # Plot the centroid point
   # ax2.plot(centroid_point.x, centroid_point.y, marker='^', color=color, markersize=3)

# Add a color bar to indicate distances
#sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#sm.set_array([])
#fig2.colorbar(sm, ax=ax2, label='Distance to Closest Point (km)')
#ax2.set_xlabel('Longitude')
#ax2.set_ylabel('Latitude')
#ax2.set_title('Site Points, Closest Points, and Centroids')
#plt.show()


# Example usage
storage_capacity_example = 1000000  # Example storage capacity in kg
capex_salt_cavern, cost_per_kg_H2_salt_cavern = capital_cost_salt_cavern(storage_capacity_example)
capex_underground_pipes, cost_per_kg_H2_underground_pipes = capital_cost_underground_pipes(storage_capacity_example)

#print(f"Salt Cavern - CAPEX: {capex_salt_cavern:.2f} USD, Cost per kg H2: {cost_per_kg_H2_salt_cavern:.2f} USD")
#print(f"Underground Pipes - CAPEX: {capex_underground_pipes:.2f} USD, Cost per kg H2: {cost_per_kg_H2_underground_pipes:.2f} USD")


# Now an example site that has 1000000kg of storage. I want to calulate the cost of the cavern and the cost of the compressors

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
#print(f"Motor Rating: {compressor.motor_rating:.2f} kW")
#print(f"Total Power: {compressor.motor_rating * compressor.n_compressors:.2f} kW")
#print(f"Total Capital Expenditure (Compressor System): ${total_capex_compressor:.2f}")
#print(f"Total Capital Expenditure (Salt Cavern Storage): ${capex_salt_cavern:.2f}")
#print(f"Total Capital Expenditure: ${total_capex:.2f}")

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


#Add legend and labels
#plt.legend()
#plt.title('Excluded and Included Sites with Geologic Storage Polygons')
#plt.xlabel('Longitude')
#plt.ylabel('Latitude')
#plt.grid(True)
#plt.show()


print(site_gdf[['latitude', 'longitude', 'hydrogen_storage_size_kg']].head(10))


# Import the functions from breakpoint.py
from breakpoint import combined_cost_salt_cavern_compressor_pipeline, combined_cost_underground_pipe_compressor, find_equilibrium_pipeline_length

# Apply the function directly using a lambda function
final_closest_points_df['salt_cavern_cost'] = final_closest_points_df.apply(
    lambda row: combined_cost_salt_cavern_compressor_pipeline(
        row['hydrogen_storage_size_kg'],
        100,  # p_outlet fixed at 100
        row['max_h2_kg_pr_hr'] * 24,  # Convert flow rate to kg/day
        row['polygon_distance']
    )['total_cost'],  # Extract total cost
    axis=1

)# Apply the function directly using a lambda function
final_closest_points_df['underground_pipe_cost'] = final_closest_points_df.apply(
    lambda row: combined_cost_underground_pipe_compressor(
        row['hydrogen_storage_size_kg'],
        100,  # p_outlet fixed at 100
        row['max_h2_kg_pr_hr'] * 24  # Convert flow rate to kg/day
    )['total_cost'],  # Extract total cost
    axis=1
)

# Display the updated DataFrame to verify the results
print(final_closest_points_df.head())
final_closest_points_df.to_csv("/Users/mmoulton/Documents/Projects/Elmundo/inputs/{}sites_final_sitelist_v1.csv".format(len(final_closest_points_df)))

# Filter sites within geologic storage polygons
sites_within_geologic_storage = site_gdf[within_polygon_mask]

# Initialize DataFrame for the comparison
within_storage_sites = sites_within_geologic_storage.copy()

# Add latitude and longitude to the DataFrame for easier reference
within_storage_sites['latitude'] = within_storage_sites['geometry'].apply(lambda geom: geom.y)
within_storage_sites['longitude'] = within_storage_sites['geometry'].apply(lambda geom: geom.x)

# Apply the function directly using a lambda function for salt cavern cost
within_storage_sites['salt_cavern_cost'] = within_storage_sites.apply(
    lambda row: combined_cost_salt_cavern_compressor_pipeline(
        row['hydrogen_storage_size_kg'],
        100,  # p_outlet fixed at 100
        row['max_h2_kg_pr_hr'] * 24,  # Convert flow rate to kg/day
        0  # Distance to cavern is zero
    )['total_cost'],  # Extract total cost
    axis=1
)

# Apply the function directly using a lambda function for underground pipe cost
within_storage_sites['underground_pipe_cost'] = within_storage_sites.apply(
    lambda row: combined_cost_underground_pipe_compressor(
        row['hydrogen_storage_size_kg'],
        100,  # p_outlet fixed at 100
        row['max_h2_kg_pr_hr'] * 24  # Convert flow rate to kg/day
    )['total_cost'],  # Extract total cost
    axis=1
)

# Display the first few rows of the within_storage_sites DataFrame
print(within_storage_sites.head())
# Save the within_storage_sites DataFrame to CSV with the filename containing the number of sites
within_storage_sites.to_csv("/Users/mmoulton/Documents/Projects/Elmundo/inputs/{}sites_within_storage_sites.csv".format(len(within_storage_sites)))

print(f"Data saved to /Users/mmoulton/Documents/Projects/Elmundo/inputs/{len(within_storage_sites)}sites_within_storage_sites_v1.csv")

def plot_salt_cavern_costs(df, state_data, geologic_storage_data, site_gdf):
    """
    Plot site points and closest points with colors representing the salt cavern storage costs.

    Parameters:
    df (DataFrame): The DataFrame containing the site points and their associated costs.
    state_data (GeoDataFrame): GeoDataFrame containing state boundary data.
    geologic_storage_data (GeoDataFrame): GeoDataFrame containing geologic storage locations.
    site_gdf (GeoDataFrame): GeoDataFrame containing the site points.
    """

    # Create a colormap and normalization based on the salt cavern cost
    cmap = plt.get_cmap('YlOrRd')
    norm = mcolors.Normalize(vmin=df['salt_cavern_cost'].min(), vmax=df['salt_cavern_cost'].max())
    
    fig, ax = plt.subplots(figsize=(12, 8))
    state_data.plot(ax=ax, color='white', edgecolor='black', linewidth=1.5)
    geologic_storage_data.plot(ax=ax, color='blue', alpha=0.3)

    for _, row in df.iterrows():
        color = cmap(norm(row['salt_cavern_cost']))
        site_point = site_gdf.loc[row['site_name']].geometry
        closest_point = row['closest_point']

        # Plot the site point
        ax.plot(site_point.x, site_point.y, marker='o', color=color, markersize=.125)

        # Plot the closest point
        #ax.plot(closest_point.x, closest_point.y, marker='x', color=color, markersize=.125)

        # Draw a line between the site point and the closest point
        #line = LineString([site_point, closest_point])
        #ax.plot(*line.xy, color=color, linewidth=.25)

    # Add a color bar based on the salt cavern cost
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Salt Cavern Storage Cost (USD)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Site Points with Salt Cavern Storage Cost')
    plt.show()

# Assuming final_closest_points_df, state_data, geologic_storage_data, and site_gdf are already defined
#plot_salt_cavern_costs(final_closest_points_df, state_data, geologic_storage_data, site_gdf)
#Whoever works on this next should make sure to use the CPI to convert all dollars to the same year. I did not do that yet.

