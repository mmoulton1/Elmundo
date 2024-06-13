import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
from inputs import INPUT_DIR
from geopy.distance import geodesic 
from geopy.point import Point


# read in sitelist file
sitelist_filename = "sitelist_50082locs.csv"
site_list = pd.read_csv(os.path.join(INPUT_DIR,sitelist_filename),index_col = "Unnamed: 0")

# convert lat/lons to numeric values
site_list["latitude"] = pd.to_numeric(site_list["latitude"], errors='coerce')
site_list["longitude"] = pd.to_numeric(site_list["longitude"], errors='coerce')

#convert site_list dataframe to geodataframe
df_geo = gpd.GeoDataFrame(site_list, 
                        geometry = gpd.points_from_xy(
                        site_list.longitude,
                        site_list.latitude
                        ))

# read in salt cavern shape file
salt_cavern_file = os.path.join(INPUT_DIR,"geologic_storage","salt_Shapefile_4326.shp")
salt_cavern = gpd.read_file(salt_cavern_file,crs=4326)

#calculate distance between two points
def find_closest_point(points, target_location):
    #target_location = (site_list['latitude'][i], site_list['longitude'][i])
    min_distance = float('inf')
    closest_point = None

    target_point = Point(target_location[0], target_location[1])

    for point in points:
        current_point = Point(point[0], point[1])
        distance = geodesic(target_point, current_point).km

        if distance < min_distance:
            min_distance = distance
            closest_point = point
            

    return closest_point, min_distance