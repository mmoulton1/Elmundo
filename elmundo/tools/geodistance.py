from geopy.distance import geodesic 
from shapely.geometry import Point as ShapelyPoint, Polygon, MultiPolygon

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


# Function to find the closest point on the polygon boundary
def find_closest_point_on_polygon(site_point, polygon):
    site_coords = (site_point.y, site_point.x)  # Assuming site_point is a Point object
    min_distance = float('inf')
    closest_point = None
    
    if polygon.geom_type == 'Polygon':
        # Handle Polygon
        for coords in polygon.exterior.coords:
            poly_point = ShapelyPoint(coords[0], coords[1])  # Use ShapelyPoint instead of Point
            distance = geodesic(site_coords, (poly_point.y, poly_point.x)).kilometers
            if distance < min_distance:
                min_distance = distance
                closest_point = poly_point
    
    elif polygon.geom_type == 'MultiPolygon':
        # Handle MultiPolygon
        for poly in polygon.geoms:
            for coords in poly.exterior.coords:
                poly_point = ShapelyPoint(coords[0], coords[1])  # Use ShapelyPoint instead of Point
                distance = geodesic(site_coords, (poly_point.y, poly_point.x)).kilometers
                if distance < min_distance:
                    min_distance = distance
                    closest_point = poly_point
    
    return closest_point, min_distance