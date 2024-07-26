import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from shapely.geometry import Point, Polygon as ShapelyPolygon

def plot_project_idea(site_point, circle_radius, salt_cavern_location):
    fig, ax = plt.subplots(figsize=(8, 8))  # Set the figure size

    # Extract coordinates from site_point dictionary
    x = site_point['x']
    y = site_point['y']

    # Plot site point
    ax.plot(x, y, marker='o', color='blue', markersize=10, label='Site Point')

    # Plot circle around site point
    circle = Circle((x, y), circle_radius, edgecolor='black', facecolor='none', linestyle='--', label='Circle')
    ax.add_patch(circle)

    # Plot salt cavern (assuming salt_cavern_location is a Polygon or similar geometry)
    if salt_cavern_location is not None:
        ax.add_patch(Polygon(salt_cavern_location, closed=True, color='green', alpha=0.5, label='Salt Cavern'))

    # Create a Shapely Polygon object for the circle
    circle_geometry = Point(x, y).buffer(circle_radius)

    # Check if salt cavern is entirely within the circle
    if salt_cavern_location is not None:
        salt_cavern_polygon = ShapelyPolygon(salt_cavern_location)
        if circle_geometry.contains(salt_cavern_polygon):
            cheaper_option = 'Salt Cavern'
        else:
            cheaper_option = 'Underground Pipe'

        ax.text(0.5, 0.1, f'Cheaper Option: {cheaper_option}', transform=ax.transAxes, fontsize=12, ha='center')

    ax.set_xlim(0, 16)  # Set x-axis limits
    ax.set_ylim(0, 16)  # Set y-axis limits

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Project Idea Demonstration')
    ax.legend()

    # Set equal aspect ratio for both axes
    ax.set_aspect('equal')

    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage: Salt cavern outside the circle
site_point = {'x': 10, 'y': 10}  # Example coordinates
circle_radius = 5  # Example radius in km

# Example salt cavern polygon (outside the circle)
salt_cavern_location = [(2, 15), (3, 15), (3, 14), (2, 14)]

plot_project_idea(site_point, circle_radius, salt_cavern_location)

# Example usage: Salt cavern inside the circle (salt cavern is cheaper option)
site_point = {'x': 10, 'y': 10}  # Example coordinates
circle_radius = 5  # Example radius in km

# Example salt cavern polygon (inside the circle)
salt_cavern_location = [(9.5, 11.5), (10.5, 11.5), (10.5, 10.5), (9.5, 10.5)]

plot_project_idea(site_point, circle_radius, salt_cavern_location)
