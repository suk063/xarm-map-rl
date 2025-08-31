import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

def fibonacci_sphere(n):
    points = []
    phi = math.pi * (3 - math.sqrt(5))  # Golden angle

    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # Project y to the surface of the sphere

        theta = phi * i  # Golden angle increment

        # Convert spherical coordinates to Cartesian coordinates
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return points

def plot_points_on_sphere(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, z coordinates from the points
    xs, ys, zs = zip(*points)

    # Plot the points on the sphere
    ax.scatter(xs, ys, zs, c='b', marker='o', s=50, label='Points on Sphere')

    # # Plot the initial points on the circle
    # circle_points = fibonacci_sphere(100)  # Increase the number for a smoother circle
    # circle_xs, circle_ys, circle_zs = zip(*circle_points)
    # ax.plot(circle_xs, circle_ys, circle_zs, color='gray', alpha=0.5, label='Points on Circle')

    # Set axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Points on a Sphere')

    # Set the aspect ratio of the plot to be equal
    ax.set_aspect('auto')

    # Set axis limits to better visualize the sphere
    max_range = max([max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs)]) / 2.0
    mid_x = (max(xs)+min(xs)) * 0.5
    mid_y = (max(ys)+min(ys)) * 0.5
    mid_z = (max(zs)+min(zs)) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Add a translucent spherical surface
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = max_range * np.outer(np.cos(u), np.sin(v)) + mid_x
    y = max_range * np.outer(np.sin(u), np.sin(v)) + mid_y
    z = max_range * np.outer(np.ones(np.size(u)), np.cos(v)) + mid_z
    ax.plot_surface(x, y, z, color='gray', alpha=0.2, linewidth=0)

    ax.legend()

    plt.show()


def inscribed_icosahedron(radius):
    t = (1.0 + np.sqrt(5.0)) / 2.0

    # Vertices of an icosahedron
    vertices = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1]
    ])

    # Scale and normalize the vertices to the desired radius
    vertices *= radius / np.linalg.norm(vertices, axis=1, keepdims=True)

    return vertices

def plot_points_on_sphere2(points, radius):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, z coordinates from the points
    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]

    # Plot the points on the sphere
    ax.scatter(xs, ys, zs, c='b', marker='o', s=50, label='Vertices of Icosahedron')

    # Set axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Vertices of an Inscribed Icosahedron on a Sphere')

    # Set the aspect ratio of the plot to be equal
    ax.set_aspect('auto')

    # Plot a translucent spherical surface with the specified radius
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='gray', alpha=0.2, linewidth=0)

    ax.legend()

    plt.show()


def generate_points_on_sphere(num_points):
    if num_points == 12:
        radius = 75
        icosahedron_vertices = inscribed_icosahedron(radius)

        rotation = Rotation.from_euler('xyz', [45, 45, 0], degrees=True)
        rotated_points = rotation.apply(icosahedron_vertices)

        plot_points_on_sphere2(rotated_points, radius)

        final_points = np.array(rotated_points) + np.array([radius + 207, 0, 0])
        print(f"Final points: {final_points}")
    else:
        points_on_sphere = fibonacci_sphere(num_points)
        plot_points_on_sphere(points_on_sphere)


def generate_grid():
    # Define the x, y, and z coordinate ranges
    x_values = [250, 400, 550]
    y_values = [-150, 0, 150]
    z_values = [60, 210, 360]

    # Create meshgrid for all combinations of x, y, and z coordinates
    x, y, z = np.meshgrid(x_values, y_values, z_values, indexing='ij')

    # Combine the meshgrid arrays to get the list of 3D coordinates
    coordinates_list = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

    print(coordinates_list)


if __name__ == "__main__":
    generate_grid()


'''
Icosahedron Output:
[[333.61432202, -17.23146904,  51.61432202],
 [294.18448861, -72.99367421,  12.18448861],
 [269.81551139,  72.99367421, -12.18448861],
 [230.38567798,  17.23146904, -51.61432202],
 [341.7805079 ,  45.11257163,   4.01830273],
 [286.01830273,  45.11257163,  59.7805079 ],
 [277.98169727, -45.11257163, -59.7805079 ],
 [222.2194921 , -45.11257163,  -4.01830273],
 [346.82748833, -27.88110258, -25.39765492],
 [256.60234508, -27.88110258,  64.82748833],
 [307.39765492,  27.88110258, -64.82748833],
 [217.17251167,  27.88110258,  25.39765492],]
'''

'''
Rearranged Meshgrid Output:
[[ 250, -75,   -30],
 [ 250, -75,  45],
 [ 250, -75,  120],
 [ 250,    0,  120],
 [ 250,    0,  45],
 [ 250,    0,   -30],
 [ 250,  75,   -30],
 [ 250,  75,  45],
 [ 250,  75,  120],
 [ 400,  75,  120],
 [ 400,  75,  45],
 [ 400,  75,   -30],
 [ 400,    0,   -30],
 [ 400,    0,  45],
 [ 400,    0,  120],
 [ 400, -75,  120],
 [ 400, -75,  45],
 [ 400, -75,   -30],
 [ 550, -75,   -30],
 [ 550, -75,  45],
 [ 550, -75,  120],
 [ 550,    0,  120],
 [ 550,    0,  45],
 [ 550,    0,   -30],
 [ 550,  75,   -30],
 [ 550,  75,  45],
 [ 550,  75,  120],]
'''