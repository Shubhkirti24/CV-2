from math import *
import numpy as np
import matplotlib.pyplot as plt

def load_points(file_path):
    with open(file_path, "r") as f:
        points = [list(map(float, line.split())) for line in f]
    return points


def compute_homogeneous_coords(points):
    return np.hstack([points, np.ones((len(points), 1))])



def generate_projection_matrix(focal_length, camera_position, look_at):
    compass, tilt, twist = look_at

    rot_compass = np.array([[np.cos(compass), -np.sin(compass), 0],
                            [np.sin(compass), np.cos(compass), 0],
                            [0, 0, 1]])

    rot_tilt = np.array([[1, 0, 0],
                         [0, np.cos(tilt), -np.sin(tilt)],
                         [0, np.sin(tilt), np.cos(tilt)]])

    rot_twist = np.array([[np.cos(twist), 0 ,-np.sin(twist)],
                          [0, 1, 0],
                          [np.sin(twist), 0 ,np.cos(twist)]])

    rotation = rot_tilt @ rot_twist @ rot_compass

    # Calculate translation matrix
    # print(camera_position)
    translation = np.array([[1, 0, 0, -camera_position[0]], [0, 1, 0, -camera_position[1]], [0, 0, 1, -camera_position[2]], [0, 0, 0, 1]])

    # Calculate intrinsic matrix
    K = np.array([[focal_length, 0, 0], [0, focal_length, 0], [0, 0, 1]])

    # Calculate projection matrix
    P = K @ np.hstack([rotation, np.zeros((3, 1))]) @ translation

    return P
  

def project_points(points, projection_matrix):
    projected_points = np.dot(projection_matrix, points.T)
    # projected_points /= projected_points[2, :]
    z_nonzero_mask = projected_points[2, :] != 0
    projected_points[:, z_nonzero_mask] /= projected_points[2, z_nonzero_mask]
    projected_points = projected_points.T

    return projected_points


def clip_points(points):
    return [point for point in points if point[2] >= 0]


def draw_points(ax, points):
    # Draw lines between adjacent points
    pr, pc = list(), list()
    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]
        ax.plot([start[0], end[0]], [start[1], end[1]], color='black', linestyle="-")
    # print(len(pr))

    # Draw circles at each point
    for point in points:
        # print(point)
        x, y, _ = point
        ax.plot(x, y, 'o', color='black')



def main():
    points = load_points("airport.pts")
    points = compute_homogeneous_coords(points)
    focal_length = 0.002
    image_width = 100
    image_height = 100
    camera_position = np.array([0, 0, 5])
    look_at = np.array([0, pi/2, 0])
    # up_direction = np.array([0, 0, -1])

    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_xlim([-5, image_width])
    ax.set_ylim([-5, image_height])

    # projection_matrix = generate_projection_matrix(focal_length, image_width, image_height, camera_position, look_at, up_direction)

    for i in range(0, 200, 4):
        # camera_position = np.array([ i*10 + i / 10,  i*10 , i+5])
        camera_position = np.array([0, 2*i, i+5])
        if i>20 and i<=30:
            look_at = np.array([0, pi/2, pi/2])
        elif i>30 and i<=40:
            look_at = np.array([0, pi/2, pi])


        projection_matrix = generate_projection_matrix(focal_length, camera_position, look_at)
        projected_points = project_points(points, projection_matrix)
        clipped_points = clip_points(projected_points)

        ax.clear()
        draw_points(ax, clipped_points)

        plt.draw()
        plt.pause(0.01)


    plt.show()


if __name__ == "__main__":
    main()
