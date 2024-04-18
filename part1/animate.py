#Author: Naman Nimbale

import numpy as np
from math import *
import os

# The following tries to avoid a warning when run on the linux machines via ssh.
if os.environ.get('DISPLAY') is None:
     import matplotlib 
     matplotlib.use('Agg')
       
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
        # if i == 2000:
        #     print(start, end)
        ax.plot([start[0], end[0]], [start[1], end[1]], color='black', linestyle="-")
    # print(len(pr))

    # Draw circles at each point
    for point in points:
        # print(point)
        x, y, _ = point
        ax.plot(x, y, 'o', color='black')

# All of the code in this file is just a starting point. Feel free
# to edit anything as you make your animation program.

# this function gets called every time a new frame should be generated.
# 
def animate_above(frame_number): 
    
    print('Current Frame-Number', frame_number)
    f_90 = int(frame_number/4)
    if frame_number>f_90 and frame_number<=2*f_90:
        camera_position = np.array([frame_number*10 + frame_number / 10, 10*frame_number, 500])
    elif frame_number>2*f_90 and frame_number<=3*f_90:
        camera_position = np.array([frame_number*50 + frame_number / 10, 100*frame_number, 500])
    elif frame_number>3*f_90 and frame_number<=4*f_90:
        camera_position = np.array([frame_number*100 + frame_number / 10, 10*frame_number, frame_number*10+5])
    else:
        camera_position = np.array([0, 2*frame_number, frame_number*10+5])


    # camera_position = np.array([ frame_number*10 + frame_number / 10,  frame_number*10 , frame_number+5])
    # camera_position = np.array([0, 2*frame_number, frame_number*10+5])
    projection_matrix = generate_projection_matrix(focal_length, camera_position, look_at)
    projected_points = project_points(points, projection_matrix)
    clipped_points = clip_points(projected_points)

    ax.clear()
    draw_points(ax, clipped_points)

    # global tx, ty, tz, compass, tilt, twist

    # ty+=20

    # pr=[]
    # pc=[]
    # for p in pts3:
    #     pr += [p[0]/100000]
    #     pc += [(p[1]+ty)/100000]

    # plt.cla()
    # plt.gca().set_xlim([-0.002,0.002])
    # plt.gca().set_ylim([-0.002,0.002])
    # line, = plt.plot(pr, pc, 'k',  linestyle="", marker=".", markersize=2)
    # return line,

# load in 3d point cloud
# with open("airport.pts", "r") as f:
#     pts3 = [ [ float(x) for x in l.split(" ") ] for l in f.readlines() ]
points = load_points("airport.pts")
points = compute_homogeneous_coords(points)

# # initialize plane pose (translation and rotation)


focal_length = 0.002
image_width = 100
image_height = 100
camera_position = np.array([0, 0, 5]) #(tx, ty, tz) = (0, 0, -10)
look_at = np.array([0, pi/2, 0]) # (compass, tilt, twist) = (0, pi/2, 0)


# create animation!
# fig, ax  = plt.subplots()
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.set_xlim([-5, image_width])
ax.set_ylim([-5, image_height])
frame_count = 50
ani = animation.FuncAnimation(fig, animate_above, frames=range(0,frame_count))
my_ani = ani
my_ani.save("movie.gif")

# uncomment if you want to display the movie on the screen (won't work on the
# remote linux servers if you are connecting via ssh)
plt.show()


