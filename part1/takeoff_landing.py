import numpy as np
from math import *
import os


# The following tries to avoid a warning when run on the linux machines via ssh.
if os.environ.get('DISPLAY') is None:
     import matplotlib 
     matplotlib.use('Agg')
       
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# All of the code in this file is just a starting point. Feel free

# 
def animate_above(frame_number): 
    global tx, ty, tz, compass, tilt, twist

    ty+=20
    tx += 5
    tz += 5
    
    pr=[]
    pc=[]

    pd = []

    for p in pts3:
        pr += [(p[0])/100000]
        pc += [(p[1]+ty)/100000]
        pd += [(p[2]+tz)/100000]
    
    if frame_number < 25:

      l = -0.0005 * (25-frame_number)/10
      delta = 0.05
      plt.cla()
      plt.gca().set_xlim([-l,l])
      plt.gca().set_ylim([-l,l])
      line, = plt.plot(pr,pc, 'k',  linestyle="", marker=".", markersize=2)
      return line,



# load in 3d point cloud
with open("airport.pts", "r") as f:
    pts3 = [ [ float(x) for x in l.split(" ") ] for l in f.readlines() ]

# initialize plane pose (translation and rotation)
(tx, ty, tz) = (0, 530, 0)
(compass, tilt, twist) = (0, pi/2, 0)

# create animation!
fig, ax  = plt.subplots()
frame_count = 22 # 5 seconds at 25 frames per second
ani = animation.FuncAnimation(fig, animate_above, frames=range(0,frame_count))



# ani.save("landing.mp4")

# uncomment if you want to display the movie on the screen (won't work on the
# remote linux servers if you are connecting via ssh)
plt.show()