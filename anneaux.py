import numpy as np
from pylab import *
from matplotlib import pyplot as plt
from matplotlib import animation

import compute_gamma
compute_gamma = reload(compute_gamma)

import cost

# First set up the figure, the axis, and the plot element we want to animate
N = 30  # number of sampled points on the path
W = 101  # image width
H = 101  # image height
fig = plt.figure()
ax = plt.axes(xlim=(0, W-1), ylim=(0, H-1))
depth_map = cost.load_image('morne_rouge.asc', W, H)


A = (3, 4)
B = (60,80)

gamma = compute_gamma.compute_gamma(depth_map,W,H,A,B)
# gamma = cost.prepare_gamma(np.array(A),np.array(B),20)

# gamma = np.array(gamma)
# im = plt.imshow(gradx, interpolation='nearest')
# plot(gamma[:,1], gamma[:,0], '-x')
# show()

fig = plt.figure()
ax = plt.axes(xlim=(0, W-1), ylim=(0, H-1))
im = plt.imshow(depth_map, interpolation='nearest')
pathplot = ax.plot([], [], '-x')
pathplot = pathplot[0]
# gamma = cost.prepare_gamma(np.array([50, 20]), np.array([50, 80]), N)
c, J_record = cost.cost(depth_map, gamma, 1, 0.25)

def color_set(im, J):
    """
    Marks given pixels with NaN on a given image.

    Args:
        im: numpy 2D array representing the input image
        J: set containing tuples of size 2 representing pixels coordinates

    Returns:
        a copy of the input image with the given pixels marked as NaN
    """
    out = im.copy()
    for p in J:
        out[p[0], p[1]] = np.nan
    return out


def animate(i):
    """
    Animation function. This is called sequentially.
    """
    a = color_set(depth_map, J_record[i])
    im.set_array(a)
    gammabis = np.array(gamma)
    pathplot.set_data(gammabis[:,1], gammabis[:,0])
    return [im,pathplot]



anim = animation.FuncAnimation(fig, animate, len(J_record),
                        interval=500, blit=False)
plt.show()
