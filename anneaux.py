import numpy as np
from pylab import *
from matplotlib import pyplot as plt
from matplotlib import animation

import cost


# First set up the figure, the axis, and the plot element we want to animate
N = 30  # number of sampled points on the path
W = 101  # image width
H = 101  # image height
fig = plt.figure()
ax = plt.axes(xlim=(0, W-1), ylim=(0, H-1))
depth_map = cost.load_image('morne_rouge.asc', W, H)

def compute_gamma(depth_map,W,H,A,B):
    r1 = 4
    r2 = 8

    r1 = 8
    r2 = 14


    grad = np.gradient(depth_map)
    gradx = grad[0]
    grady = grad[1]

    def cross_prod(X, Y):
        return X[0]*Y[1] - X[1]*Y[0]
    def dist(X, Y):
        return sqrt((X[0] - Y[0])**2 + (X[1] - Y[1])**2)

    gamma = list()
    gamma.append((A[0],A[1]))
    iter = 0
    maxit = 100

    while gamma[-1] != B and iter < maxit:
        iter = iter + 1
        curpt = gamma[-1]
        grad_cur = [gradx[curpt[0],curpt[1]], grady[curpt[0],curpt[1]]]
        score = np.zeros((W,H))
        for i in range(W):
            for j in range(H):
                dist_to_cur = dist(curpt, [i,j])
                dist_to_B = dist([i,j], B)
                dist_from_cur_to_B = dist(curpt, B)
                if dist_to_B < dist_from_cur_to_B and dist_to_cur > r1 and dist_to_cur < r2:
                    score[i,j] = abs(cross_prod(grad_cur, [gradx[i,j], grady[i,j]]))
                    # if([i,j] == B or dist([i,j], B) < r2):
                    #     score[i,j] = inf

                else:
                    score[i,j] = -1


        imax,jmax = np.unravel_index(score.argmax(), score.shape)
        if dist(curpt, B) < r2:
            imax,jmax=B[0],B[1]
        gamma.append((imax,jmax))
    gamma.append(B)
    return gamma

A = (3, 4)
B = (60,80)

gamma = compute_gamma(depth_map,W,H,A,B)
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
