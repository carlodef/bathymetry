import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

import cost


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


def animate(i, depth_map, gamma, J_record, im, pathplot):
    """
    Animation function. This is called sequentially.
    """
    a = color_set(depth_map, J_record[i])
    im.set_array(a)
    gammabis = np.array(gamma)
    pathplot.set_data(gammabis[:,1], gammabis[:,0])
    return [im, pathplot]


def main(filename='morne_rouge.asc', W=101, H=101, N=30, gamma=None,
        eps_v=1, eps_h=0.25):
    """
    Launches the animation.
    """
    depth_map = cost.load_image(filename, W, H)
    if gamma is None:
        gamma = cost.prepare_gamma(np.array([50, 20]), np.array([50, 80]), N)
    J_record = cost.cost(depth_map, gamma, eps_v, eps_h)[1]

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(0, W-1), ylim=(0, H-1))
    pathplot = ax.plot([], [], '-x')[0]
    im = plt.imshow(depth_map, interpolation='nearest')

    anim = animation.FuncAnimation(fig, animate, frames=N,
            fargs=(depth_map, gamma, J_record, im, pathplot), interval=400, blit=False)
    plt.show()

if __name__ == '__main__':
    main()
