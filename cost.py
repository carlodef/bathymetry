import numpy as np

from PIL import Image


def load_image(filename, w, h):
    """
    Loads an image from an ascii file.

    Args:
        filename: path to the file
        w, h: image width and height

    Returns:
        numpy 2D array containing the image.
    """
    f = open(filename, "r")
    lines = f.readlines()
    assert(w*h == len(lines))
    out = np.zeros((w, h))
    for i in range(h):
        for j in range(w):
            l = lines[i*w + j]
            out[i, j] = l.split()[2] 
    return out


def dilate(S, r):
    """
    Computes the dilation of a subset of R^2 with a disk of given radius.

    Args:
        S: set of tuples of size 2
        r: disk radius

    Returns:
        dilated set
    """
    out = set() 
    for p in S:
        for i in range(-r, r+1):
            for j in range(-r, r+1):
                q = (p[0] + i, p[1] + j)
                out.add(q)
    return out 


def translate(S, t):
    """
    Computes the translation of a subset of R^2.

    Args:
        S: set
        t: tuple representing the vector of the translation

    Returns:
        translated set
    """
    out = set()
    for p in S:
        q = (p[0] + t[0], p[1] + t[1])
        out.add(q)
    return out


def cost(im, gamma, eps_v, eps_h):
    """
    Computes the cost associated to a given path and depth map.

    Args:
        im: water depth map
        gamma: list of tuples of length 2, representing the input path 
        eps_v:
        eps_h:
    """
    N = len(gamma)
    J_record = []
    J = set()
    J.add(gamma[0])
    for k in range(N-1):
        J_record.append(J)
        J = dilate(J, eps_v)
        q1 = gamma[k+1]
        q0 = gamma[k]
        t = (q1[0] - q0[0], q1[1] - q0[1])
        J = translate(J, t)
        J_temp = set()
        for p in J:
            if p[0] < 0 or p[0] > im.shape[1]-1:
                continue
            if p[1] < 0 or p[1] > im.shape[0]-1:
                continue
            if np.abs(im[p[0], p[1]] - im[q1[0], q1[1]]) < eps_h:
                J_temp.add(p)
        J = J_temp

    return len(J), J_record


def color_set(im, J):
    out = im.copy()
    for p in J:
        out[p[0], p[1]] = np.nan
    return out

def color_sets(im, J_record):
    out = im.copy()
    for J in J_record:
        out = color_set(out, J)
    return out


def prepare_gamma(A, B, N):
    """
    Returns a straight path.

    Args:
        A, B: endpoints
        N: number of samples

    Returns:
        list of tuplesof size 2. Each tuple represents a point.
    """
    gamma_0 = list()
    for k in range(N+1):
        p = A + float(k)/N * (B - A)
        p = np.round(p).astype(int)
        gamma_0.append((p[0], p[1]))
    return gamma_0


def main(N=12):
    """
    """
    depth_map = load_image('morne_rouge.asc', 101, 101)
    gamma = prepare_gamma(np.array([50, 20]), np.array([50, 80]), N)
    c, J_record = cost(depth_map, gamma, 1, 0.25)

    # sequence of images showing the evolution of the J sets
    for i in range(N):
        im = color_set(depth_map, J_record[i])
        im = Image.fromarray(im)
        im.save('im%02d.tif' % i)


if __name__ == '__main__':
    main()
