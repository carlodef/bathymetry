import numpy as np
import matplotlib.pylab as plt
import scipy.sparse

import cost
import plot_sets


def main(A=(93, 4), B=(6, 85), plots_dir=None, filename="morne_rouge.asc",
         w=101, h=101, n=8, eps_v=1, eps_h=0.5):
    """
    """
    depth_map = cost.load_image(filename, w, h)

    # compute the straight path with the same number of control points
    path = cost.prepare_gamma(np.array(A), np.array(B), n)
    c, J_record = cost.cost(depth_map, path, eps_v, eps_h)
    print "the baseline path is ", path
    print "the baseline cost is ", c

    # plot it 
    plot_sets.main(filename, w, h, N=len(path), gamma=path, eps_v=eps_v, eps_h=eps_h, figures_path=plots_dir)


if __name__ == '__main__':
    main()
