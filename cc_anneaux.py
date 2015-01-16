import numpy as np
from pylab import *
from matplotlib import pyplot as plt
from matplotlib import animation

import plot_sets
import cc_compute_gamma
import cc_cost


def main(A=(93, 4), B=(6, 85), plots_dir=None, filename="morne_rouge.asc", W=101, H=101, alpha = .005):
    depth_map = cc_cost.load_image('morne_rouge.asc', W, H)
    gamma = cc_compute_gamma.compute_gamma(depth_map,W,H,A,B, alpha)
    c, J_record = cc_cost.cost(depth_map, gamma, 1, 0.25)
    plot_sets.main(filename, W, H, N=len(gamma), gamma=gamma,figures_path=plots_dir)
