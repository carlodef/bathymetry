import numpy as np
from pylab import *
from matplotlib import pyplot as plt
from matplotlib import animation

import greedy_compute_gamma
import greedy_cost
import plot_sets

def main(A=(93, 4), B=(6, 85), plots_dir=None, filename="morne_rouge.asc", W=101, H=101):
    depth_map = greedy_cost.load_image(filename, W, H)
    gamma = greedy_compute_gamma.compute_gamma(depth_map,W,H,A,B)
    c, J_record = greedy_cost.cost(depth_map, gamma, 1, 0.25)
    print c
    plot_sets.main(filename, W, H, N=len(gamma), gamma=gamma,figures_path=plots_dir)
