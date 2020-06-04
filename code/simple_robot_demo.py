from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
from sklearn import mixture
from IPython.display import display
from scipy.stats import multivariate_normal
import numpy as np
import math
import os
import GPy as GPy
from Environment import Environment
from path_generator import Dubins_EqualPath_Generator

#解决路径规划问题的不同方法

# x1min, x1max, x2min, x2max constrain the extent of the rectangular domain
#ranges = (-10, 10, -10, 10)
#world = Environment(ranges, NUM_PTS = 20, variance = 100.0, lengthscale = 3.0, visualize = True)

# Pick one of the path generation models
DEPG = Dubins_EqualPath_Generator(frontier_size=6, horizon_length=5, turning_radius=1, sample_step=0.5)
s = DEPG.get_path_set((10, 10, 0))
start = np.array((10, 10, 0)).reshape(1,3)
# Methods for calling all of the data
#返回6目标点
m = DEPG.get_frontier_points()
m = np.array(m)
#返回6个目标点对应的路径点
l = DEPG.get_sample_points()

# Plotting for convenience
fig, ax = plt.subplots(1, 1)
ax.axis('equal')
for key, val in l.items():
    f = np.array(val)
    plt.plot(f[:, 0], f[:, 1], 'b*')
    #plt.show()
plt.plot(start[:, 0], start[:, 1], 'ro')
plt.plot(m[:, 0], m[:, 1], 'r*')
plt.show()