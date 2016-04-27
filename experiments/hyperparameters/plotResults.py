# Written by: Erick Cobos T.
# Date: 26-Apr-2016
"""
Script to plot the results from a random sample hyperparmater search
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import griddata

# Values taken from the Spreadsheet
alpha = np.array([4.02E-006 ,1.19E-005, 1.79E-005, 1.91E-005, 2.64E-005, 3.85E-005, 3.86E-005, 1.05E-004, 1.48E-004, 4.36E-004, 4.85E-004, 1.15E-003, 1.96E-003, 2.09E-003, 4.06E-003, 1.60E-002, 1.67E-002, 1.87E-002, 3.20E-002, 3.60E-002, 4.03E-002, 4.16E-002, 4.77E-002, 1.18E-001, 1.67E-001, 1.95E-001, 2.09E-001, 2.34E-001, 2.83E-001, 5.85E-001])
lambda_ = np.array([1.98E+001, 4.20E+001, 1.13E+000, 2.01E+001, 3.43E-002, 3.82E-003, 1.05E+000, 2.21E+000, 1.27E-001, 1.15E+002, 3.50E-003, 4.08E-002, 6.55E+002, 1.89E-001, 1.72E+002, 6.73E+002, 5.35E-003, 1.36E-002, 2.43E+001, 3.01E+001, 1.39E+001, 4.42E-001, 3.62E+002, 1.59E-001, 2.37E+002, 3.16E-001, 9.64E+000, 2.25E-001, 3.74E-001, 1.82E-003])

# Example: random values
values = np.array([-1.067251, 0.345497, 2.303435, 0.369687, -0.299575, -0.879544, 0.572913, -2.053532, -0.741751, -0.129250, 0.269780, 1.739360, 1.072975, 0.349465, -0.357556, -1.049498, -0.039290, 0.643444, -1.968745, 0.608591, -0.519992, -0.754841, -0.078200, 0.244976, 0.185562, 0.451222, -0.488289, -1.255517, -1.106825, -0.426704])

# Validation against alpha
order = np.argsort(alpha)
plt.plot(alpha[order], values[order])
plt.xlabel('Alpha')
plt.ylabel('IOU')
plt.show()

# Validation loss against lambda
order = np.argsort(lambda_)
plt.plot(lambda_[order], values[order])
plt.xlabel('Lambda')
plt.ylabel('IOU')
plt.show()

# Grid of alpha and lambda_
#xs, ys = np.meshgrid(np.linspace(alpha.min(), alpha.max()), np.linspace(lambda_.min(), lambda_.max()))
#xs, ys = np.meshgrid(alpha, lambda_) 
#xs, ys = np.meshgrid(alpha, lambda_[order[0:-5]]) # Discard 5 highest lambdas
xs, ys = np.meshgrid(np.logspace(-6, 0), np.logspace(-3,3))
zs = griddata((alpha, lambda_), values, (xs, ys), method = 'nearest')
plt.pcolor(xs, ys, zs, cmap = cm.jet)
plt.xlabel('Alpha')
plt.ylabel('Lambda')
plt.colorbar()
plt.show()

# Trisurphace of alpha and lambda
fig = plt.figure()
ax = fig.gca(projection='3d')

# Simple
surf = ax.plot_trisurf(alpha, lambda_, values, cmap=cm.jet, edgecolors='none')

# Slightly more complicated. Does not work!
#refiner = tri.UniformTriRefiner(tri.Triangulation(alpha, lambda_))
#new, new_z = refiner.refine_field(values, subdiv=4)
#surf = ax.plot_trisurf(new.x, new.y, new_z, cmap=cm.jet, edgecolors='none')

ax.set_xlabel('Alpha')
ax.set_ylabel('Lambda')
ax.set_zlabel('IOU')
fig.colorbar(surf)
plt.show()
