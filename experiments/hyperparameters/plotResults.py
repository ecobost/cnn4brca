# Written by: Erick Cobos T.
# Date: 26-Apr-2016
"""
Commands to plot results from a random sample hyperparmater search.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import griddata

# Hyperparameter search 1
# Values taken from the Spreadsheet hyperparameter_search.odt
alpha = np.array([4.02E-006 ,1.19E-005, 1.79E-005, 1.91E-005, 2.64E-005, 3.85E-005, 3.86E-005, 1.05E-004, 1.48E-004, 4.36E-004, 4.85E-004, 1.15E-003, 1.96E-003, 2.09E-003, 4.06E-003, 1.60E-002, 1.67E-002, 1.87E-002, 3.20E-002, 3.60E-002, 4.03E-002, 4.16E-002, 4.77E-002, 1.18E-001, 1.67E-001, 1.95E-001, 2.09E-001, 2.34E-001, 2.83E-001, 5.85E-001])
lambda_ = np.array([1.98E+001, 4.20E+001, 1.13E+000, 2.01E+001, 3.43E-002, 3.82E-003, 1.05E+000, 2.21E+000, 1.27E-001, 1.15E+002, 3.50E-003, 4.08E-002, 6.55E+002, 1.89E-001, 1.72E+002, 6.73E+002, 5.35E-003, 1.36E-002, 2.43E+001, 3.01E+001, 1.39E+001, 4.42E-001, 3.62E+002, 1.59E-001, 2.37E+002, 3.16E-001, 9.64E+000, 2.25E-001, 3.74E+001, 1.82E+002])
iou = np.array([0.015, 0.015, 0.073, 0.015, 0.13, 0.17, 0.019, 0.015, 0.085, 0.015, 0.145, 0.015, 0.015, 0.015, 0.015, 0.015, 0, 0, 0.015, 0.015, 0, 0.001, 0.015, 0, 0, 0.017, 0, 0, 0, 0])
# val_logistic_loss (DIVERGENT were topped to 2)
#values = np.array([0.4034, 0.5027, 0.2258, 0.5268, 0.2381, 0.2041, 0.1833, 0.1464, 0.1078, 0.2323, 0.0826, 0.0864, 0.0819, 0.0846, 0.0787, 0.0785, 2, 2, 0.0786, 0.0785, 2, 2, 0.0795, 2, 2, 2, 2, 2, 2, 2])

# Hyperparameter search 2
alpha = np.array([1.04E-006, 1.11E-006, 1.42E-006, 1.54E-006, 1.82E-006, 2.15E-006, 2.22E-006, 3.15E-006, 3.57E-006, 7.23E-006, 1.03E-005, 1.25E-005, 1.95E-005, 5.75E-005, 7.02E-005])
lambda_ = np.array([3.01E-003, 3.39E-004, 2.15E-003, 2.18E-003, 3.77E-002, 1.27E-002, 5.42E-004, 4.32E-004, 3.88E-002, 5.01E-003, 7.40E-005, 1.82E-005, 3.29E-002, 4.44E-003, 1.16E-005])
iou = np.array([0.124, 0.099, 0.101, 0.108, 0.117, 0.114, 0.124, 0.119, 0.12, 0.118, 0.117, 0.117, 0.077, 0.084, 0.115])

# Hyperparameter search 4 (First 17 only)
alpha = np.array([2.37E-006, 2.93E-006, 3.12E-006, 4.64E-006, 1.12E-005, 2.11E-005, 2.13E-005, 5.92E-005, 1.49E-004, 3.23E-004, 4.52E-004, 5.44E-004, 7.01E-004, 7.13E-004, 1.69E-003, 4.92E-003, 9.73E-003])
lambda_ = np.array([1.53E-004, 6.13E-001, 8.34E-004, 1.46E-003, 2.14E-003, 2.04E-003, 2.23E+000, 3.64E-003, 3.85E-001, 4.67E-002, 1.63E-001, 1.89E-004, 1.14E+000, 6.57E-003, 4.58E-001, 2.99E-002, 1.35E+000])
iou = np.array([0.262, 0.25, 0.266, 0.319, 0.296, 0.312, 0.015, 0.221, 0.015, 0.015, 0.017, 0.285, 0.015, 0.116, 0.015, 0.015, 0.015])

# Hyperparameter search 5 (6 )
alpha = np.array([3.56E-006, 1.17E-005, 1.31E-005, 1.63E-005, 1.82E-005, 2.01E-005, 2.27E-005, 3.36E-005, 3.80E-005, 4.08E-005, 4.18E-005, 6.56E-005, 1.20E-004, 1.43E-004, 2.59E-004, 3.79E-004, 3.96E-004, 1.63E-003, 1.65E-003, 1.84E-003])
lambda_ = np.array([3.04E-005, 7.39E-004, 5.51E-005, 7.58E-005, 5.05E-005, 1.23E-003, 7.95E-004, 5.29E-004, 4.81E-004, 1.02E-003, 8.48E-004, 3.61E-003, 4.54E-002, 2.88E-004, 1.44E-004, 2.98E-003, 2.79E-002, 2.80E-005, 1.02E-002, 4.50E-003])
iou = np.array([0.122, 0.128, 0.129, 0.129, 0.127, 0.126, 0.124, 0.125, 0.127, 0.128, 0.123, 0.122, 0.015, 0.136, 0.061, 0.12, 0.015, 0.068, 0.015, 0.127])


# IOU against alpha
order = np.argsort(alpha)
plt.plot(alpha[order], iou[order], 'k-o')
plt.xlabel('Alpha')
plt.ylabel('IOU')
plt.show()

# IOU against lambda
order = np.argsort(lambda_)
plt.plot(lambda_[order], iou[order], 'k-o')
plt.xlabel('Lambda')
plt.ylabel('IOU')
plt.show()

# Grid of alpha and lambda_ in x,y; IOU in z
#xs, ys = np.meshgrid(np.linspace(alpha.min(), alpha.max()), np.linspace(lambda_.min(), lambda_.max()))
#xs, ys = np.meshgrid(alpha, lambda_) 
#xs, ys = np.meshgrid(alpha, lambda_[order[0:-5]]) # Discard 5 highest lambdas
xs, ys = np.meshgrid(np.logspace(-6, -1), np.logspace(-4, 1))
zs = griddata((alpha, lambda_), iou, (xs, ys), method = 'nearest')
plt.pcolor(xs, ys, zs, cmap = cm.gray)
plt.xlabel('Alpha')
plt.ylabel('Lambda')
plt.colorbar()
plt.show()

# Trisurphace of alpha and lambda
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_trisurf(alpha, lambda_, iou, cmap=cm.gray, edgecolors='none')
ax.set_xlabel('Alpha')
ax.set_ylabel('Lambda')
ax.set_zlabel('IOU')
fig.colorbar(surf)
plt.show()
