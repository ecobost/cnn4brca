# Written by: Erick Cobos T
# Date: 21-Sep-2016
""" Little script to plot the results of training networks with increasing amounts of data"""
import numpy as np
import matplotlib.pyplot as plt

num_examples = [0, 9, 18, 27, 36, 43]
iou = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

plt.plot(num_examples, iou, linewidth=2)
plt.xlabel('Number of data examples')
plt.ylabel('IOU')

