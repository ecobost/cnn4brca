# Written by: Erick Cobos T
# Date: November-2016
""" 
Plots results for every fold and every experiment.
"""
import numpy as np
import matplotlib.pyplot as plt

# Set some params
NUMBER_OF_POINTS = 102 # number of points used for the interpolation
UPPER_LIMIT = 16 # the highest FP/image(highest value in x axis)

# Load sensitivity and FP/image results
sensitivity = np.loadtxt('sensitivity.csv', delimiter=',') # 25 folds x 100 points
fp_image = np.loadtxt('fp_image.csv', delimiter=',') # 25 folds x 100 points

# Sample the sensitivity between zero and the desired upper limit
froc = np.empty((25, NUMBER_OF_POINTS))
desired_fps = np.linspace(0, UPPER_LIMIT, NUMBER_OF_POINTS)
for i in range(25):
	froc[i] = np.interp(desired_fps, fp_image[i], sensitivity[i])

# Calculate average FROC per experiment
mean_froc = froc.reshape(-1, 5, NUMBER_OF_POINTS).mean(axis=1)

# Calculate standard error of the mean (sigma_sample/sqrt(n))
std_froc = froc.reshape(-1,5,NUMBER_OF_POINTS).std(axis=1)/np.sqrt(5)

# Plotting params.
colors = ['magenta', 'blue', 'green', 'yellow', 'red'] # color for each experiment
markers = ['+', 'x', '|', '.', '2'] # marker style for each fold
labels = ['Experiment 1.1', 'Experiment 1.2', 'Experiment 1.3', 'Experiment 2', 'Experiment 3'] #labels for each experiment
		   
# Plot all folds
for i in range(25):
	plt.plot(desired_fps, froc[i], color=colors[i//5], marker=markers[i%5], linestyle='dashed', linewidth=0.8, alpha=0.3)
			 
# Plot a shaded standard error area
for i in range(5):
	plt.fill_between(desired_fps, mean_froc[i] - std_froc[i], mean_froc[i] + std_froc[i], color=colors[i], alpha=0.3)
					 
# Plot the averages FROC
for i in range(5):
	plt.plot(desired_fps, mean_froc[i], color=colors[i], label=labels[i], linewidth=2.5)

# Add labels
plt.legend(loc='lower right')
plt.xlabel('FP/image')
plt.ylabel('Sensitivity')


######### IOUs
iou = np.loadtxt('iou.csv', delimiter=',') # 25 folds x 100 points
print(iou.max(axis=1)) # IOU with the best possible threshold for every fold

# Plot all IOUS
plt.figure()
for i in range(25):
	plt.plot(np.linspace(0.01, 0.99, 100), iou[i], color=colors[i//5], marker=markers[i%5])
	
# Add labels
for i in range(5):
	plt.plot([], [], color=colors[i], label=labels[i]) # empty lines for legend
plt.legend(loc='upper left')
plt.xlabel('Threshold')
plt.ylabel('IOU')

# Show 
plt.show()
