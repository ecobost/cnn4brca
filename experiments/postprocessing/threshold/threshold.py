# To generate the thresold experiments


import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# Get some random image and add some signal (barely)
#true = np.zeros((100,100));
#true[40:60, 40:60] = 1;
original = np.random.rand(100,100);
original[40:60, 40:60] += 0.2;
original = original/1.2;
plt.gray()
plt.imshow(original)
plt.show()

# Uncorrected threshold at p> 0.7
uncorrected = original*(original > 0.7);
plt.imshow(uncorrected);
plt.show();

# Create labeled image 
kernel = np.ones((3,3));
labeledImage, numberOfLabels = ndimage.label(uncorrected, structure = kernel);

# Delete clusters less than 5
extentCluster = uncorrected.copy();
counts = np.bincount(labeledImage.flatten())
for label in range(1, numberOfLabels):
	if (counts[label] < 10):
		extentCluster[labeledImage == label] = 0;
		
plt.imshow(extentCluster)
plt.show()

# Delete clusters whose integrated area is less than 1
TCEF = uncorrected - 0.7;
TCEF[TCEF < 0] = 0;
for label in range(1, numberOfLabels):
	if(np.sum(TCEF[labeledImage == label]) < 1):
		TCEF[labeledImage == label] = 0;

plt.imshow(TCEF)
plt.show()
	


