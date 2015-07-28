# Written by: Erick Cobos (a01184587@itesm.mx)
# Date: 27-Jul-2015
'''
Methods to generate small image patches from full mammograms. Justification for most decisions taken are provided in the thesis. Run as:
	python3 retrieval

Methods createMassPatches and createMCCPatches could also be used independently.

Configuration params
-----------------
_path: string
	Directory containing the mammograms. Default: .
_recursive: boolean (!Not yet working)
	Whether folders in the directory should also be explored. Default: False
_ext: string
	Image extension of the mammograms. Default: .jpg
_extOut: string
	Extension of the patches file. Default: .pch
_originalPixelSize: float
	Pixel size in mm of the mammograms. Default: 0.05
_patchSizeInPixels: int
	Desired size of the patch in pixels. Default: 127
_maxBlackAllowed: int
	Percentage of the patch which is allowed to be black. Anything over that
	will be deleted. Default: 0.25
_grayValues: float
	Number of intensity values the patch will have after constrast stretching. 
	Default: 65535

Notes
---------------
Developed in python 3.x. Needs changes to work in previous versions.
 
Uses PILLOW 3.0. Will not work with PIL.

Uses 64-bit floats to calculate the mean and variance statistics, 32-bit machines
may lose precision. Using images with depth 19 or more will cause lose of precision
when calculating the variance; 8-bit or 16-bit images are alright.

'''
_path = "."
_recursive = False
_ext = ".jpg"
_extOut = ".pch"
_originalPixelSize = 0.05
_patchSizeInPixels = 127
_maxBlackAllowed = 0.25
_grayValues = 65535

import glob
import os
import numpy as np
import scipy.misc
from PIL import Image


def createPatches(patchSizeInMm, stride, subfolder, path = _path, ext = _ext,
		extOut = _extOut, originalPixelSize = _originalPixelSize,
		patchSizeInPixels = _patchSizeInPixels,
		maxBlackAllowed = _maxBlackAllowed, grayValues = _grayValues):
	'''
	Creates smaller images (patches) by sliding a window over big images.
	
	Creates a subfolder to store the patches of every image in the path directory. Crops the image according to the patchSize and originalPixelSize. Assigns its respective labels and resizes it to pacthSizeInPixels pixels. Accumulates some overall statistics (mean and var) for feature normalization (later).
	
	Parameters
	-------
	patchSizeInMm: int
		Desired size of the patch in milimeters.
	stride: int
		Amount of space milimeters that the window is moved at each step.
	subfolder: string
		Name of the subfolder where the patches are going to be stored.
	path: string
		Directory containing the mammograms.
	ext: string
		Image extension of the mammograms.
	extOut: string
		Extension of the patches file.
	originalPixelSize: float
		Pixel size in mm of the mammograms.
	patchSizeInPixels: int
		Desired size of the patch in pixels.
	maxBlackAllowed: int
		Percentage of the patch which is allowed to be black.
	grayValues: float
		Number of intensity values the patch will have after constrast 
		stretching.

	'''
	# Store current working directory to come back here  after function exits
	oldPath = os.getcwd()

	# Create subfolder
	os.chdir(path)
	os.makedirs(subfolder, exist_ok = True)

	# Calculate the number of pixels that represent the patch and stride in the 		# original big images.
	bigPatchSize = round(patchSizeInMm / originalPixelSize)
	bigStride = round(stride / originalPixelSize)

	# Initialize the mean and var statistics
	overallMean = np.zeros( (patchSizeInPixels, patchSizeInPixels) )
	overallVar = np.zeros( (patchSizeInPixels, patchSizeInPixels) )

	# Initialize other statistics
	numberOfPatches = 0
	numberOfImages = 0

	# Recursivity could be added in here: a for over all folders (os.walk) plus chdir to the new directory. To decide: should a new subfolder be created or everything added into the big subfolder (change accordingly), what about the statistics in each subfolder or outside everything (if there is many folders). If various subfolders are going to be used we could just call it recursively (as a depth-frist tree traversal) before the subfolder creation.

	# For each file with extension ext
	for imageFile in glob.glob("*" + ext):

		# Retrieve name and extension
		fileName, ext = os.path.splitext(imageFile)

		# Read image as numpy array (float32 type) 
		image = scipy.misc.imread(imageFile, flatten = True)

		# Create copy of image with black spaces (0-4) as True, else False
		blackImage = (image <= 4)

		# Initialize lists for patches and labels. I use a list because
		# append works faster. Later it is transformed to ndarray.
		imagePatches = []
		imageLabels = []

		# Slide window across x (rows) and y (columns) of image. 
		M, N = image.shape
		for x in range(0, M - bigPatchSize, bigStride):
			for y in range(0, N - bigPatchSize, bigStride):

				# If patch is more than 25% black, skip it
				blackPatch = blackImage[x : x + bigPatchSize,
							y : y + bigPatchSize]
				if(blackPatch.mean() > maxBlackAllowed):
					continue
				
				# Crop patch
				patch = image[x : x + bigPatchSize,
						y : y + bigPatchSize]

				# Background reduction and contrast stretching
				patch = adjustContrast(patch, grayValues)

#TODO Write this method		# Assign label
				label = np.array([0, 0, 0]);

				# Reduce.  If mode is not provided to toimage it is
				# set to 'L' (uint8 grayscale).
				size = (patchSizeInPixels, patchSizeInPixels)
				patchIm = scipy.misc.toimage(patch, mode = 'F')
				patchIm = patchIm.resize(size, Image.LANCZOS)
				patch = np.array(patchIm)

				# Append
				imagePatches.append(patch)
				imageLabels.append(label)
		
		# Transform lists into numpy arrays. Patches are stored as rows.
		# Patch x can be accesed as imagePatches[x,:,:], same with labels.
		imagePatches = np.array(imagePatches)
		imageLabels = np.array(imageLabels) 

		# Update the number of patches and number of images.
		oldNumberOfPatches = numberOfPatches
		numberOfPatches += imagePatches.shape[0]
		numberOfImages += 1
		
		# Update the overall mean as a running average.
		tmpSum = np.sum(imagePatches, axis = 0, dtype = np.float64)
		overallMean = (overallMean * (oldNumberOfPatches / numberOfPatches) 					+ tmpSum / numberOfPatches)

		# Update the overall variance as a running average. Not exact because
		# the mean is only an approximation up to this point.
		tmpSum = np.sum( (imagePatches - overallMean) ** 2, axis = 0,
				dtype = np.float64)
		overallVar = (overallVar * (oldNumberOfPatches / numberOfPatches) 
				+ tmpSum / numberOfPatches)

#TODO Still don't like the accuracy. Compare with Welford method (https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance). Use 4 images and see which one gives a more exact approx to the real var.


		# Save image patches and labels to subfolder.
		np.save(subfolder + "/" + fileName + "_pch", imagePatches)
		np.save(subfolder + "/" + fileName + ".lbl", imageLabels)
#TODO: extOut, maybe let it as .npy or extOut.npy. Waht about extension for labels. add _labelSuffix = "L". 
# mammogram4.pch.npy or mammogram4_pch.npy or mammogram4.npy
# if i say glob(.npy) it wil be hard to get both npy and L, it may be that some pictures are named_L. Maybe it is better to put another extension because it is easier to do pattern matchiong.
	
		# Print a signal (.) to show that it is still running
		print('.', end = "", flush = True)
	
	# Recursivity: END here.

	# Save statistics(mean and var)
	np.save("overallMean", overallMean)
	np.save("overallVar", overallVar)
	
	# Print some statistics
	print("\nPatches were succesfully generated!")
	print("Processed images:", numberOfImages)
	print("Total number of patches:", numberOfPatches)
	print("Average mean of patches:")
	print(overallMean)
	print("Average variance of patches:")
	print(overallVar)

	# Change back to old directory
	os.chdir(oldPath)



def adjustContrast(array, newMax):
	"""
	Performs background reduction by substracting the mean of the image and
	stretches the contrast (linear normalization) to cover the entire range
	of pixel values 0-newMax (newMax = 255 for an 8-bit image, newMax = 6555
	for a 16-bit image, etc.)

	Parameters
	----------------
	array: ndarray
		A numpy 2-dimensional array containing the image
	newMax: float
		Maximum intensity value to consider for the normalization. Could
		also be 1.

	Returns
	----------------
	result: ndarray
		A numpy array (dtype = 'float64') containing the processed image.

	Notes
	---------------
	Works only for grayscale images.

	"""
	# To produce float64 results, comment it to preserve input format.
	# result = array.astype('float')
	
	# To avoid modifying the input array
	result = array.copy()
	
	# Background reduction
	image_mean = result.mean()
	result[result < image_mean] = image_mean;

	# Contrast stretching
	image_min = result.min()
	image_max = result.max()	
	result -= image_min
	result *= (newMax / (image_max-image_min))
	
	return result



# Utilities to create mass and mcc patches
def createMassPatches(patchSizeInMm = 20, stride = 2, subfolder = "massPatches"):
	'''
	Calls createPatches with the default values for mass. It can be simply called as createMassPatches().

	Parameters are documented in help(createPatches).
	'''
	createPatches(patchSizeInMm, stride, subfolder)


def createMCCPatches(patchSizeInMm = 10, stride = 1, subfolder = "mccPatches"):
	'''
	Calls createPatches with the default values for microcalcification clusters. It can be simply called as createMCCPatches().

	Parameters are documented in help(createPatches).
	'''
	createPatches(patchSizeInMm, stride, subfolder)



# To run the module as a script
if __name__ == "__main__":
	createMassPatches()
	createMCCPatches()

# friday 6:55, 2:45
# Too many patches. May need slightly bigger stride (3 mm will probably do). Need from 500-1000 patches per mamogram. Maybe it's better to generate more than 1000 (small stride) and drop some normal ones to upsample the lessions (Nop, way too many images of the same lession, just translated, overfitting). Maybe put 2mm for microcalc, too.
# Maybe use uint8/uint16 format instead of float32 for weight.
# 2234 127x127 patches in uint8 36MB, float32 144MB. Exactly 4 times more.
'''
Tests: (already performed)
	Does not delete or crash if subfolder already exists. It ignores it.
	Overwrites files if they already exist.
	It does deletes patches with more than 25% black.
	It prints format32

Tests (to do):
	Mean and var work. Errors in the 1*e-5 ratio. 
	check other online var and mean.
	Works with various images
'''
