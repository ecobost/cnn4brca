# Written by: Erick Cobos (a01184587@itesm.mx)
# Date: 27-Jul-2015
'''
Methods to generate small image patches from full mammograms. Justification for most decisions taken are provided in the thesis. Run as:
	python3 retrieval.py

Methods createMassPatches and createMCCPatches could also be used independently.

Configuration params
-----------------
_path: string
	Directory containing the mammograms. Default: .
_recursive: boolean (!Not working)
	Whether folders in the directory should also be explored. Default: False
_ext: string
	Image extension of the mammograms. Default: .jpg
_patchSuffix: string
	String to attach at the end of the patches file name. Default: _pch
_labelSuffix: string
	String to attach at the end of the labels file name. Default: _lbl
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
when calculating the variance; 8-bit or 16-bit images are alright. Variance is not
a very good approximation; enough for feature scaling, though.

Using very small strides, thus producing many patches per image (over 3K) may cause
memory bottlenecks

Tests (performed)
----------------
	Does not delete or crash if subfolder already exists. It ignores it.
	It works if there is no images.
	Overwrites files if they already exist.
	It deletes patches with more than 25% black.
	It prints float32 images, float64 stats
	Mean is quite good, variance is not so. Tried Welford's method, no better.
	Works with various images

'''
_path = "."
_recursive = False
_ext = ".jpg"
_patchSuffix = "_pch"
_labelSuffix = "_lbl"
_originalPixelSize = 0.05
_patchSizeInPixels = 127
_maxBlackAllowed = 0.25
_grayValues = 65535

import glob
import os
import numpy as np
import scipy.misc
from PIL import Image


def createPatches(patchSizeInMm, stride, suffix, path = _path, ext = _ext,
		patchSuffix = _patchSuffix, labelSuffix = _labelSuffix,
		originalPixelSize = _originalPixelSize, patchSizeInPixels =
		_patchSizeInPixels, maxBlackAllowed = _maxBlackAllowed,
		grayValues = _grayValues):
	'''
	Creates smaller images (patches) by sliding a window over big images.
	
	Creates a subfolder to store the patches of every image in the path directory. Crops the image according to the patchSize and originalPixelSize. Assigns its respective labels and resizes it to pacthSizeInPixels pixels. Accumulates some overall statistics (mean and var) for feature normalization (later). Saves the patches and labels per image and overall statistics on disk.
	
	Parameters
	-------
	patchSizeInMm: int
		Desired size of the patch in milimeters.
	stride: int
		Amount of space (milimeters) that the window is moved at each step..
	suffix: string
		Suffix for the folder where the patches are going to be stored and 
		the mean and variance files.
	path: string
		Directory containing the mammograms.
	ext: string
		Image extension of the mammograms.
	patchSuffix: string
		String to attach at the end of the patches file name.
	labelSuffix: string
		String to attach at the end of the labels file name.
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
	folderName = "patches" + suffix
	os.makedirs(folderName, exist_ok = True)

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
		overallMean = (overallMean * (oldNumberOfPatches / numberOfPatches)
				+ tmpSum / numberOfPatches)

		# Update the overall variance as a running average. Not exact because
		# the mean is only an approximation up to this point.
		tmpSum = np.sum( (imagePatches - overallMean) ** 2, axis = 0,
				dtype = np.float64)
		overallVar = (overallVar * (oldNumberOfPatches / numberOfPatches)
				+ tmpSum / numberOfPatches)

		# Save image patches and labels to subfolder.
		np.save(folderName + "/" + fileName + patchSuffix, imagePatches)
		np.save(folderName + "/" + fileName + labelSuffix, imageLabels)
	
		# Print a signal (.) to show that it is still running
		print('.', end = "", flush = True)
	
	# Recursivity: END here.

	# Save statistics(mean and var)
	np.save("mean" + suffix, overallMean)
	np.save("var" + suffix, overallVar)
	
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
	
	# Background reduction.
	image_mean = result.mean()
	result[result < image_mean] = image_mean;

	# Contrast stretching
	image_min = result.min()
	image_max = result.max()	
	result -= image_min
	result *= (newMax / (image_max-image_min))
	
	return result



# Utilities to create mass and mcc patches
def createMassPatches(patchSizeInMm = 20, stride = 3, suffix = "_mass"):
	'''
	Calls createPatches with the default values for mass. It can be simply called as createMassPatches().

	Parameters are documented in help(createPatches).
	'''
	print("Generating patches for masses (20 mm, stride 3 mm)")
	createPatches(patchSizeInMm, stride, suffix)
	print("-------------------------------")


def createMCCPatches(patchSizeInMm = 10, stride = 3, suffix = "_mcc"):
	'''
	Calls createPatches with the default values for microcalcification clusters. It can be simply called as createMCCPatches().

	Parameters are documented in help(createPatches).
	'''
	print("Generating patches for microcalcifications (10 mm, stride 3 mm)")
	createPatches(patchSizeInMm, stride, suffix)
	print("-------------------------------")



# To run the module as a script
if __name__ == "__main__":
	createMassPatches()
	createMCCPatches()

# Maybe use uint8/uint16 format instead of float32 for weight.
# Each patch is 66KB. 1K is 65MB. 1M is 64GB.
