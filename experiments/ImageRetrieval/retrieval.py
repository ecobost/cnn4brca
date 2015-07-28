# Written by: Erick Cobos (a01184587@itesm.mx)
# Date: 27-Jul-2015
'''
Methods to generate small image patches from full mammograms. Justification for most decisions taken are provided in the thesis. Run as:
	python3 retrieval
Methods could also be used independently.

Configuration params
-------
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
	Default: 255

'''
_path = "."
_recursive = False
_ext = ".jpg"
_extOut = ".pch"
_originalPixelSize = 0.05
_patchSizeInPixels = 127
_maxBlackAllowed = 0.25
_grayValues = 255

import glob
import Image from PIL
import numpy as np
import os
import scipy.misc



def createPatches(patchSizeInMm, stride, subfolder, path = _path, ext = _ext,
		extOut = _extOut, originalPixelSize = _originalPixelSize,
		patchSizeInPixels = _patchSizeInPixels,
		maxBlackAllowed = _maxBlackAllowed, grayValues = _grayValues):
	'''
	Creates smaller images (patches) by sliding a window over big images.
	
	Creates a subfolder to store the patches of every image in the path directory. Crops the image according to the patchSize and originalPixelSize. Assigns its respective labels and resizes it to pacthSizeInPixels pixels. Accumulates some overall statistics (mean and std) for feature normalization (later).
	
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
	os.mkdir(subfolder)

	# Calculate the number of pixels that represent the patch and stride in
	# the original big images.
	bigPatchSize = round(patchSizeInMm / originalPixelSize)
	bigStride = round(stride / originalPixelSize)

	# Initialize the mean and std statistics
	overallMean = np.zeros( (patchSizeInPixels, patchSizeInPixels) )
	overallStd = np.zeros( (patchSizeInPixels, patchSizeInPixels) )

	# Recursivity could be added in here: a for over all folders (os.walk) plus chdir to the new directory. To decide: should a new subfolder be created or everything added into the big subfolder (change accordingly), what about the statistics in each subfolder or outside everything (if there is many folders). If various subfolders are going to be used we could just call it recursively (as a depth-frist tree traversal) before the subfolder creation.

	# For each file with extension ext
	for imageFile in glob.glob("*" + ext):

		# Retrieve name and extension
		fileName, ext = os.path.splitext(imageFile)

		# Read image as numpy array (float32 type) 
		image = scipy.misc.imread(imageFile, flatten = True)

		# Create copy of image with black spaces (0-4) as True, else False
		blackImage = (image <= 4)

		# Initialize 3-d volume where patches are going to be contained
#TODO: How to initialize. Don't know the final size(amount of strides could be calculated but black spaces get skipped)		imagePatches = 

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
	
#TODO Write this method		# Assign label (?)

				# Reduce
				size = (patchSizeInPixels, patchSizeInPixels)
				patch = patch.resize(size, Image.LANCZOS)
				patch = np.array(patch)

				# Append (patch, label) per image on 3-d matrix
#TODO From here on		How to initialize. use append or insert (both inefficient). Lists are more efficient. Maybe a list of patches. calculating mean is harder though but maybe i can convert back to a 3-d array. How inefficient is this though

		#Accumulate the mean and std of patch values in image so that we can normalize per feature on entire database. Runing average.

		#save 3-d .pch on disk.
		#im.save(fileName + extOut, "JPEG")
	
		# Print a signal (.) to show that it is still running
		print('.', end = "", flush = True)
	
	#Save the mean and std of all images on disk (for later normalization of test and entire mammogram).

	# Change back to old directory
	os.chdir(oldPath)



def adjustContrast(array, newMax):
	"""
	Performs background reduction by substracting the mean of the image and
	stretches the contrast (linear normalization) to cover the entire range
	of pixel values 0-newMax (newMax = 255 for an 8-bit image).

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

#friday 1:40
# Too many pacthes. May need slightly bigger stride (3 mm will probably do). Need from 500-1000 patches per mamogram. Maybe it's better to generate more than 1000 (small stride) and drop some normal ones to upsample the lessions (Nop, way too many images of the same lession, just translated, overfitting). Maybe put 2mm for microcalc, too.
