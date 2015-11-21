# Written by: Erick Cobos (a01184587@itesm.mx)
# Date: 25-Jul-2016
def adjustContrast(array, newMax = 255):
	"""
	Performs background reduction by substracting the mean of the image and
	stretches the contrast (linear normalization) to cover the entire range
	of pixel values 0-newMax (newMax = 255 for an 8-bit image).

	Note: Works only for grayscale 2-d images.

	Parameters
	-----------------
	array: ndarray
		A numpy 2-dimensional array containing the image

	newMax: int, float
		Number of intensity values to have after constrast stretching.
		Could also be 1.

	Returns
	-----------------
	result: ndarray
		A numpy array (dtype = 'float64') containing the processed image.

	"""
	# To produce float64 results, comment it to maintain uint8 format.
	result = array.astype('float')
	
	# Background reduction
	#image_mean = result.mean()
	#result[result < image_mean] = image_mean;

	# Contrast stretching
	image_min = result.min()
	image_max = result.max()	
	result -= image_min
	result *= (newMax / (image_max-image_min))
	
	return result
