# Written by: Erick M. Cobos T.
# Date: October-2016
""" Script to enhance (background reduction + normalization) and downsample 
images in the data set.

Note: At the end of the output csv file there may be an empty line. Remove if 
necessary

Example:
	python3 prepare_DB.py
"""
import csv
from PIL import Image, ImageStat, ImageOps

# Set some parameters
input_filename = 'bcdr_d01_img.csv'	# name of input csv
output_filename = 'data.csv' # name for the output csv (stores filenames)
downsampling_factor = (128 * 0.007)/2 # scaling factor for an image with 0.007cm
									  # per pixel (spatial resolution) to get a 
									  # 2x2 cm area in 128 x 128 pixels
image_suffix = '_image' # suffix for images
label_suffix = '_label' # suffix for label images
network_subsampling = 4 # amount of network subsampling: ours will subsample
						# images by a factor of 4 (2 pooling layers).

# Reading bcdr_d01_img.csv to get the image names
with open(input_filename, newline='') as input_file, \
	 open(output_filename, 'w') as output_file:
	reader = csv.reader(input_file, skipinitialspace=True)
	next(reader) # skip header

	# For each mammogram
	for row in reader:
		filename = row[3]
		basename = filename[:-4]

		# Load the mammogram and label
		mammogram = Image.open(filename);
		label = Image.open(basename + '_mask.png');

		# Enhance it (global background reduction + normalization)
		stat = ImageStat.Stat(mammogram, label)
		global_mean = stat.mean[0]
		mammogram = Image.eval(mammogram, lambda x:
					global_mean if x <= global_mean else x)
		mammogram = ImageOps.autocontrast(mammogram)

		# Downsample it
		new_width = round(mammogram.width * downsampling_factor)
		new_height = round(mammogram.height * downsampling_factor)
		mammogram = mammogram.resize((new_width, new_height), Image.LANCZOS)
		label = label.resize((new_width, new_height), Image.NEAREST)

		# Crop the image to delete unnnecesary background. Make sure every
		# dimension is divisible by our network subsampling factor
		bbox = label.getbbox() # bounding-box 4-tuple: upper-left x,
							   # upper-left y, lower-right x, lower-right y
		bbox = list(bbox)
		new_width = abs(bbox[0] - bbox[2])
		new_height = abs(bbox[1] - bbox[3])
		lacked_width = network_subsampling - (new_width % network_subsampling)
		lacked_height = network_subsampling - (new_height % network_subsampling)

		if bbox[0] - lacked_width >= 0:
			bbox[0] -= lacked_width
		elif bbox[2] + lacked_width <= label.width:
			bbox[2] += lacked_width
		else:
			# Drop some valid columns
			bbox[0] += (network_subsampling - lacked_width)

		if bbox[1] - lacked_height >= 0:
			bbox[1] -= lacked_height
		elif bbox[3] + lacked_height <= label.height:
			bbox[3] += lacked_height
		else:
			# Drop some valid rows
			bbox[1] += (network_subsampling - lacked_height)

		mammogram = mammogram.crop(bbox)
		label = label.crop(bbox)

		# Save images 
		mammogram.save(basename + image_suffix + ".png")
		label.save(basename + label_suffix + ".png")
		
		# Save filenames to csv
		output_file.write(basename + image_suffix + ".png" + "," + 
						  basename + label_suffix + ".png" + "\n")
