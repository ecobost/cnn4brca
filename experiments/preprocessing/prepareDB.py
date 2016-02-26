# Written by: Erick M. Cobos T.
# Date: 25-Feb-2015
 
# Script to enhance (global background reduction + normalization), downsample and
# augment the dataset.

# Load modules
import csv
from PIL import Image, ImageStat, ImageOps

# Set some parameters
input_filename = "bcdr_d01_img.csv"	# Name of input .csv
output_filename = "validation.csv"	# Produced .csv (stores filenames of images)
downsampling_factor = (112 * 0.007)/2 	# Scaling factor for an image with 0.007 cm
					# per pixel (spatial resolution) to get a
					# 2x2 cm area in 112 x 112 pixels
label_suffix = "_label"			# Suffix of label images
small_label_suffix = "_smallLabel"	# Suffix for the smaller labels
network_subsampling = 16		# How much will the network subsample the
					# image. Ours outputs an image 16x smaller

# Reading bcdr_d01_img.csv to get the image names
with open(input_filename, newline = '') as csv_file, \
     open(output_filename, 'w') as output_file:
	reader = csv.reader(csv_file, skipinitialspace = True)
	next(reader) # Skip header

	# For each mammogram
	for row in reader:
		filename = row[3]
		basename = filename[:-4]

		# Load the mammogram and label
		mammogram = Image.open(filename);
		label = Image.open(basename + "_mask.png");

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
		# dimension is divisible by 16 (our network subsampling factor)
		bbox = label.getbbox() 	# bounding-box 4-tuple: upper-left x,
					# upper-left y, lower-right x, lower-right y
		bbox = list(bbox)
		new_width = abs(bbox[0] - bbox[2])
		new_height = abs(bbox[1] - bbox[3])
		lacked_width = network_subsampling - \
						(new_width % network_subsampling)
		lacked_height = network_subsampling - \
						(new_height % network_subsampling)

		if bbox[0] - lacked_width >= 0:
			bbox[0] -= lacked_width
		elif bbox[2] + lacked_width <= label.width:
			bbox[2] += lacked_width
		else:
			# Just drop some valid columns
			bbox[0] += (network_subsampling - lacked_width)

		if bbox[1] - lacked_height >= 0:
			bbox[1] -= lacked_height
		elif bbox[3] + lacked_height <= label.height:
			bbox[3] += lacked_height
		else:
			# Just drop some valid rows
			bbox[1] += (network_subsampling - lacked_height)

		mammogram = mammogram.crop(bbox)
		label = label.crop(bbox)

		# Create smaller label (downsampled by 16)
		small_width = round(label.width/network_subsampling)
		small_height = round(label.height/network_subsampling)
		small_label = label.resize((small_width, small_height),
								Image.NEAREST)

		# Augment images (verbose)
		mammogram_v1 = mammogram
		label_v1 = label
		small_label_v1 = small_label

#TODO: Fix that rotating it augments the dimension of the image, even the 180 one
# and even in small images

		mammogram_v2 = mammogram_v1.rotate(90, expand = True)
		label_v2 = label_v1.rotate(90, expand = True)
		small_label_v2 = small_label_v1.rotate(90, expand = True)

		mammogram_v3 = mammogram_v1.rotate(180, expand = True)
		label_v3 = label_v1.rotate(180, expand = True)
		small_label_v3 = small_label_v1.rotate(180, expand = True)

		mammogram_v4 = mammogram_v1.rotate(270, expand = True)
		label_v4 = label_v1.rotate(270, expand = True)
		small_label_v4 = small_label_v1.rotate(270, expand = True)

		mammogram_v5 = ImageOps.mirror(mammogram)
		label_v5 = ImageOps.mirror(label)
		small_label_v5 = ImageOps.mirror(small_label)

		mammogram_v6 = mammogram_v5.rotate(90, expand = True)
		label_v6 = label_v5.rotate(90, expand = True)
		small_label_v6 = small_label_v5.rotate(90, expand = True)

		mammogram_v7 = mammogram_v5.rotate(180, expand = True)
		label_v7 = label_v5.rotate(180, expand = True)
		small_label_v7 = small_label_v5.rotate(180, expand = True)

		mammogram_v8 = mammogram_v5.rotate(270, expand = True)
		label_v8 = label_v5.rotate(270, expand = True)
		small_label_v8 = small_label_v5.rotate(270, expand = True)

		# Save images (verbose)
		mammogram_v1.save(basename + "_v1.png")
		label_v1.save(basename + label_suffix + "_v1.png")
		small_label_v1.save(basename + small_label_suffix + "_v1.png")

		mammogram_v2.save(basename + "_v2.png")
		label_v2.save(basename + label_suffix + "_v2.png")
		small_label_v2.save(basename + small_label_suffix + "_v2.png")

		mammogram_v3.save(basename + "_v3.png")
		label_v3.save(basename + label_suffix + "_v3.png")
		small_label_v3.save(basename + small_label_suffix + "_v3.png")

		mammogram_v4.save(basename + "_v4.png")
		label_v4.save(basename + label_suffix + "_v4.png")
		small_label_v4.save(basename + small_label_suffix + "_v4.png")

		mammogram_v5.save(basename + "_v5.png")
		label_v5.save(basename + label_suffix + "_v5.png")
		small_label_v5.save(basename + small_label_suffix + "_v5.png")

		mammogram_v6.save(basename + "_v6.png")
		label_v6.save(basename + label_suffix + "_v6.png")
		small_label_v6.save(basename + small_label_suffix + "_v6.png")

		mammogram_v7.save(basename + "_v7.png")
		label_v7.save(basename + label_suffix + "_v7.png")
		small_label_v7.save(basename + small_label_suffix + "_v7.png")

		mammogram_v8.save(basename + "_v8.png")
		label_v8.save(basename + label_suffix + "_v8.png")
		small_label_v8.save(basename + small_label_suffix + "_v8.png")

		# Save filenames to csv
		output_file.write(basename + "_v1.png" + "," + 
				basename + label_suffix + "_v1.png" + "," +
				basename + small_label_suffix + "_v1.png" + "\n")

		output_file.write(basename + "_v2.png" + "," +
				basename + label_suffix + "_v2.png" + "," +
				basename + small_label_suffix + "_v2.png" + "\n")

		output_file.write(basename + "_v3.png" + "," +
				basename + label_suffix + "_v3.png" + "," +
				basename + small_label_suffix + "_v3.png" + "\n")

		output_file.write(basename + "_v4.png" + "," +
				basename + label_suffix + "_v4.png" + "," +
				basename + small_label_suffix + "_v4.png" + "\n")

		output_file.write(basename + "_v5.png" + "," +
				basename + label_suffix + "_v5.png" + "," +
				basename + small_label_suffix + "_v5.png" + "\n")

		output_file.write(basename + "_v6.png" + "," +
				basename + label_suffix + "_v6.png" + "," +
				basename + small_label_suffix + "_v6.png" + "\n")

		output_file.write(basename + "_v7.png" + "," +
				basename + label_suffix + "_v7.png" + "," +
				basename + small_label_suffix + "_v7.png" + "\n")

		output_file.write(basename + "_v8.png" + "," +
				basename + label_suffix + "_v8.png" + "," +
				basename + small_label_suffix + "_v8.png" + "\n")
