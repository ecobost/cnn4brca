# Written by: Erick Cobos T.
# Date: October 2016
""" Trains a convolutional network with training and validation sets.

	If defined, uses VAL_CSV_PATH as the validation set; otherwise, splits the 
	training set in training and validation.
	
	It uses all available CPUs and a single GPU (if available) in one machine, 
	i.e., it is not distributed.
	
	Example:
		python3 train_with_val_split.py
"""
import tensorflow as tf
import numpy as np
import random
import os
from utils import log, read_csv_info

# Import network definition
import model_v4 as model

# Set training parameters
TRAINING_STEPS = 163*8*5 # 163 mammograms (approx) * 8 augmentations * 5 epochs
LEARNING_RATE = 4e-5
LAMBDA = 4e-4
RESUME_TRAINING = False

# Set some path
DATA_DIR = "data" # folder with training data (images and labels)
MODEL_DIR = "run119" # folder to store model checkpoints and summary files
CSV_PATH = "data/training_1.csv" # path to csv file with image,label filenames
VAL_CSV_PATH = None # path to validation set. If undefined, split training set
NUM_VAL_PATIENTS = 10 # number of patients for validation set; used only if 
					  # val_csv is not provided

					  
def val_split(csv_path, num_val_patients, model_dir):
	""" Divides the data set into training and validation sets sampling patients
	at random.
	
	Args:
		csv_path: An string. Path to the csv with image and label filenames.
			Records are expected as 'image_filename,label_filename'
		num_val_patients: An integer. Number of patients for the validation set.
		model_dir: An string. Path to the directory to store new csvs with 
			training and validation info.
			
	Returns:
		training_image_filenames: A list of strings. Filenames for training 
			images.
		training_label_filenames: A list of strings. Filenames for training 
			labels.
		val_image_filenames: A list of strings. Filenames for validation images.
		val_label_filenames: A list of strings. Filenames for validation labels.
	"""
	# Read csv file
	with open(csv_path) as csv_file:
		lines = csv_file.read().splitlines()

	# Get patients at random
	val_patients = set()
	while len(val_patients) < num_val_patients:
		patient_name = random.choice(lines).split('/')[0]
		val_patients.add(patient_name)

	# Divide val and training set
	val_lines = [line for line in lines if line.split('/')[0] in val_patients]
	training_lines = [l for l in lines if l.split('/')[0] not in val_patients]
	
	# Write training and val csvs to disk
	with open(os.path.join(model_dir, 'val.csv'), 'w') as val_file:
		val_file.write('\n'.join(val_lines))
	with open(os.path.join(model_dir, 'training.csv'), 'w') as training_file:
		training_file.write('\n'.join(training_lines))
			
	# Generate lists of filenames
	training_image_filenames = [line.split(',')[0] for line in training_lines]
	training_label_filenames = [line.split(',')[1] for line in training_lines]
	val_image_filenames = [line.split(',')[0] for line in val_lines]
	val_label_filenames = [line.split(',')[1] for line in val_lines]
	
	return (training_image_filenames, training_label_filenames,
			val_image_filenames, val_label_filenames)

def next_filename(image_filenames, label_filenames):
	""" Creates an infinite shuffling queue with (image, label) filename pairs
	and returns the next example.
	
	Args:
		image_filenames: A list of strings. Image filenames
		label_filenames: A list of strings. Label filenames.
		
	Returns:
		next_filenames: A tuple of strings. The next image, label pair
	"""
	with tf.name_scope('filename_queue'):
		# Transform input to tensors
		image_filenames = tf.convert_to_tensor(image_filenames)
		label_filenames = tf.convert_to_tensor(label_filenames)
		
		# Create a never-ending, shuffling queue and return the next pair
		next_filenames = tf.train.slice_input_producer([image_filenames,
														label_filenames])
		
	return next_filenames

def preprocess_example(image_filename, label_filename, data_dir):
	""" Loads an image (and its label) and augments it.
	
	Args:
		image_filename: A string. Image filename
		label_filename: A string. Label filename
		data_dir: A string. Path to the data directory.
	
	Returns:
		whitened_image: A tensor of floats with shape [height, width, channels].
			Image after preprocessing.
		whitened_label: A tensor of floats with shape [height, width]. Label
	"""
	with tf.name_scope('decode_image'):
		# Load image
		image_path = data_dir + os.path.sep + image_filename
		image_content = tf.read_file(image_path)
		image = tf.image.decode_png(image_content)
		
		# Load label image
		label_path = data_dir + os.path.sep + label_filename
		label_content = tf.read_file(label_path)
		label = tf.image.decode_png(label_content)
		
	with tf.name_scope('augment_image'):
		# Mirror the image (horizontal flip) with 0.5 chance
		flip_prob = tf.random_uniform([])
		flipped_image = tf.cond(tf.less(flip_prob, 0.5), lambda: image,
								lambda: tf.image.flip_left_right(image))
		flipped_label = tf.cond(tf.less(flip_prob, 0.5), lambda: label,
								lambda: tf.image.flip_left_right(label))
										
		# Rotate image at 0, 90, 180 or 270 degrees
		number_of_rot90s = tf.random_uniform([], maxval=4, dtype=tf.int32)
		rotated_image = tf.image.rot90(flipped_image, number_of_rot90s)
		rotated_label = tf.image.rot90(flipped_label, number_of_rot90s)
		
	with tf.name_scope('whiten_image'):		
		# Whiten the image (zero-center and unit variance)
		whitened_image = tf.image.per_image_whitening(rotated_image)
		whitened_label = tf.squeeze(rotated_label) # not whiten, just unwrap it
				
	return whitened_image, whitened_label
		
def train(training_steps = TRAINING_STEPS, learning_rate=LEARNING_RATE, 
		 lambda_=LAMBDA, resume_training=RESUME_TRAINING, data_dir = DATA_DIR,
		 model_dir=MODEL_DIR, csv_path=CSV_PATH, val_csv_path=VAL_CSV_PATH, 
		 num_val_patients = NUM_VAL_PATIENTS):
	""" Trains a convolutional network reporting results for a validation set"""
	# Create model directory
	if not os.path.exists(model_dir): os.makedirs(model_dir)
	
	# Read csv file(s) with training info
	if val_csv_path:
		training_images, training_labels = read_csv_info(csv_path)
		val_images, val_labels = read_csv_info(val_csv_path) 
	else:
		training_images, training_labels, val_images, val_labels = val_split(
										  csv_path, num_val_patients, model_dir)
	
	# Create an stream of filenames and return the next pair
	training_filenames = next_filename(training_images, training_labels)
	val_filenames = next_filename(val_images, val_labels)
	
	# Variables that change between runs: need to be feeded to the graph
	image_filename = tf.placeholder(tf.string, name='image_filename')
	label_filename = tf.placeholder(tf.string, name='label_filename')
	drop = tf.placeholder(tf.bool, shape=(), name='drop') # Dropout? (T/F)
	
	# Read and augment image
	image, label = preprocess_example(image_filename, label_filename, data_dir)

	# Define the model
	prediction = model.forward(image, drop)
	
	# Compute the loss
	logistic_loss = model.loss(prediction, label)
	loss = logistic_loss + lambda_ * model.regularization_loss()
		
	# Set an optimizer
	train_op, global_step = model.update_weights(loss, learning_rate)
	
	# Get a summary writer
	summary_writer = tf.train.SummaryWriter(model_dir)
	summaries = tf.merge_all_summaries()
	
	# Get a saver (for checkpoints)
	saver = tf.train.Saver()

	# Use CPU-only. To enable GPU, delete this and call with tf.Session() as ...
	config = tf.ConfigProto(device_count={'GPU':0})
	
	# Launch graph
	with tf.Session(config=config) as sess:
		# Initialize variables
		if resume_training:
			checkpoint_path = tf.train.latest_checkpoint(model_dir)
			log("Restoring model from:", checkpoint_path)
			saver.restore(sess, checkpoint_path)	
		else:
			tf.initialize_all_variables().run()
			summary_writer.add_graph(sess.graph)
		
		# Start queue runners
		queue_runners = tf.train.start_queue_runners()
				
		# Initial log
		step = global_step.eval()
		log("Starting training @", step)
		
		# Training loop
		for i in range(training_steps):
			# Train
			filenames = sess.run(training_filenames)
			feed_dict = {image_filename: filenames[0], 
						 label_filename: filenames[1], drop: True}
			train_logistic_loss, train_loss, _ = sess.run([logistic_loss, loss,
														   train_op], feed_dict)
			step += 1
			
			# Report losses (calculated before the training step)
			loss_summary = tf.scalar_summary(['training/logistic_loss', 
											  'training/loss'],
								 			 [train_logistic_loss, train_loss], 
								 			 collections=[])
			summary_writer.add_summary(loss_summary.eval(), step - 1)
			log("Training loss @", step - 1, ":", train_logistic_loss,
				"(logistic)", train_loss, "(total)")
			
			# Write summaries
			if step%50 == 0 or step == 1:
				summary_str = summaries.eval(feed_dict)
				summary_writer.add_summary(summary_str, step)
				log("Summaries written @", step)
			
			# Evaluate model
			if step%100 == 0 or step == 1:
				log("Evaluating model")
				
				# Average loss over 5 val images
				val_loss = 0
				number_of_images = 5
				for j in range(number_of_images):
					filenames = sess.run(val_filenames)
					feed_dict ={image_filename: filenames[0], 
								label_filename: filenames[1], drop: False}
					one_loss = logistic_loss.eval(feed_dict)
					val_loss += (one_loss / number_of_images)

				# Report validation loss	
				loss_summary = tf.scalar_summary('val/logistic_loss', val_loss, 
												 collections=[])
				summary_writer.add_summary(loss_summary.eval(), step)
				log("Validation loss @", step, ":", val_loss)
			
			# Write checkpoint	
			if step%250 == 0 or i == (training_steps - 1):
				checkpoint_name = os.path.join(model_dir, 'chkpt')
				checkpoint_path = saver.save(sess, checkpoint_name, step)
				log("Checkpoint saved in:", checkpoint_path)
			
		# Final log
		log("Done!")
		
	# Flush and close the summary writer
	summary_writer.close()

# Trains a model from scratch
if __name__ == "__main__":
	train()
	
	# Optionally: Compute FROC
	log('Computing FROC')
	os.system('python3 compute_FROC.py ' + MODEL_DIR +  ' ' +
			   os.path.join(MODEL_DIR, 'val.csv'))
