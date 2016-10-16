# Written by: Erick Cobos T.
# Date: October 2016
""" Trains a convolutional network with the provided data.

	It uses all available CPUs and a single GPU (if available) in one machine, 
	i.e., it is not distributed.
	
	Example:
		python3 train.py
"""
import tensorflow as tf
import numpy as np
import os.path
from utils import log, read_csv_info

# Import network definition
import model_v4 as model

# Set training parameters
TRAINING_STEPS = 205*8*30 # 205 mammograms (approx) * 8 augmentations * 5 epochs
LEARNING_RATE = 4e-5
LAMBDA = 4e-4
RESUME_TRAINING = False

# Set some paths
DATA_DIR = "data" # folder with training data (images and labels)
MODEL_DIR = "run116" # folder to store model checkpoints and summaries
CSV_PATH = "data/training_1.csv" # path to csv file with image,label filenames


def new_example(image_filenames, label_filenames, data_dir):
	""" Creates an infinite queue of filenames, augments and preprocess the 
	image and returns a new example: (image, label) pair.
	
	Args:
		image_filenames: A list of strings. Image filenames
		label_filenames: A list of strings. Label filenames.
		data_dir: A string. Path to the data directory.
	
	Returns:
		whitened_image: A tensor of floats with shape [height, width, channels].
			Image after preprocessing.
		whitened_label: A tensor of floats with shape [height, width]. Label
	"""
	with tf.name_scope('filename_queue'):
		# Transform input to tensors
		image_filenames = tf.convert_to_tensor(image_filenames)
		label_filenames = tf.convert_to_tensor(label_filenames)
		
		# Create a never-ending, shuffling queue and return the next pair
		image_filename, label_filename = tf.train.slice_input_producer(
											 [image_filenames, label_filenames])
	
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
		 model_dir=MODEL_DIR, csv_path=CSV_PATH):
	""" Creates and trains a convolutional network for image segmentation."""
	# Create model directory
	if not os.path.exists(model_dir): os.makedirs(model_dir)
	
	# Read csv file with training info
	image_filenames, label_filenames = read_csv_info(csv_path)
		
	# Shufle, augment and preprocess input
	image, label = new_example(image_filenames, label_filenames, data_dir)

	# Define the model (with dropout)
	prediction = model.forward(image, drop=tf.constant(True))
	
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
			train_logistic_loss, train_loss, _ = sess.run([logistic_loss, loss,
														   train_op])
			step += 1
			
			# Report losses (calculated before the training step)
			loss_summary = tf.scalar_summary(['logistic_loss', 'loss'],
											 [train_logistic_loss, train_loss], 
											 collections=[])
			summary_writer.add_summary(loss_summary.eval(), step - 1)	
			log("Training loss @", step - 1, ":", train_logistic_loss,
				"(logistic)", train_loss, "(total)")
			
			# Write summaries
			if step%50 == 0 or step == 1:
				summary_str = summaries.eval()
				summary_writer.add_summary(summary_str, step)
				log("Summaries written @", step)
			
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
