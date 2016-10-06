# Written by: Erick Cobos T.
# Date: October 2016
""" Trains the convolutional network with the provided data.

	Example:
		python3 train.py
"""
import model_v4.py as model
import tensorflow as tf
import os

# Set training parameters
TRAINING_STEPS = 163*8*5 # 163 mammograms (approx) * 8 augmentations * 5 epochs
LEARNING_RATE = 4e-5
LAMBDA = 4e-4
RESUME_TRAINING = False

# Set some paths
DATA_DIR = "data" # folder with training data (images and labels)
MODEL_DIR = "run116" # folder to store model checkpoints and summaries
CSV_PATH = "training.csv" # path to csv file with image and label filenames


def read_csv_info(csv_path):
	""" Reads the csv file and returns two lists: one with image filenames and 
	one with label filenames."""
	filenames = np.loadtxt(csv_path, dtype=bytes, delimiter=',').astype(str)
	
	return list(filenames[:,0]), list(filenames[:,1])

def new_example(image_filenames, label_filenames, data_dir):
	""" Creates an example queue and returns a new example: (image, label).
	
	Creates a never-ending suffling queue of filenames, dequeues and decodes a 
	csv record, loads an example (image and label) in memory and augments it.
	
	, creates a FIFO queue for examples, adds a single-thread
	QueueRunner object to the graph to perform prefetching operations and 
	dequeues an example.
	
	Uses queues to improve performance (as recommended in the tutorials). We 
	could not use tf.train.batch() to automatically create the example queue 
	because	our images differ in size.
	
	Args:
		csv_path: A string. Path to csv file with image and label filenames.
			Each record is expected to be in the form:
			'image_filename,label_filename'
		data_dir: A string. Path to the data directory. Default is "."
	
	Returns:
		An (image, label) tuple where image is a tensor of floats with shape
		[image_height, image_width, image_channels] and label is a tensor of
		integers with shape [image_height, image_width]
	"""
	with tf.name_scope('filename_queue'):
		# Transform input to tensors
		image_filenames = tf.convert_to_tensor(image_filenames)
		label_filenames = tf.convert_to_tensor(label_filenames)
		
		# Create never-ending shuffling queue of filenames
		filename_queue = tf.train.slice_input_producer([image_filenames, 
														label_filenames])
	
	with tf.name_scope('decode_image'):
		# Dequeue next example
		image_filename, label_filename = filename_queue.dequeue()
	
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
		
	#TODO: Test whether prefetching helps now that it doesn't do sess.run(example)
				
	return whitened_image, whitened_label

def log(*messages):
	""" Simple logging function."""
	formatted_time = "[{}]".format(time.ctime())
	print(formatted_time, *messages)
	
def train(training_steps = TRAINING_STEPS, learning_rate=LEARNING_RATE, 
		 lambda_=LAMBDA, resume_training=RESUME_TRAINING, csv_path=CSV_PATH, 
		 model_dir=MODEL_DIR):
	""" Creates and trains a convolutional network for image segmentation. 
	
	It creates an example queue; defines a model, loss function and optimizer;
	and trains the model.
	
	Args:
		restore_variables: A boolean. Whether to restore variables from a 
			previous execution. Default to False.
	
	"""
	# Read csv file with training info
	image_filenames, label_filenames =  read_csv_info(csv_path)
		
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
	if not os.path.exists(model_dir): os.makedirs(model_dir)
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
											 collections = [])
			summary_writer.add_summary(loss_summary, step - 1)	
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
