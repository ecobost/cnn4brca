# Written by: Erick Cobos T (a01184857@itesm.mx)
# Date: March-2016
""" TensorFlow implementation of the convolutional network described in Ch. 3 of
the thesis report.

It loads each mammogram and its label to memory, computes the function described
by the network, and produces a segmentation of the same size as the original 
mammogram. The network outputs a heatmap of logits indicating the probability of
mass accross the mammogram. The network uses separate lists of (preprocessed and
augmented) mammograms for training and validation. Labels have value 0 for
background, 127 for breast tissue and 255 for breast masses.

Software design follows examples from the TensorFlow tutorials. It uses all
available CPUs and a single GPU (if available) in one machine, i.e., it is not
distributed.

See specific methods for implementation details.

Example:
	To train a network from scratch:
		$ python3 model.py

	To resume training (restoring variables from the latest checkpoint):
		$ python3 model.py arg
	where arg is any argument.
	
	You can also load the module in a python3 terminal and run
		>>> model.main(restore_variables)
	where restore_variables is a boolean signaling whether you want to resume
	training (True) or start from scratch (False) (default).
	
Note:
	To run main() more than once in the same Python terminal you will need to
	reset the Tensorflow graph (tf.reset_default-graph()) to clear previous
	events.
"""
import tensorflow as tf
import math
import os.path
import time
import sys

# Set some training parameters
TRAINING_STEPS = 500
LEARNING_RATE = 1e-4
LAMBDA = 1e-3

# Set some paths
training_dir = "training" 
"""string: Folder with training data."""

val_dir = "val"
"""string: Folder with validation data."""

training_csv = os.path.join(training_dir, "training.csv")
"""string: Path to csv file with image and label filenames for training."""

val_csv = os.path.join(val_dir, "val.csv")
"""string: Path to csv file with image and label filenames for validation."""

summary_dir = "summary"
"""string: Folder to store summary files."""

checkpoint_dir = "checkpoint"
"""string: Folder to store model checkpoints."""


def new_example(csv_path, data_dir=".", capacity=5, name='new_example'):
	""" Creates an example queue and returns a new example: (image, label).
	
	Reads the csv file, creates a never-ending suffling queue of filenames,
	dequeues and decodes a csv record, loads an image and label in memory,
	whitens the image, creates a FIFO queue for examples, adds a single-thread
	QueueRunner object to the graph to perform prefetching operations and 
	dequeues an example.
	
	Uses queues to improve performance (as recommended in the tutorials). We can
	not use tf.train.batch() to automatically create the example queue because
	our images differ in size.
	
	Args:
		csv_path: A string. Path to csv file with image and label filenames.
			Each record is expected to be in the form:
			'image_filename,label_filename,smallLabel_filename'
		data_dir: A string. Path to the data directory. Default is "."
		capacity: An integer. Maximum amount of examples that may be stored in 
			the example queue. Default is 5.
		name: A string. Name for the produced examples. Default is 'new_example'
	
	Returns:
		An (image, label) tuple where image is a tensor of floats with shape
		[image_height, image_width, image_channels] and label is a tensor of
		integers with shape [image_height, image_width]
	"""
	with tf.name_scope('filename_queue'):
		# Read csv file
		with open(csv_path) as f:
			lines = f.read().splitlines()
			
		# Create never-ending shuffling queue of filenames
		filename_queue = tf.train.string_input_producer(lines)
	
	with tf.name_scope('decode_image'):
		# Decode a csv record
		csv_record = filename_queue.dequeue()
		image_filename, label_filename, _ = tf.decode_csv(csv_record,
														  [[""], [""], [""]])
	
		# Load image
		image_path = data_dir + os.path.sep + image_filename
		image_content = tf.read_file(image_path)
		image = tf.image.decode_png(image_content)
		
		# Load label image
		label_path = data_dir + os.path.sep + label_filename
		label_content = tf.read_file(label_path)
		label = tf.image.decode_png(label_content)
		label = tf.squeeze(label) # Unwrap label

	with tf.name_scope('whitening'):
		# Preprocess image (whitening)
		image = tf.image.per_image_whitening(image)
	
	with tf.name_scope('example_queue'):
		# Create example queue
		example_queue = tf.FIFOQueue(capacity, [image.dtype, label.dtype])
		enqueue_op = example_queue.enqueue((image, label))

		# Create queue_runner
		queue_runner = tf.train.QueueRunner(example_queue, [enqueue_op])
		tf.train.add_queue_runner(queue_runner)
		
	example = example_queue.dequeue(name=name)
			
	return example
	
def model(image, drop):
	""" A fully convolutional network for image segmentation.

	The architecture is modelled on the VGG-16 network but smaller. It has
	approximately 2.9 million parameters.

	Architecture:
		INPUT -> [[CONV -> Leaky RELU]*2 -> MAXPOOL]*2 -> [CONV -> Leaky RELU]*3
		-> MAXPOOL -> FC -> Leaky RELU -> FC -> SIGMOID -> BICUBIC
	Input size: 112 x 112
	Downsampling size (before BICUBIC): 7 x 7 
	Output size: 112 x 112 (16x upsampling)

	See Section 3.3 of the thesis report for further details.

	Args:
		image: A 3D tensor. The input image
		drop: A boolean. If True, dropout is active.

	Returns:
		A 2D tensor. The predicted segmentation: a logit heatmap.
	"""
	def initialize_weights(shape):
		""" Initializes filter weights with random values.

		Values are drawn from a normal distribution with zero mean and standard
		deviation = sqrt(2/n_in) where n_in is the number of connections to the 
		filter: 90 for a 3x3 filter with depth 10 for instance.
		"""
		n_in = shape[0] * shape[1] * shape[2]
		values = tf.random_normal(shape, 0, math.sqrt(2/n_in))
		
		return values

	def conv_op(input, filter_shape, strides):
		""" Creates filters and biases and performs the convolution."""
		# Create filter and biases
		filter = tf.Variable(initialize_weights(filter_shape), name='weights')
		biases = tf.Variable(tf.zeros([filter_shape[3]]), name='biases')
		
		# Add weights to the weights collection (for regularization)
		tf.add_to_collection(tf.GraphKeys.WEIGHTS, filter)
		
		# Perform 2-d convolution
		w_times_x = tf.nn.conv2d(input, filter, strides, padding='SAME')
		output = tf.nn.bias_add(w_times_x, biases)
		
		return output

	def leaky_relu(x, alpha=0.1):
		""" Leaky ReLU activation function."""
		with tf.name_scope('leaky_relu'):
			output = tf.maximum(tf.mul(alpha, x), x)
		return output

	def dropout(x, keep_prob):
		""" During training, performs dropout. Otherwise, returns original."""
		output = tf.nn.dropout(x, keep_prob) if drop else x
		return output

	def conv_layer(input, filter_shape, strides=[1, 1, 1, 1], keep_prob=1):
		""" Adds a convolutional layer to the graph. 
	
		Creates filters and biases, computes the convolutions, passes the output
		through a leaky ReLU activation function and applies dropout. Equivalent
		to calling conv_op()->leaky_relu()->dropout().

		Args:
			input: A tensor of floats with shape [batch_size, input_height,
				input_width, input_depth]. The input volume.
			filter_shape: A list of 4 integers with shape [filter_height, 
			filter_width, input_depth, output_depth]. This determines the size
			and number of filters of the convolution.
			strides: A list of 4 integers. The amount of stride in the four
				dimensions of the input.
			keep_prob: A float. Probability of dropout in the layer.
			
		Returns:
			A tensor of floats with shape [batch_size, output_height,
			output_width, output_depth]. The product of the convolutional layer.
		"""
		# conv -> relu -> dropout
		conv = conv_op(input, filter_shape, strides) 
		relu = leaky_relu(conv)
		output = dropout(relu, keep_prob)
		
		# Summarize activations
		scope = tf.get_default_graph()._name_stack[0] # No easier way
		tf.histogram_summary(scope + '/activations', output)
		
		return output


	batch = tf.expand_dims(image, 0)	# Batch with a single image
	
	# conv1 -> conv2 -> pool1
	with tf.name_scope('conv1'):
		conv1 = conv_layer(batch, [6, 6, 1, 56], [1, 2, 2, 1], keep_prob=0.9)
	with tf.name_scope('conv2'):
		conv2 = conv_layer(conv1, [3, 3, 56, 56], keep_prob=0.9)
	with tf.name_scope('pool1'):
		pool1 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

	# conv3 -> conv4 -> pool2
	with tf.name_scope('conv3'):
		conv3 = conv_layer(pool1, [3, 3, 56, 84], keep_prob=0.8)
	with tf.name_scope('conv4'):
		conv4 = conv_layer(conv3, [3, 3, 84, 84], keep_prob=0.8)
	with tf.name_scope('pool2'):
		pool2 = tf.nn.max_pool(conv4, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

	# conv5 -> conv6 -> conv7 -> pool3
	with tf.name_scope('conv5'):
		conv5 = conv_layer(pool2, [3, 3, 84, 112], keep_prob=0.7)
	with tf.name_scope('conv6'):
		conv6 = conv_layer(conv5, [3, 3, 112, 112], keep_prob=0.7)
	with tf.name_scope('conv7'):
		conv7 = conv_layer(conv6, [3, 3, 112, 112], keep_prob=0.7)
	with tf.name_scope('pool3'):
		pool3 = tf.nn.max_pool(conv7, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
	
	# fc1 -> fc2
	# FC layers implemented as size-preserving convolutional layers
	with tf.name_scope('fc1'):
		fc1 = conv_layer(pool3, [7, 7, 112, 448], keep_prob=0.6)
	with tf.name_scope('fc2') as scope:
		fc2 = conv_op(fc1, [1, 1, 448, 1], [1, 1, 1, 1])
		tf.histogram_summary(scope + 'activations', fc2)
		
	# upsampling
	with tf.name_scope('upsampling'):
		new_dimensions = tf.shape(fc2)[1:3] * 16
		output = tf.image.resize_bilinear(fc2, new_dimensions)
	
	prediction = tf.squeeze(output)	# Unwrap segmentation

	return prediction
	
def logistic_loss(prediction, label):
	""" Logistic loss function averaged over pixels in the breast area.
	
	Pixels in the background are ignored.
	
	Args:
		prediction: A 2D tensor of floats. The predicted heatmap of logits.
		label: A 2D tensor of integers. Possible labels are 0 (background), 127
			(breast tissue) and 255 (breast mass).

	Returns:
		A float. The loss.
	"""
	with tf.name_scope('logistic_loss'):
		# Generate binary masks.
		mass = tf.to_float(tf.equal(label, 255))
		breast_area = tf.to_float(tf.greater(label, 0))

		# Compute loss per pixel
		pixel_loss = tf.nn.sigmoid_cross_entropy_with_logits(prediction, mass)
	
		# Weight the errors (1 for pixels in breast area, zero otherwise)
		weighted_loss = tf.mul(pixel_loss, breast_area)
	
		# Average over pixels in the breast area
		loss = tf.reduce_sum(weighted_loss)/tf.reduce_sum(breast_area)

	return loss
	
def regularization_loss():
	""" Calculates the l2 regularization loss from the collected weights."""
	with tf.name_scope("regularization_loss"):
		# Compute the (halved and squared) l2-norm of each weight matrix
		weights = tf.get_collection(tf.GraphKeys.WEIGHTS)
		l2_losses = [tf.nn.l2_loss(x) for x in weights]
		
		# Add all regularization losses
		loss = tf.add_n(l2_losses)
		
	return loss
	
def train(loss, learning_rate):
	""" Sets up an ADAM optimizer, computes gradients and updates variables.
	
	Args:
		loss: A float. The loss function to minimize.
		learning_rate: A float. The learning rate for ADAM.
	
	Returns:
		train_op: The operation to run for training.
		global_step: The current number of training steps made by the optimizer.
	"""
	# Set optimization parameters
	global_step = tf.Variable(0, name='global_step', trainable=False)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, 
									   beta2=0.995, epsilon=1e-06)
	
	# Compute and apply gradients		   
	gradients = optimizer.compute_gradients(loss)
	train_op = optimizer.apply_gradients(gradients, global_step=global_step)
	
	# Summarize gradients
	for gradient, variable in gradients:
		if gradient:
			tf.histogram_summary(variable.op.name + '/gradients', gradient)

	return train_op, global_step
	
def log(*messages):
	""" Simple logging function."""
	formatted_time = "[{}]".format(time.ctime())
	print(formatted_time, *messages)
	
def my_scalar_summary(tag, value):
	""" Manually creates an scalar summary that is not added to the graph."""
	float_value = float(value)
	summary_value = tf.Summary.Value(tag=tag, simple_value=float_value)
	return tf.Summary(value=[summary_value])
		
def main(restore_variables=False):
	""" Creates and trains a convolutional network for image segmentation. 
	
	It creates an example queue; defines a model, loss function and optimizer;
	and trains the model.
	
	Args:
		restore_variables: A boolean. Whether to restore variables from a 
			previous execution. Default to False.
	
	"""
	# Create example queue and get a new example
	example = new_example(training_csv, training_dir, name='example')
	val_example = new_example(val_csv, val_dir, name='val_example')

	# Variables that may change between runs: feeded to the graph every time.
	image = tf.placeholder(tf.float32, name='image')	# x
	label = tf.placeholder(tf.uint8, name='label')	# y
	drop = tf.placeholder(tf.bool, shape=[], name='drop')	# Dropout? (T/F)

	# Define the model
	prediction = model(image, drop)
	
	# Compute the loss
	empirical_loss = logistic_loss(prediction, label)
	loss = empirical_loss + LAMBDA * regularization_loss()
		
	# Set an optimizer
	train_op, global_step = train(loss, learning_rate=LEARNING_RATE)
	
	# Get a summary writer, saver and coordinator
	summaries = tf.merge_all_summaries()
	summary_writer = tf.train.SummaryWriter(summary_dir)
	saver = tf.train.Saver()
	coord = tf.train.Coordinator()

	# Launch the graph
	with tf.Session() as sess:
		# Initialize variables
		if restore_variables:
			checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
			saver.restore(sess, checkpoint_path)
			log("Variables restored from:", checkpoint_path)
		else:
			tf.initialize_all_variables().run()
			summary_writer.add_graph(sess.graph_def)
			
		# Uncomment this to create different folders after every restore.
		#run_path = os.path.join(summary_dir, "run{}".format(global_step.eval()))
		#summary_writer = tf.train.SummaryWriter(run_path, sess.graph_def)
		
		# Start queue runners
		queue_runners = tf.train.start_queue_runners(coord=coord)
				
		# Initial log
		step = global_step.eval()
		log("Starting training @", step)
		
		# Training loop
		for i in range(TRAINING_STEPS):
			# Train
			train_image, train_label = sess.run(example)
			feed_dict = {image: train_image, label: train_label, drop: True}
			train_logistic_loss, train_loss, _ = sess.run([empirical_loss, loss,
														   train_op], feed_dict)
			step += 1
			
			# Report losses (calculated before the training step)
			logistic_loss_summary = my_scalar_summary('training/logistic_loss',
											 		  train_logistic_loss)
			summary_writer.add_summary(logistic_loss_summary, step - 1)
			loss_summary = my_scalar_summary('training/loss', train_loss)
			summary_writer.add_summary(loss_summary, step - 1)		
			log("Training loss @", step - 1, ":", train_logistic_loss,
				"(logistic)", train_loss, "(total)")
			
			# Write summaries
			if step%25 == 0 or step == 1:
				summary_str = summaries.eval(feed_dict)
				summary_writer.add_summary(summary_str, step)
				log("Summaries written @", step)
			
			# Evaluate model
			if step%25 == 0 or step == 1:
				log("Evaluating model")
				
				# Average loss over 4 val images
				val_loss = 0
				number_of_images = 4
				for j in range(number_of_images):
					val_image, val_label = sess.run(val_example)
					feed_dict ={image: val_image, label: val_label, drop: False}
					one_loss = empirical_loss.eval(feed_dict)
					val_loss += (one_loss / number_of_images)

				# Report validation loss	
				loss_summary = my_scalar_summary('val/logistic_loss', val_loss)
				summary_writer.add_summary(loss_summary, step)
				log("Validation loss @", step, ":", val_loss)
			
			# Write checkpoint	
			if step%100 == 0:
				checkpoint_name = os.path.join(checkpoint_dir, 'model')
				checkpoint_path = saver.save(sess, checkpoint_name, step)
				log("Checkpoint saved in:", checkpoint_path)
			
		# Final log
		log("Done!")
		
		# Stop queue runners
		coord.request_stop()
		coord.join(queue_runners)
		
	# Flush and close the summary writer
	summary_writer.close()
	
# Trains a model from scratch if called without arguments (python3 model.py)
# Otherwise, restores variables from the latest checkpoint in checkpoint_dir.
if __name__ == "__main__":
	main(len(sys.argv) > 1)
	
# Tests:
# Filenames are shuffled
# Both queues (filename and example) work fine.
# Images are read and preprocessed correctly
# Inference works alright for 112 x 112 images
# Dropout works fine
# Graph definition in Tensorboard looks okay.
# Inference works alright for big images (tested with 1305x1579 in desktop)
# Loss in 112x112 and big images works.
# Summaries work fine
# Checkpoint works fine
# Validation works fine
# Regularization works fine
