# Written by: Erick Cobos T (a01184857@itesm.mx)
# Date: Sep-2016
""" TensorFlow implementation of the convolutional network described in Ch. 3
(Experiment 4) of the thesis report. Works for Tensorflow 0.10.0

It loads each mammogram and its label to memory, computes the function described
by the network and produces a segmentation of the same size as the original 
mammogram. The network outputs a heatmap of logits indicating the probability of
mass accross the mammogram. It uses separate lists of (preprocessed and 
augmented) mammograms for training and validation. Labels have value 0 for
background, 127 for breast tissue and 255 for breast masses. It uses a weighted 
loss function where errors commited on breast masses are weighted by 0.9, errors
on normal breast tissue are weighted by 0.1 and errors on background are 
ignored (weighted by 0).

Software design follows examples from the TensorFlow tutorials. It uses all
available CPUs and a single GPU (if available) in one machine, i.e., it is not
distributed.

See specific methods for implementation details.

Example:
	To train a network from scratch:
		$ python3 model_v3.py

	To resume training (restoring variables from the latest checkpoint):
		$ python3 model_v3.py arg
	where arg is any argument (--reload is recommeded).
	
	You can also load the module in a python3 terminal and run
		>>> model.main(restore_variables)
	where restore_variables is a boolean signaling whether you want to resume
	training (True) or start from scratch (False) (default).
	
Note:
	To run main() more than once in the same Python terminal you will need to
	reset the Tensorflow graph (tf.reset_default_graph()) to clear previous
	events.
"""
import tensorflow as tf
import os.path
import time
import sys

# Set some training parameters
TRAINING_STEPS = 163*8*5 # 163 mammograms (approx) * 8 augmentations * 5 epochs
LEARNING_RATE = 4e-5
LAMBDA = 4e-4

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
	augments the image, creates a FIFO queue for examples, adds a single-thread
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
		image_filename, label_filename = tf.decode_csv(csv_record, [[""], [""]])
	
		# Load image
		image_path = data_dir + os.path.sep + image_filename
		image_content = tf.read_file(image_path)
		image = tf.image.decode_png(image_content)
		
		# Load label image
		label_path = data_dir + os.path.sep + label_filename
		label_content = tf.read_file(label_path)
		label = tf.image.decode_png(label_content)
		
	with tf.name_scope('augmentation'):
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
		
		# Whiten the image (zero-center and unit variance)
		whitened_image = tf.image.per_image_whitening(rotated_image)
		whitened_label = tf.squeeze(rotated_label) # not whiten, just unwrap it
	
	with tf.name_scope('example_queue'):
		# Create example queue
		dtypes = [whitened_image.dtype, whitened_label.dtype]
		example_queue = tf.FIFOQueue(capacity, dtypes)
		enqueue_op = example_queue.enqueue((whitened_image, whitened_label))

		# Create queue_runner
		queue_runner = tf.train.QueueRunner(example_queue, [enqueue_op])
		tf.train.add_queue_runner(queue_runner)
		
	example = example_queue.dequeue(name=name)
			
	return example
	
def model(image, drop):
	""" A fully convolutional network for image segmentation.

	The architecture is modelled as a small ResNet network. It has 0.9 million
	parameters. It uses strided convolutions (instead of pooling) and dilated 
	convolutions to aggregate content and obtain segmentations with good
	resolution. 

	Input size: 128 x 128
	Downsampling size (before BILINEAR): 32 x 32 
	Output size: 128 x 128 (4x upsampling)

	See Section 3.5 of the thesis report for details.

	Args:
		image: A 3D tensor. The input image
		drop: A boolean. If True, dropout is active.

	Returns:
		A 2D tensor. The predicted segmentation: a logit heatmap.
	"""
	# Define some local functions
	def initialize_weights(filter_shape):
		""" Initializes filter weights with random values.

		Values are drawn from a normal distribution with zero mean and standard
		deviation = sqrt(2/n_in) where n_in is the number of connections to the 
		filter: 90 for a 3x3 filter with depth 10 for instance.
		"""
		n_in = filter_shape[0] * filter_shape[1] * filter_shape[2]
		values = tf.random_normal(filter_shape, 0, tf.sqrt(2/n_in))
		
		return values

	def pad_conv_input(input, filter_shape, strides):
		""" Pads a batch mirroring the edges of each feature map."""

		# Calculate the amount of padding (as done when padding=SAME)
		
		# Proper way (works for any input, filter, stride combination)
		#input_shape = tf.shape(input)
		#in_height = input_shape[1]
		#in_width = input_shape[2]
		#out_height = tf.to_int32(tf.ceil(in_height / strides[1]))
		#out_width  = tf.to_int32(tf.ceil(in_width / strides[2]))
		#pad_height = (out_height - 1) * strides[1] + filter_shape[0] - in_height
		#pad_width = (out_width - 1) * strides[2] + filter_shape[1] - in_width

		# Easier way (works if height and width is divisible by their stride)
		pad_height = filter_shape[0] - strides[1]
		pad_width = filter_shape[1] - strides[2]
		
		pad_top = pad_height // 2
		pad_bottom = pad_height - pad_top
		pad_left = pad_width // 2
		pad_right = pad_width - pad_left 

		# Pad mirroring edges of each feature map
		padding = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
		padded_input = tf.pad(input, padding, 'SYMMETRIC')

		return padded_input

	def conv_op(input, filter_shape, strides=[1, 1, 1, 1]):
		""" Creates filters and biases, and performs a convolution.
		
		Args:
			input: A tensor of floats with shape [batch_size, input_height,
				input_width, input_depth]. The input volume.
			filter_shape: A list of 4 integers with shape [filter_height, 
				filter_width, input_depth, output_depth]. This determines the 
				size and number of filters of the convolution.
			strides: A list of 4 integers. The amount of stride in the four
				dimensions of the input.
			
		Returns:
			A tensor of floats with shape [batch_size, output_height,
			output_width, output_depth]. The product of the convolutional 
			operation.
		"""
		# Create filter and biases
		filter = tf.Variable(initialize_weights(filter_shape), name='weights')
		biases = tf.Variable(tf.zeros([filter_shape[3]]), name='biases')
		
		# Add weights to the weights collection (for regularization)
		tf.add_to_collection(tf.GraphKeys.WEIGHTS, filter)
		
		# Do mirror padding (to deal better with borders)
		padded_input = pad_conv_input(input, filter_shape, strides)

		# Perform 2-d convolution
		w_times_x = tf.nn.conv2d(padded_input, filter, strides, padding='VALID')
		output = tf.nn.bias_add(w_times_x, biases)
		
		return output

	def pad_atrous_input(input, filter_shape, dilation):
		"""Pads a batch mirroring feature map's edges. For atrous convolution"""
		# Calculate the amount of padding (as done when padding=SAME)
		pad_height = dilation * (filter_shape[0] - 1)
		pad_width = dilation * (filter_shape[1] - 1)
		
		pad_top = pad_height // 2
		pad_bottom = pad_height - pad_top
		pad_left = pad_width // 2
		pad_right = pad_width - pad_left 

		# Pad mirroring edges of each feature map
		padding = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
		padded_input = tf.pad(input, padding, 'SYMMETRIC')

		return padded_input

		
	def atrous_conv_op(input, filter_shape, dilation):
		""" Creates filters and biases, and performs a dilated convolution.
		
		Args:
			input: A tensor of floats with shape [batch_size, input_height,
				input_width, input_depth]. The input volume.
			filter_shape: A list of 4 integers with shape [filter_height, 
				filter_width, input_depth, output_depth]. This determines the 
				size and number of filters of the convolution.
			dilation: A positive integer. The amount of dilation in the spatial
				dimensions of the volume.
	
		Returns:
			A tensor of floats with shape [batch_size, output_height,
			output_width, output_depth]. The product of the convolutional 
			operation.
		"""
		# Create filter and biases
		filter = tf.Variable(initialize_weights(filter_shape), name='weights')
		biases = tf.Variable(tf.zeros([filter_shape[3]]), name='biases')
		
		# Add weights to the weights collection (for regularization)
		tf.add_to_collection(tf.GraphKeys.WEIGHTS, filter)

		# Do mirror padding (to deal better with borders)
		padded_input = pad_atrous_input(input, filter_shape, dilation)

		# Perform dilated 2d convolution
		w_times_x = tf.nn.atrous_conv2d(padded_input, filter, dilation,
										padding='VALID')
		output = tf.nn.bias_add(w_times_x, biases)

		return output

	def leaky_relu(x, alpha=0.1):
		""" Leaky ReLU activation function."""
		with tf.name_scope('leaky_relu'):
			output = tf.maximum(tf.mul(alpha, x), x)
		return output

	def dropout(x, keep_prob):
		""" Performs dropout if training. Otherwise, returns original."""
		output = tf.cond(drop, lambda: tf.nn.dropout(x, keep_prob), lambda: x)		
		return output
		
	# Create a batch with a single image
	batch = tf.expand_dims(image, 0)	
	
	# Define the architecture
	with tf.name_scope('conv1'):
		conv = conv_op(batch, [6, 6, 1, 32], [1, 2, 2, 1]) 
		relu = leaky_relu(conv)
		conv1 = dropout(relu, keep_prob=0.9)
	with tf.name_scope('conv2'):
		conv = conv_op(conv1, [3, 3, 32, 32]) 
		relu = leaky_relu(conv)
		conv2 = dropout(relu, keep_prob=0.9)

	with tf.name_scope('conv3'):
		conv = conv_op(conv2, [3, 3, 32, 64], [1, 2, 2, 1]) 
		relu = leaky_relu(conv)
		conv3 = dropout(relu, keep_prob=0.8)
	with tf.name_scope('conv4'):
		conv = conv_op(conv3, [3, 3, 64, 64]) 
		relu = leaky_relu(conv)
		conv4 = dropout(relu, keep_prob=0.8)

	with tf.name_scope('conv5'):
		conv = atrous_conv_op(conv4, [3, 3, 64, 128], dilation=2) 
		relu = leaky_relu(conv)
		conv5 = dropout(relu, keep_prob=0.7)
	with tf.name_scope('conv6'):
		conv = atrous_conv_op(conv5, [3, 3, 128, 128], dilation=2) 
		relu = leaky_relu(conv)
		conv6 = dropout(relu, keep_prob=0.7)
	with tf.name_scope('conv7'):
		conv = atrous_conv_op(conv6, [3, 3, 128, 128], dilation=2) 
		relu = leaky_relu(conv)
		conv7 = dropout(relu, keep_prob=0.7)
	with tf.name_scope('conv8'):
		conv = atrous_conv_op(conv7, [3, 3, 128, 128], dilation=2) 
		relu = leaky_relu(conv)
		conv8 = dropout(relu, keep_prob=0.7)
	
	with tf.name_scope('conv9'):
		conv = atrous_conv_op(conv8, [3, 3, 128, 256], dilation=4) 
		relu = leaky_relu(conv)
		conv9 = dropout(relu, keep_prob=0.6)
		
	with tf.name_scope('fc'):
		fc = atrous_conv_op(conv9, [8, 8, 256, 1], dilation=4)
		
	with tf.name_scope('upsampling'):
		new_dimensions = tf.shape(fc)[1:3] * 4
		output = tf.image.resize_bilinear(fc, new_dimensions)
		
	# Summarize activations (verbose)
	tf.histogram_summary('conv1/activations', conv1)
	tf.histogram_summary('conv2/activations', conv2)
	tf.histogram_summary('conv3/activations', conv3)
	tf.histogram_summary('conv4/activations', conv4)
	tf.histogram_summary('conv5/activations', conv5)
	tf.histogram_summary('conv6/activations', conv6)
	tf.histogram_summary('conv7/activations', conv7)
	tf.histogram_summary('conv8/activations', conv8)
	tf.histogram_summary('conv9/activations', conv9)
	tf.histogram_summary('fc/activations', fc)
	
	# Unwrap segmentation
	prediction = tf.squeeze(output)	

	return prediction
	
def logistic_loss(prediction, label):
	""" Logistic loss function averaged over pixels in the breast area.
	
	Losses are weighted depending on the tissue where they occur: losses on 
	masses are weighted by 0.9, on normal tissue by 0.1 and on the background by
	0 (ignored).
	
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
		tissue = tf.to_float(tf.equal(label, 127))
		breast_area = mass + tissue
		
		# Compute loss per pixel
		pixel_loss = tf.nn.sigmoid_cross_entropy_with_logits(prediction, mass)
	
		# Weight the errors
		weighted_loss = 0.9 * tf.mul(pixel_loss, mass)
		weighted_loss += 0.1 * tf.mul(pixel_loss, tissue)
	
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
		if gradient is not None:
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
	#TODO: Maybe read and split the csv file here and the call new example to do the dequeing
	# Create example queue and get a new example
	example = new_example(training_csv, training_dir, name='example')
	val_example = new_example(val_csv, val_dir, name='val_example')

	# Variables that may change between runs: feeded to the graph every time.
	image = tf.placeholder(tf.float32, name='image')	# x
	label = tf.placeholder(tf.uint8, name='label')	# y
	drop = tf.placeholder(tf.bool, shape=(), name='drop')	# Dropout? (T/F)

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
	if not os.path.exists(summary_dir): os.makedirs(summary_dir)
	saver = tf.train.Saver()
	if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
	coord = tf.train.Coordinator()

	# Use CPU-only. To enable GPU, delete this and call with tf.Session() as ...
	config = tf.ConfigProto(device_count={'GPU':0})
	
	# Launch graph
	with tf.Session(config=config) as sess:
		# Initialize variables
		if restore_variables:
			checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
			saver.restore(sess, checkpoint_path)
			log("Variables restored from:", checkpoint_path)
		else:
			tf.initialize_all_variables().run()
			summary_writer.add_graph(sess.graph)
		
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
					val_image, val_label = sess.run(val_example)
					feed_dict ={image: val_image, label: val_label, drop: False}
					one_loss = empirical_loss.eval(feed_dict)
					val_loss += (one_loss / number_of_images)

				# Report validation loss	
				loss_summary = my_scalar_summary('val/logistic_loss', val_loss)
				summary_writer.add_summary(loss_summary, step)
				log("Validation loss @", step, ":", val_loss)
			
			# Write checkpoint	
			if step%250 == 0 or i == (TRAINING_STEPS - 1):
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
