# Written by: Erick Cobos T (a01184857@itesm.mx)
# Date: 17-March-2016
""" TensorFlow implementation of the convolutional network described in Chapter 3 of the thesis report.

It loads each mammogram and its label to memory, computes the function described by the convolutional network, and produces a segmentation of the same size as the original mammogram. The network outputs a heatmap of the probability of mass accross the mammogram.

The network uses separate lists of (preprocessed and augmented) mammograms for training and validation. Labels have value 0 for background, 127 for breast tissue and 255 for breast masses. 

The design follows guidelines from the TensorFlow tutorials. It uses all available CPUs and a single GPU (if available) in one machine. It is not distributed.

See specific methods for details.
"""

import tensorflow as tf
import math

# Set some parameters
working_dir = "./"	# Directory where search starts and results are written.
training_dir = "training/"	# Training folder
val_dir = "val/"	# Validation folder
training_csv = "training.csv"	# File with image and label filenames for training
val_csv = "val.csv"	# File with image and label filenames for validation


def read_csv(csv_filename):
	""" Reads csv files and creates a never-ending suffling queue of filenames.
	
	Args:
		csv_filename: A string. File with image and label filenames. 

	Returns:
		A queue with strings. Each string is a csv record in the form
		'image_filename,label_filename,smallLabel_filename'
	"""
	with open(csv_filename) as f:
	    lines = f.read().splitlines()

	filename_queue = tf.train.string_input_producer(lines)
	return filename_queue


def new_example(filename_queue, data_dir):
	""" Creates a single new example: an image and its label segmentation.
	
	Dequeues and decodes one csv record, loads images (.png) in	memory, whitens 
	them and returns them.

	Args:
		filename_queue: A queue with strings. Each string is a csv record in the
			form 'image_filename,label_filename,smallLabel_filename'
		data_dir: A string. Path to the data directory.

	Returns:
		image: A 3D tensor of size [image_height, image_width, image_channels]
		label: A 3D tensor of size [image_height, image_width, image_channels]
	"""
	# Reading csv file
	csv_record = filename_queue.dequeue()
	image_filename, label_filename, _ = tf.decode_csv(csv_record,
									  [[""], [""], [""]])
	
	# Reading image
	image_content = tf.read_file(data_dir + image_filename)
	label_content = tf.read_file(data_dir + label_filename)
	image = tf.image.decode_png(image_content)
	label = tf.image.decode_png(label_content)

	# Preprocessing (image whitening)
	image = tf.image.per_image_whitening(image)
	
	#tf.image_summary('images', images)
	return image, label


def create_example_queue(filename_queue, data_dir="./", queue_capacity=5):
	""" Creates a FIFO queue with examples: (image, label) tuples.

	Creates a FIFO queue and adds a single-thread QueueRunner object to the graph
	to perform prefetching operations.

	Note:
		We can not use tf.train.batch()/shuffle_batch() to automatically create 
		this queue because our images differ in size.

	Args:
		filename_queue: A queue with strings. Each string is a csv record in the
			form 'image_filename,label_filename,smallLabel_filename'
		data_dir: A string. Path to the data directory. Default is ./
		queue_capacity: An integer. Maximum amount of examples that may be 
			stored in the queue. Default is 5.

	Returns:
		A queue with (image, label) tuples.
	"""
	# Processing new example
	image, label = new_example(filename_queue, data_dir)
	
	# Creating queue
	example_queue = tf.FIFOQueue(queue_capacity, [image.dtype, label.dtype])
	enqueue_op = example_queue.enqueue((image, label))

	# Creating queue_runner
	queue_runner = tf.train.QueueRunner(example_queue, [enqueue_op])
	tf.train.add_queue_runner(queue_runner)

	return example_queue


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
		A 2D tensor. The predicted segmentation for the input image.
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

	def conv_op(input, filter_shape, strides, name='conv'):
		""" Creates filters and biases and performs the convolution."""
		filter = tf.Variable(initialize_weights(filter_shape), name='weights')
		biases = tf.Variable(tf.zeros([filter_shape[3]]), name='biases')

		w_times_x = tf.nn.conv2d(input, filter, strides, padding='SAME')
		output = tf.nn.bias_add(w_times_x, biases, name=name)
		return output

	def leaky_relu(x, alpha=0.1, name='leaky_relu'):
		""" Leaky ReLU activation function."""
		return tf.maximum(alpha * x, x, name=name)

	def dropout(x, keep_prob, name='dropout'):
		""" During training, performs dropout. Otherwise, returns original."""
		output = tf.nn.dropout(x, keep_prob, name=name) if drop else x
		return output

	def conv_layer(input, filter_shape, strides=[1, 1, 1, 1], keep_prob=1):
		""" Adds a convolutional layer to the graph. 
	
		Creates filters and biases, computes the convolutions, passes the
		output through a leaky ReLU activation function and applies dropout.
		Equivalent to calling conv_op()->leaky_relu()->dropout().

		Args:
			input: A tensor of floats with shape [batch_size, input_height,
				input_width, input_depth]. The input volume.
			filter_shape: A list of 4 integers with shape [filter_height,
				filter_width, input_depth, output_depth]. This determine 
				the size and number of filters of the convolution.
			strides: A list of 4 integers. The amount of stride in the four
				dimensions of the input.
			keep_prob: A float. Probability of dropout in the layer.
			
		Returns:
			A tensor of floats with shape [batch_size, output_height, 
			output_width, output_depth]. The product of the convolutional
			layer.	
		"""
		conv = conv_op(input, filter_shape, strides) 
		relu = leaky_relu(conv)
		output = dropout(relu, keep_prob)
		return output


	batch = tf.expand_dims(image, 0) # Batch with a single image
	
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
	with tf.name_scope('fc2'):
		conv = conv_op(fc1, [1, 1, 448, 1], [1, 1, 1, 1])
		fc2 = tf.sigmoid(conv)

	# upsampling
	with tf.name_scope('upsampling'):
		new_dimensions = tf.shape(fc2)[1:3] * 16
		output = tf.image.resize_bicubic(fc2, new_dimensions, name='upsampling')
	
	prediction = tf.squeeze(output) # Unwrap segmentation

	return prediction

#TODO: Check the graph and change names if needed
#TODO: Check all works
#TODO: Check it works for bigger images.
#TODO: Change in_training to 'drop' because that's what it does.

def train():
	""" Creates and trains a convolutional network for image segmentation. """
	# Create a suffling queue with image and label filenames
	filename_queue = read_csv(working_dir + training_dir + training_csv)
	val_filename_queue = read_csv(working_dir + val_dir + val_csv)

	# Create a queue to prefetch examples. If unnecessary, use new_example()
	example_queue = create_example_queue(filename_queue, working_dir+training_dir)
	val_example_queue = create_example_queue(val_filename_queue, 
							     working_dir+val_dir)

	# Create the computation graph

	# Variables that may change between executions: feeded to the graph every run.
	image = tf.placeholder(tf.float32) # x
	label = tf.placeholder(tf.float32) # y
	drop = tf.placeholder(tf.boolean, shape = []) # Perform dropout (Yes/No).

	# Define the model 
	prediction = model(image, drop)

	# Compute the loss
	#loss = logistic_loss(image, label)

	# Set optimization parameters
	#globalStep = tf.Variable(0, name="global_step",trainable=False)

	# Summaries

	
	# start session
#	init = start()

	# Start training
"""Pseudo-code
Define the model
	Create each layer(maybe with a function)
Define the loss for the model
Define the optimization
Add summaries (images, too)
Start session: where session, variables initialization, threads/coordinator, queuerunners, summary writer and everything else is started,
for epochs number of epochs
	Create a new batch (with a single image) 
		Zero-mean the image
	Prepare the feed
	Train the network

	every epochsSave
		Checkpoint (saver, global_step)
	every epochsToSummary
		Write all summaries
""" 


def test():
	"""For rapid testing"""

	# Test images
	image = tf.image.decode_png(tf.read_file("smallMammogram.png"))
	label = tf.image.decode_png(tf.read_file("smallLabel.png"))
	image = tf.image.per_image_whitening(image)

	# Define the model 
	prediction = model(image, False)

	# Launch the graph.
	sess = tf.Session()
	sess.run(tf.initialize_all_variables())

	# Create graph here so i can chek it

	# ALL EVALUATIONS HERE
	res = sess.run(prediction)

	sess.close()

	return res

# If called as 'python3 model.py' run the main method.
if __name__ == "__main__":	
	pass
	#TODO: train()


"""
% Define some tf.constants for the weights before multiplying.
%Have somwtrhing lik
% breasttissue = (tf.equals(127) float32)
% breastmass = (tf.equals(255) float32)
% background = (tf.equals(0) float32)
% weightMask = breasttissueweight* breasttissue + breastmassWeight* brestWeight + backgorund*backgroundWeight
% labelMask = breastMass
%
%% Or simpler option
%% weightMask = tf.greaterthan(0)
%% labelMask = tf.equal()
%
%%% Or as above but ignoring background
%%% breastissue =...
%%% breastMass = ...
%%% weightMask = breasttissue*breastWeight + breastMass*massWeight
%%% labelMask = breastMass
%
% lossPerPixel = weightMask *( labelMask*log(tata) + (1-labelMask))
% loss = tf.reduce_sum(loss)/tf.reduce_sum(breastTissue + breastMass)


# If I am gonna decrease the learning rate evry x number of steps I can do it with tf.train.exponential_decay() (it's not really exponential)

# summarize images and labels
# write a summarize function that uses tf.histogram_summary and tf.scalar_summary (sparsity see cifar model) in activations after relu and maybe in weight gradients (histogram to see if all are positive in the first and penultimate layer maybe) and maybe first layer filters (not so often though, maybe not), summarize the training and val loss, too. Summarize the reduce_mean of (predicitions) to see whether they start at around 0.5 and decrease (because there is not many positives)
# Summarize only every number of operations, loss should probably be reported every time.
# Maybe accumulate the loss function over every batch that is not printed and then print and average, that way it is probbaly smoother, or just log/summarize the loss for every batch. A batch is an image in our case
# you cna use merge_summary (instead of merge_all_summaries) to merge only a subset and save them to file.
# maybe put all sumarries in a single fuinction that returns the string with all sumaries, and then write it in the main loop. Or just when defining the graph put all sumaries in asingle function and then during trainng call tf.merge all summarie snormally.

# To save checkpoints (see Tensorflow Mechanics 101/ Tensorbord: visualizations how to). Or here: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/cifar10/cifar10_train.py
saver = tf.train.Saver()
saver.save(sess, "tmp/model.ckpt", global_step = global_step)


# Check the MNIST example to see where to define global_step (in the training function.


# <Maybe the main is a call to model = myCOnv(opts, session), model.train and model.eval (as in the Word2vec example). It is build a class myCOnv that has parameters for everything it needs, and then I just call it for it to run the code.
	

# How to write somethings (rather than how they say to)
optimizer = tf.train.ADAMOptimizer


with Session as sess:
	sess.run(tf.initiallize_all_variables)
	summary_writer = tf.train.SummaryWriter(FLAGS.train_dir)
	summary_writer.add_graph(sess.graph_def)
	
	tf.train.start_queue_runners(sess) # Needed for queues
	for i in steps
		
		feed = {x: batch_xs, y_: batch_ys}
		sess.run(optimizer.minimize(loss), feed_dict = feed)

		summary_str = sess.run(tf.merge_all_summaries(), feed_dict=feed_dict)
		summary_writer.add_summary(summary_str, step)

# Create the feed before the sess.run
feed = {x: batch_xs, y_: batch_ys}
sess.run(train_step, feed_dict = feed)

# For eval, define it in the same model or use a scope as in rnn/ptb/ptb_word_lm

# Define l2 norm as element'wise square plus reduce?sum, or as tr(AtxA)
# What about this:https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/models/image/mnist/convolutional.py
  # L2 regularization for the fully connected parameters.
  regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                  tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))

"""

# Tests:
# Filenames are shuffled
# Both queues (filename and example) work fine.
# Images are read and preprocessed correctly
# Inference works alright for 112 x 112 images
# Dropout works fine

"""
# Graph definition in Tensorboard looks okay.
% 112 x 112 with no background (sanity checks)
% See whether numbers become so small (because they always predict no) that gradients vanish (if so, I need to change the cost function)
% 112 x 112 with background
% medium image(300x300) with no background (sanity checks, including overfitting one with a single image)
% medium image with background
% large image (1000x1000) (to test memory)
% biggest possible image (1579x1305)
%  if failed (biggest image only for testing)
"""
