# Written by: Erick Cobos T (a01184857@itesm.mx)
# Date: 15-March-2016

"""TensorFlow implementation of the convolutional network described in Chapter 3 of the thesis report.

It loads each mammogram and its label to memory, computes the function described by the convolutional network, and produces a segmentation of the same size as the original mammogram. The network outputs a heatmap of the probability of mass accross the mammogram.

The network uses separate lists of (preprocessed and augmented) mammograms for training and validation. Labels have value 0 for background, 127 for breast tissue and 255 for breast masses. 

The design follows guidelines from the TensorFlow tutorials. It uses all available CPUs and a single GPU (if available) in one machine. It is not distributed.

See specific methods for details.
"""

import tensorflow as tf

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
		image: A 3D Tensor of size [image_height, image_width, image_channels]
		label: A 3D Tensor of size [image_height, image_width, image_channels]
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
		data_dir: A string. Path to the data directory.
		queue_capacity: An integer. Maximum amount of examples that may be 
			stored in the queue.

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


def train():
	""" Creates and trains a convolutional network for image segmentation. """
	# Create a suffling queue with image and label filenames
	filename_queue = read_csv(working_dir + training_dir + training_csv)
	val_filename_queue = read_csv(working_dir + val_dir + val_csv)

	# Create an example queue. 
	# Queues prefetch data. If not needed, use new_example() directly.
	example_queue = create_example_queue(filename_queue, working_dir+training_dir)
	val_example_queue = create_example_queue(val_filename_queue, 
							     working_dir+val_dir)

	# Create model


# TODO: Create the model
# TODO: Check its graph in Tensorboard
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
	filename_queue = read_csv(working_dir + training_dir + training_csv)

	example_queue = create_example_queue(filename_queue, working_dir+training_dir)

	example = example_queue.dequeue()

	# Launch the graph.
	sess = tf.Session()
	coord = tf.train.Coordinator()

	# Start all queue_runners
	threads = tf.train.start_queue_runners(sess, coord = coord)

	# ALL EVALUATIONS HERE
	res = sess.run(example)
	  
	# When done, ask the threads to stop.
	coord.request_stop()
	coord.join(threads)
	sess.close()

	return res

# If called as 'python3 model.py' run the main method.
if __name__ == "__main__":	
	passim.
	#TODO: train()


"""
% wight initialization
tf.random_normal(mean = 0, std = ...)
%For biases maybe tf.fill (0.1,..)


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


# automatically resize
tf.image.resize_bilinear()

# use name_scope
with tf.name_scope("conv1") as scope:
	...
	conv1 = tf.nn.relu(..., name = scope.name)

# If I am gonna decrease the learning rate evry x number of steps I can do it with tf.train.exponential_decay() (it's not really exponential)

# summarize images and labels
# write a summarize function that uses tf.histogram_summary and tf.scalar_summary (sparsity see cifar model) in activations after relu and maybe in weight gradients (histogram to see if all are positive in the first and penultimate layer maybe) and maybe first layer filters (not so often though, maybe not), summarize the training and val loss, too. Summarize the reduce_mean of (predicitions) to see whether they start at around 0.5 and decrease (because there is not many positives)
# Summarize only every number of operations, loss should probably be reported every time.
# Maybe accumulate the loss function over every batch that is not printed and then print and average, that way it is probbaly smoother, or just log/summarize the loss for every batch. A batch is an image in our case
# you cna use merge_summary (instead of merge_all_summaries) to merge only a subset and save them to file.
# maybe put all sumarries in a single fuinction that returns the string with all sumaries, and then write it in the main loop. Or just when defining the graph put all sumaries in asingle function and then during trainng call tf.merge all summarie snormally.


# To save checkpoints (see Tensorflow Mechanics 101/ Tensorbord: visualizations how to)
saver = tf.train.Saver()
saver.save(sess, "tmp/model.ckpt", global_step = global_step)


# Check the MNIST example to see where to define global_step


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


#You can use tf.Session(config=tf.ConfigProto(log_device_placement=True)) to check where are operations run (CPU or GPU) or use Tensorboard.

#Multiple CPU is used automatically. Multiple GPU needs for me to do data parallelism (see cifar tutorial).

# For eval, define it in the same model or use a scope as in rnn/ptb/ptb_word_lm

# Define l2 norm as element'wise square plus reduce?sum, or as tr(AtxA)


"""

# Tests:
# Filenames are shuffled
# Both queues (filename and example) work fine.
# Images are read and preprocessed correctly

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
