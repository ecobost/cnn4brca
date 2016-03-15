# Written by: Erick Cobos T (a01184857@itesm.mx)
# Date: 15-March-2016

"""TensorFlow implementation of the convolutional network described in Chapter 3 of the thesis report.

It loads each mammogram and its label to memory, computes the function described by the convolutional network, and produces a segmentation of the same size as the original mammogram. The network outputs a heatmap of the probability of mass accross the mammogram.

The network uses separate lists of (preprocessed and augmented) mammograms for training and validation. Labels have value 0 for background, 127 for breast tissue and 255 for breast masses. 

The design is loosely based on the examples offered in the TensorFlow tutorials. It uses all available CPUs and a single GPU (if available) in one machine. It is not distributed.

See code for details.
"""
import tensorflow as tf
import csv

"""
Pseudo-code
Declare some constants
Create queue of image and labels filenames (for trainnig and validation) (create a function)
Define the model
	Create each layer(maybe with a function)
Define the loss for the model
Define the optimization
Add summaries
Start session and initialize variables, writer and queues
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

# TODO: Create the filename queues, see if it works
# TODO: Create the preprocessing image queues (nextBatch)
# TODO: Create the model
# TODO: Check its graph in Tensorboard

# Set some parameters
training_dir = "" # Path to the training directory where results are saved
training_path = "" # Path to the csv file holding the image and label filenames
val_file = ""
#

# If called as 'python3 model.py' run the main method.
if __name__ == "__main__"	
	main()



"""
% wight initialization
tf.random_normal(mean = 0, std = ...)
%For biases maybe tf.fill (0.1,..)

%Zero-mean image
tf.image.per_image_whitening()

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


# Check this answer for reading images
http://stackoverflow.com/questions/34340489/tensorflow-read-images-with-labels?rq=1
#Actually, it ays exactly how to do it here: https://www.tensorflow.org/versions/r0.7/how_tos/reading_data/index.html#reading-data
with open('training.csv', 'r') as f:
    lines = f.readlines()
filename_queue = string_input_producer(lines) #give it all filenames as tensor strings in csv format, it will automatically shuffle them between every epoch and cycle thrugh them, then read em with.
image_filename, label_filename = tf.decode_csv(filename_queue.dequeue(), [[""], [""]]) # I could decode this with Python
string_image = tf.read_file(imageFilename)
string_label = tf.read_file(labelFilename)
image = tf.image.decode_png(string_image)
label = tf.image.decode_png(string_label)
return image, label
Preprocessing
Here I could directly use image, label or use tf.batch with capacity 5 to prefetch a number of examples (5) (create an example_queue). If i don't use this, will it be slower (maybe it creates a example queue no matter what)?. Time how long does sess.run([image, label]) takes with and without it. for 100 steps for instance. I would alos have to do something else so the threads in the background have time to reload. maybe just do marix multiplications.

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
		

		sess.run(optimizer.minimize(loss))

		summary_str = sess.run(tf.merge_all_summaries(), feed_dict=feed_dict)
		summary_writer.add_summary(summary_str, step)

# Create the feed before the sess.run
feed = {x: batch_xs, y_: batch_ys}
sess.run(train_step, feed_dict = feed)


#You can use tf.Session(config=tf.ConfigProto(log_device_placement=True)) to check where are operations run (CPU or GPU) or use Tensorboard.

#Multiple CPU is used automatically. Multiple GPU needs for me to do data parallelism (see cifar tutorial).

#If possible, make it so it can use variable batch_sizes (for inference in various images), by not defining batch_size as a constant. If i need it I could do batch_size = tf.shape(input)[0], and levaing the first dimension (the batch_size) in placeholder as None.

# For eval, define it in the same model or use a scope as in rnn/ptb/ptb_word_lm

# Define l2 norm as element'wise square plus reduce?sum, or as tr(AtxA)

# Tests
% Tensorboard graph definition looks fine?
% 112 x 112 with no background (sanity checks)
% See whether numbers become so small (because they always predict no) that gradients vanish (if so, I need to change the cost function)
% 112 x 112 with background
% medium image(300x300) with no background (sanity checks, including overfitting one with a single image)
% medium image with background
% large image (1000x1000) (to test memory)
% biggest possible image (1579x1305)
%  if failed (biggest image only for testing)

"""
