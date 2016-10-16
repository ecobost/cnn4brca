# Written by: Erick Cobos T (a01184857@itesm.mx)
# Date: Oct-2016
""" TensorFlow implementation of the convolutional network described in Ch. 3
(Experiment 1) of the thesis report.

The network outputs a heatmap of logits indicating the probability of mass 
accross the mammogram. Labels have value 0 for background, 127 for breast tissue
and 255 for breast masses (the positive class).

Works for Tensorflow 0.11.rc0
"""
import tensorflow as tf

def forward(image, drop):
	""" A convolutional network for image segmentation.

	Small network (7 layers, 206 K parameters), uses strided convolutions to 
	replace pooling and dilated convolutions in the last layers. It also mirrors
	the image on the edges to avoid artifacts.

	Input size: 52 x 52
	Downsampling size (before BILINEAR): 13 x 13
	Output size: 52 x 52 (4x upsampling)
	Effective receptive field: 101 x 101

	Args:
		image: A tensor with shape [height, width, channels]. The input image
		drop: A boolean. Un

	Returns:
		prediction: A tensor of floats with shape [height, width]: the predicted 
		segmentation (a heatmap of logits).
	"""
	# Define some local functions
	def initialize_weights(filter_shape):
		""" Initializes filter weights with random values.

		Values drawn from a normal distribution with zero mean and standard 
		deviation = sqrt(2/n_in) where n_in is the number of connections to the 
		filter, e.g., 90 for a 3x3 filter with depth 10.
		"""
		n_in = filter_shape[0] * filter_shape[1] * filter_shape[2]
		values = tf.random_normal(filter_shape, 0, tf.sqrt(2/n_in))
		
		return values

	def pad_conv_input(input, filter_shape, strides):
		""" Pads a batch mirroring the edges of each feature map.
		
		Calculates the amount of padding needed to preserve spatial dimensions
		and uses mirror padding on each feature map.
		"""
		# Calculate the amount of padding (as done when padding=SAME)
		
		# Proper way (works for any input, filter, stride combination)
		#input_shape = tf.shape(input)
		#in_height = input_shape[1]
		#in_width = input_shape[2]
		#out_height = tf.to_int32(tf.ceil(in_height / strides[1]))
		#out_width  = tf.to_int32(tf.ceil(in_width / strides[2]))
		#pad_height = (out_height - 1) * strides[1] + filter_shape[0] - in_height
		#pad_width = (out_width - 1) * strides[2] + filter_shape[1] - in_width

		# Easier way (works if height and width are divisible by their stride)
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
		""" Creates filters and biases, and performs a convolution."""
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
		"""Pads a batch for atrous convolution mirroring feature maps' edges."""
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
		""" Creates filters and biases, and performs a dilated convolution."""
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
		
	# Create a batch with a single image
	batch = tf.expand_dims(image, 0)	
	
	# Define the architecture
	with tf.name_scope('conv1'):
		conv1 = tf.nn.relu(conv_op(batch, [5, 5, 1, 32], [1, 2, 2, 1]))
	with tf.name_scope('conv2'):
		conv2 = tf.nn.relu(conv_op(conv1, [3, 3, 32, 32]))

	with tf.name_scope('conv3'):
		conv3 = tf.nn.relu(conv_op(conv2, [3, 3, 32, 64], [1, 2, 2, 1]))
	with tf.name_scope('conv4'):
		conv4 = tf.nn.relu(conv_op(conv3, [3, 3, 64, 64]))

	with tf.name_scope('conv5'):
		conv5 = tf.nn.relu(atrous_conv_op(conv4, [3, 3, 64, 96], dilation=2)) 
	with tf.name_scope('conv6'):
		conv6 = tf.nn.relu(atrous_conv_op(conv5, [3, 3, 96, 96], dilation=2)) 
		
	with tf.name_scope('fc'):
		fc = atrous_conv_op(conv6, [5, 5, 96, 1], dilation=3)
	
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
	tf.histogram_summary('fc/activations', fc)
	
	# Unwrap segmentation
	prediction = tf.squeeze(output)	

	return prediction

def loss(prediction, label):
	""" Logistic loss function averaged over pixels in the breast area.
	
	Pixels in the background are ignored.
	
	Args:
		prediction: A tensor of floats with shape [height, width]. The predicted
			heatmap of logits.
		label: A tensor of integers with shape [height, width]. Labels are 0 
			(background), 127 (breast tissue) and 255 (breast mass).

	Returns:
		A float: The loss.
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
	""" Calculates the l2 regularization loss from the collected weights.
	
	Returns:
		A float: The loss
	"""
	with tf.name_scope("regularization_loss"):
		# Compute the (halved and squared) l2-norm of each weight matrix
		weights = tf.get_collection(tf.GraphKeys.WEIGHTS)
		l2_losses = [tf.nn.l2_loss(x) for x in weights]
		
		# Add all regularization losses
		loss = tf.add_n(l2_losses)
		
	return loss
	
def update_weights(loss, learning_rate):
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
