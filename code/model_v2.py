# Written by: Erick Cobos T (a01184857@itesm.mx)
# Date: April-2016
# Modified: October-2016
""" TensorFlow implementation of the convolutional network described in Ch. 3
(Experiment 2) of the thesis report.

The network outputs a heatmap of logits indicating the probability of mass
accross the mammogram. Labels have value 0 for background, 127 for breast tissue
and 255 for breast masses. The loss function weights errors on breast masses by
15, on normal breast tissue by 1 and on background by 0 (ignored) to encourage
the network to learn better features to correctly distinguish lesions.

Works for Tensorflow 0.11.rc0
"""
import tensorflow as tf

def forward(image, drop):
	""" A fully convolutional network for image segmentation.

	The architecture is modelled as a small VGG-16 network. It has approximately
	2.9 million parameters. Uses mirror padding to avoid artifacts

	Architecture:
		INPUT -> [[CONV -> Leaky RELU]*2 -> MAXPOOL]*2 -> [CONV -> Leaky RELU]*3
		-> MAXPOOL -> FC -> Leaky RELU -> FC -> SIGMOID -> BILINEAR
	Input size: 112 x 112
	Downsampling size (before BILINEAR): 7 x 7 
	Output size: 112 x 112 (16x upsampling)
	Effective receptive field: 184 x 184

	Args:
		image: A tensor with shape [height, width, channels]. The input image
		drop: A boolean. If True, dropout is active.

	Returns:
		prediction: A tensor of floats with shape [height, width]: the predicted 
		segmentation (a heatmap of logits).
	"""
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

	def conv_op(input, filter_shape, strides):
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

	def leaky_relu(x, alpha=0.1):
		""" Leaky ReLU activation function."""
		with tf.name_scope('leaky_relu'):
			output = tf.maximum(tf.mul(alpha, x), x)
		return output

	def dropout(x, keep_prob):
		""" During training, performs dropout. Otherwise, returns original."""
		output = tf.cond(drop, lambda: tf.nn.dropout(x, keep_prob), lambda: x)		
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
		scope = tf.get_default_graph()._name_stack # No easier way
		tf.histogram_summary(scope + '/activations', output)
		
		return output

	batch = tf.expand_dims(image, 0)	# batch with a single image
	
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
	
def loss(prediction, label):
	""" Logistic loss function averaged over pixels in the breast area.
	
	Errors are weighted according to where they occur: on masses by 15, on 
	normal breast tissue by 1 and on the background by zero (ignored).
	
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
		tissue = tf.to_float(tf.equal(label, 127))
		breast_area = mass + tissue
		
		# Compute loss per pixel
		pixel_loss = tf.nn.sigmoid_cross_entropy_with_logits(prediction, mass)
	
		# Weight the errors
		weighted_loss = 15 * tf.mul(pixel_loss, mass)
		weighted_loss += 1 * tf.mul(pixel_loss, tissue)
	
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
