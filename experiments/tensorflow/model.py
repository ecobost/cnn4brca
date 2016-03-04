
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

For loading, maybe load all names in a list and do a nextIMage that retursn a tuple image, label
next image will take care of reshuffling and lading

# automatically resize
tf.image.resize_bilinear(

# How to write somethings (rather than how they say to)
optimizer = tf.train.ADAMOptimizer


with Session as sess:
	sess.run(tf.initiallize_all_variables)
	summary_writer = tf.train.SummaryWriter(FLAGS.train_dir)
	summary_writer.add_graph(sess.graph_def)

	for i in steps
		sess.run(optimizer.minimize(loss))
		summary_str = sess.run(tf.merge_all_summaries, feed_dict=feed_dict)



# Tests
% 112 x 112 with no background (sanity checks)
% See whether numbers become so small (because they always predict no) that gradients vanish (if so, I need to change the cost function)
% 112 x 112 with background
% medium image(300x300) with no background (sanity checks, including overfitting one with a single image)
% medium image with background
% large image (1000x1000) (to test memory)
% biggest possible image (1579x1305)
%  if failed (biggest image only for testing)

