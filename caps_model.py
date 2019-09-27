import tensorflow as tf
import numpy as np
from utils import squash, patch_size
from tensorflow.contrib import slim
import capslayer as cl

num_band = 103  # paviaU 103
num_classes = 9

def capnet(X):
	X=tf.reshape(X,[-1,patch_size,patch_size,num_band])
	conv_args={
		"filters":64,
		"kernel_size":4,
		"strides":1,
		"padding":"VALID",
		"activation":tf.nn.relu
	}

def conv_net(x):
	with slim.arg_scope([slim.conv2d, slim.fully_connected],
						activation_fn=tf.nn.relu):
		net = tf.reshape(x, [-1, patch_size, patch_size, num_band])
		net = slim.conv2d(net, 300, 3, padding='VALID',
						  weights_initializer=tf.contrib.layers.xavier_initializer())
		net = slim.max_pool2d(net, 2, padding='SAME')
		net = slim.conv2d(net, 200, 3, padding='VALID',
						  weights_initializer=tf.contrib.layers.xavier_initializer())
		net = slim.max_pool2d(net, 2, padding='SAME')
		net = slim.flatten(net)

		net = slim.fully_connected(net, 200)
		net = slim.fully_connected(net, 100)
		logits = slim.fully_connected(net, num_classes, activation_fn=None)
	return logits


def CapsNetWithPooling(X):
	print(X.shape[0])
	X = tf.reshape(X, [-1, patch_size, patch_size, num_band])
	# First layer, convolutional.
	conv1_params = {
		"filters": 64,
		"kernel_size": 4,
		"strides": 1,
		"padding": "valid",
		"activation": tf.nn.relu,
		"name": "conv1"
	}
	# Use 256 9*9 filters to extract features in the first conv layer.
	conv1 = tf.layers.conv2d(X, **conv1_params)
	conv1 = tf.layers.max_pooling2d(conv1, 2, strides=2, padding="same")

	# conv1_params = {
	# 	"filters": 200,
	# 	"kernel_size": 3,
	# 	"strides": 1,
	# 	"padding": "same",
	# 	"activation": tf.nn.relu,
	# 	"name": "conv1_2"
	# }
	# conv1 = tf.layers.conv2d(conv1, **conv1_params)
	# conv1 = tf.layers.max_pooling2d(conv1, 2, strides=2, padding="same")

	# Primary layer. Contains a convolution layer and the first cpasule layer.\
	# We extract 32 features and each feature will be casted to 6*6 capsules, whose dimension is [8].
	# FIXME: The caps1_caps scalar should be modified to fit into different size of data sets.
	caps1_maps = 32
	caps1_dims = 8
	conv2_params = {
		"filters": caps1_maps * caps1_dims,
		"kernel_size": 3,
		"strides": 2,
		"padding": "valid",
		"activation": tf.nn.relu,
		"name": "conv2"
	}
	conv2 = tf.layers.conv2d(conv1, **conv2_params)
	caps1_caps=np.prod(conv2.shape[1:4])
	caps1_raw = tf.reshape(conv2, [-1, caps1_caps, caps1_dims], name="caps1_raw")
	# Squash the capsules.
	caps1_output = squash(caps1_raw, name="caps1_output")

	# Digit layer
	# 10 capsule in the digit layer and each capsule outputs a vector with dimension of 16.
	caps2_caps = num_classes
	caps2_dims = 16
	init_sigma = 0.1
	w_init = tf.random_normal(
		shape=(1, caps1_caps, caps2_caps, caps2_dims, caps1_dims),
		stddev=init_sigma, dtype=tf.float32, name="W_init"
	)
	W = tf.Variable(w_init, name="W")
	# Tile the W to fit to the batch.
	batch_size = tf.shape(X)[0]
	w_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")
	# To use the tf.matmul() function, we have to make the Primary layer's dimension fit to the W_tile tensor.
	caps1_output_expanded = tf.expand_dims(caps1_output, -1, name="caps1_output_expanded1")
	caps1_output_expanded = tf.expand_dims(caps1_output_expanded, 2, name="caps1_output_expanded2")
	caps1_output_tiled = tf.tile(caps1_output_expanded, [1, 1, caps2_caps, 1, 1], name="caps1_output_tiled")
	caps2_predicted = tf.matmul(w_tiled, caps1_output_tiled, name="caps2_predicted")

	# Dynamic routing.
	# FIXME: There may be something wrong...
	# Initialize b_i,j
	raw_weights = tf.zeros([batch_size, caps1_caps, caps2_caps, 1, 1], dtype=np.float32, name="raw_weights")

	# First round.
	# Use softmax to calculate c=softmax(b).
	routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")
	# Multiply and sum
	weighted_prediction = tf.multiply(routing_weights, caps2_predicted, name="weighted_prediction")
	weighted_sum = tf.reduce_sum(weighted_prediction, axis=1, keep_dims=True, name="weighted_sum")
	caps2_output_1 = squash(weighted_sum, axis=-2, name="caps2_output_round_1")

	# Second round
	caps2_output_1_tiled = tf.tile(caps2_output_1, [1, caps1_caps, 1, 1, 1], name="caps2_1st_output_tiled")
	# Update the new b_i,j
	agreement = tf.matmul(caps2_predicted, caps2_output_1_tiled, transpose_a=True, name="agreement")
	raw_weights_round2 = tf.add(raw_weights, agreement, name="raw_weight_round_2")
	# Use softmax to calculate c=softmax(b).
	routing_weights_round2 = tf.nn.softmax(raw_weights_round2, dim=2, name="routing_weights_round_2")
	# Multiply and sum
	weighted_prediction_round2 = tf.multiply(routing_weights_round2, caps2_predicted,
											 name="weighted_prediction_round_2")
	weighted_sum_round2 = tf.reduce_sum(weighted_prediction_round2, axis=1, keep_dims=True, name="weighted_sum_round_2")
	caps2_output_2 = squash(weighted_sum_round2, axis=-2, name="caps2_output_round_2")

	# Third round
	caps2_output_3_tiled = tf.tile(caps2_output_2, [1, caps1_caps, 1, 1, 1], name="caps2_2nd_output_tiled")
	# Update the new b_i,j
	agreement = tf.matmul(caps2_predicted, caps2_output_3_tiled, transpose_a=True, name="agreement")
	raw_weights_round3 = tf.add(raw_weights_round2, agreement, name="raw_weight_round_3")
	# Use softmax to calculate c=softmax(b).
	routing_weights_round3 = tf.nn.softmax(raw_weights_round3, dim=2, name="routing_weights_round_2")
	# Multiply and sum
	weighted_prediction_round3 = tf.multiply(routing_weights_round3, caps2_predicted,
											 name="weighted_prediction_round_2")
	weighted_sum_round3 = tf.reduce_sum(weighted_prediction_round3, axis=1, keep_dims=True, name="weighted_sum_round_2")
	caps2_output_3 = squash(weighted_sum_round3, axis=-2, name="caps2_output_round_3")

	return caps2_output_3


def CapsNet(X):
	print(X.shape[0])
	X = tf.reshape(X, [-1, patch_size, patch_size, num_band])
	# First layer, convolutional.
	conv1_params = {
		"filters": 64,
		"kernel_size": 3,
		"strides": 1,
		"padding": "valid",
		"activation": tf.nn.relu,
		"name": "conv1"
	}
	# Use 256 9*9 filters to extract features in the first conv layer.
	conv1 = tf.layers.conv2d(X, **conv1_params)
	# conv1 = tf.layers.max_pooling2d(conv1,2,strides=1,padding="same")

	# Primary layer. Contains a convolution layer and the first cpasule layer.\
	# We extract 32 features and each feature will be casted to 6*6 capsules, whose dimension is [8].
	# FIXME: The caps1_caps scalar should be modified to fit into different size of data sets.
	caps1_maps = 32
	caps1_caps = caps1_maps * 3 * 3
	caps1_dims = 8
	conv2_params = {
		"filters": caps1_maps * caps1_dims,
		"kernel_size": 3,
		"strides": 2,
		"padding": "valid",
		"activation": tf.nn.relu,
		"name": "conv2"
	}
	conv2 = tf.layers.conv2d(conv1, **conv2_params)
	caps1_raw = tf.reshape(conv2, [-1, caps1_caps, caps1_dims], name="caps1_raw")
	# Squash the capsules.
	caps1_output = squash(caps1_raw, name="caps1_output")

	# Digit layer
	# 10 capsule in the digit layer and each capsule outputs a vector with dimension of 16.
	caps2_caps = num_classes
	caps2_dims = 16
	init_sigma = 0.1
	w_init = tf.random_normal(
		shape=(1, caps1_caps, caps2_caps, caps2_dims, caps1_dims),
		stddev=init_sigma, dtype=tf.float32, name="W_init"
	)
	W = tf.Variable(w_init, name="W")
	# Tile the W to fit to the batch.
	batch_size = tf.shape(X)[0]
	w_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")
	# To use the tf.matmul() function, we have to make the Primary layer's dimension fit to the W_tile tensor.
	caps1_output_expanded = tf.expand_dims(caps1_output, -1, name="caps1_output_expanded1")
	caps1_output_expanded = tf.expand_dims(caps1_output_expanded, 2, name="caps1_output_expanded2")
	caps1_output_tiled = tf.tile(caps1_output_expanded, [1, 1, caps2_caps, 1, 1], name="caps1_output_tiled")
	caps2_predicted = tf.matmul(w_tiled, caps1_output_tiled, name="caps2_predicted")

	# Dynamic routing.
	# FIXME: There may be something wrong...
	# Initialize b_i,j
	raw_weights = tf.zeros([batch_size, caps1_caps, caps2_caps, 1, 1], dtype=np.float32, name="raw_weights")

	# First round.
	# Use softmax to calculate c=softmax(b).
	routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")
	# Multiply and sum
	weighted_prediction = tf.multiply(routing_weights, caps2_predicted, name="weighted_prediction")
	weighted_sum = tf.reduce_sum(weighted_prediction, axis=1, keep_dims=True, name="weighted_sum")
	caps2_output_1 = squash(weighted_sum, axis=-2, name="caps2_output_round_1")

	# Second round
	caps2_output_1_tiled = tf.tile(caps2_output_1, [1, caps1_caps, 1, 1, 1], name="caps2_1st_output_tiled")
	# Update the new b_i,j
	agreement = tf.matmul(caps2_predicted, caps2_output_1_tiled, transpose_a=True, name="agreement")
	raw_weights_round2 = tf.add(raw_weights, agreement, name="raw_weight_round_2")
	# Use softmax to calculate c=softmax(b).
	routing_weights_round2 = tf.nn.softmax(raw_weights_round2, dim=2, name="routing_weights_round_2")
	# Multiply and sum
	weighted_prediction_round2 = tf.multiply(routing_weights_round2, caps2_predicted,
											 name="weighted_prediction_round_2")
	weighted_sum_round2 = tf.reduce_sum(weighted_prediction_round2, axis=1, keep_dims=True, name="weighted_sum_round_2")
	caps2_output_2 = squash(weighted_sum_round2, axis=-2, name="caps2_output_round_2")

	# Third round
	caps2_output_3_tiled = tf.tile(caps2_output_2, [1, caps1_caps, 1, 1, 1], name="caps2_2nd_output_tiled")
	# Update the new b_i,j
	agreement = tf.matmul(caps2_predicted, caps2_output_3_tiled, transpose_a=True, name="agreement")
	raw_weights_round3 = tf.add(raw_weights_round2, agreement, name="raw_weight_round_3")
	# Use softmax to calculate c=softmax(b).
	routing_weights_round3 = tf.nn.softmax(raw_weights_round3, dim=2, name="routing_weights_round_2")
	# Multiply and sum
	weighted_prediction_round3 = tf.multiply(routing_weights_round3, caps2_predicted,
											 name="weighted_prediction_round_2")
	weighted_sum_round3 = tf.reduce_sum(weighted_prediction_round3, axis=1, keep_dims=True, name="weighted_sum_round_2")
	caps2_output_3 = squash(weighted_sum_round3, axis=-2, name="caps2_output_round_3")

	return caps2_output_3


def CapsNet_2(X):
	# don't use this model.
	print(X.shape[0])
	X = tf.reshape(X, [-1, patch_size, patch_size, num_band])
	# First layer, convolutional.
	conv1_params = {
		"filters": 256,
		"kernel_size": 3,
		"strides": 1,
		"padding": "valid",
		"activation": tf.nn.relu,
		"name": "conv1"
	}
	# Use 256 3*3 filters to extract features in the first conv layer.
	conv1 = tf.layers.conv2d(X, **conv1_params)

	conv1_params_2 = {
		"filters": 150,
		"kernel_size": 3,
		"strides": 1,
		"padding": "same",
		"activation": tf.nn.relu,
		"name": "conv1_1"
	}
	conv1_1 = tf.layers.conv2d(conv1, **conv1_params_2)

	# Primary layer. Contains a convolution layer and the first cpasule layer.\
	# We extract 32 features and each feature will be casted to 6*6 capsules, whose dimension is [8].
	# FIXME: The caps1_caps scalar should be modified to fit into different size of data sets.
	caps1_maps = 32
	caps1_caps = caps1_maps * 3 * 3
	caps1_dims = 8
	conv2_params = {
		"filters": caps1_maps * caps1_dims,
		"kernel_size": 3,
		"strides": 2,
		"padding": "valid",
		"activation": tf.nn.relu,
		"name": "conv2"
	}
	conv2 = tf.layers.conv2d(conv1_1, **conv2_params)
	caps1_raw = tf.reshape(conv2, [batch_size, -1, caps1_dims], name="caps1_raw")
	# caps1_caps=tf.shape(caps1_raw)[1]
	# Squash the capsules.
	caps1_output = squash(caps1_raw, name="caps1_output")

	# Digit layer
	# 10 capsule in the digit layer and each capsule outputs a vector with dimension of 16.
	caps2_caps = 150
	caps2_dims = 12
	init_sigma = 0.1
	w_init = tf.random_normal(
		shape=(1, caps1_caps, caps2_caps, caps2_dims, caps1_dims),
		stddev=init_sigma, dtype=tf.float32, name="W_init"
	)
	W = tf.Variable(w_init, name="W")
	# Tile the W to fit to the batch.
	batch_size = tf.shape(X)[0]
	w_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")
	# To use the tf.matmul() function, we have to make the Primary layer's dimension fit to the W_tile tensor.
	caps1_output_expanded = tf.expand_dims(caps1_output, -1, name="caps1_output_expanded1")
	caps1_output_expanded = tf.expand_dims(caps1_output_expanded, 2, name="caps1_output_expanded2")
	caps1_output_tiled = tf.tile(caps1_output_expanded, [1, 1, caps2_caps, 1, 1], name="caps1_output_tiled")
	caps2_predicted = tf.matmul(w_tiled, caps1_output_tiled, name="caps2_predicted")

	# Dynamic routing.
	# FIXME: There may be something wrong...
	# Initialize b_i,j
	raw_weights = tf.zeros([batch_size, caps1_caps, caps2_caps, 1, 1], dtype=np.float32, name="raw_weights")

	# First round.
	# Use softmax to calculate c=softmax(b).
	routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")
	# Multiply and sum
	weighted_prediction = tf.multiply(routing_weights, caps2_predicted, name="weighted_prediction")
	weighted_sum = tf.reduce_sum(weighted_prediction, axis=1, keep_dims=True, name="weighted_sum")
	caps2_output_1 = squash(weighted_sum, axis=-2, name="caps2_output_round_1")

	# Second round
	caps2_output_1_tiled = tf.tile(caps2_output_1, [1, caps1_caps, 1, 1, 1], name="caps2_1st_output_tiled")
	# Update the new b_i,j
	agreement = tf.matmul(caps2_predicted, caps2_output_1_tiled, transpose_a=True, name="agreement")
	raw_weights_round2 = tf.add(raw_weights, agreement, name="raw_weight_round_2")
	# Use softmax to calculate c=softmax(b).
	routing_weights_round2 = tf.nn.softmax(raw_weights_round2, dim=2, name="routing_weights_round_2")
	# Multiply and sum
	weighted_prediction_round2 = tf.multiply(routing_weights_round2, caps2_predicted,
											 name="weighted_prediction_round_2")
	weighted_sum_round2 = tf.reduce_sum(weighted_prediction_round2, axis=1, keep_dims=True, name="weighted_sum_round_2")
	caps2_output_2 = squash(weighted_sum_round2, axis=-2, name="caps2_output_round_2")

	# Third round
	caps2_output_3_tiled = tf.tile(caps2_output_2, [1, caps1_caps, 1, 1, 1], name="caps2_2nd_output_tiled")
	# Update the new b_i,j
	agreement = tf.matmul(caps2_predicted, caps2_output_3_tiled, transpose_a=True, name="agreement")
	raw_weights_round3 = tf.add(raw_weights_round2, agreement, name="raw_weight_round_3")
	# Use softmax to calculate c=softmax(b).
	routing_weights_round3 = tf.nn.softmax(raw_weights_round3, dim=2, name="routing_weights_round_2")
	# Multiply and sum
	weighted_prediction_round3 = tf.multiply(routing_weights_round3, caps2_predicted,
											 name="weighted_prediction_round_2")
	weighted_sum_round3 = tf.reduce_sum(weighted_prediction_round3, axis=1, keep_dims=True, name="weighted_sum_round_2")
	caps2_output_3 = tf.squeeze(squash(weighted_sum_round3, axis=-2, name="caps2_output_round_3"))

	# another capslayer.
	caps3_caps = num_classes
	caps3_dims = 16
	init_sigma = 0.1
	w_2_init = tf.random_normal(
		shape=(1, caps2_caps, caps3_caps, caps3_dims, caps2_dims),
		stddev=init_sigma, dtype=tf.float32, name="W_init2"
	)
	W_2 = tf.Variable(w_2_init, name="W2")
	# Tile the W to fit to the batch.
	batch_size = tf.shape(X)[0]
	w_2_tiled = tf.tile(W_2, [batch_size, 1, 1, 1, 1], name="W_tiled2")
	# caps2_output_expanded=caps2_output_3
	# To use the tf.matmul() function, we have to make the Primary layer's dimension fit to the W_tile tensor.
	caps2_output_expanded = tf.expand_dims(caps2_output_3, -1, name="caps1_output_expanded12")
	caps2_output_expanded = tf.expand_dims(caps2_output_expanded, 2, name="caps1_output_expanded22")
	caps2_output_tiled = tf.tile(caps2_output_expanded, [1, 1, caps3_caps, 1, 1], name="caps1_output_tiled2")
	print()
	print(tf.shape(caps2_output_tiled))
	print()
	caps3_predicted = tf.matmul(w_2_tiled, caps2_output_tiled, name="caps2_predicted2")

	# Dynamic routing.
	# FIXME: There may be something wrong...
	# Initialize b_i,j
	raw_weights = tf.zeros([batch_size, caps2_caps, caps3_caps, 1, 1], dtype=np.float32, name="raw_weights2")

	# First round.
	# Use softmax to calculate c=softmax(b).
	routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights2")
	# Multiply and sum
	weighted_prediction = tf.multiply(routing_weights, caps3_predicted, name="weighted_prediction2")
	weighted_sum = tf.reduce_sum(weighted_prediction, axis=1, keep_dims=True, name="weighted_sum2")
	caps2_output_1 = squash(weighted_sum, axis=-2, name="caps2_output_round_12")

	# Second round
	caps2_output_1_tiled = tf.tile(caps2_output_1, [1, caps2_caps, 1, 1, 1], name="caps2_1st_output_tiled2")
	# Update the new b_i,j
	agreement = tf.matmul(caps3_predicted, caps2_output_1_tiled, transpose_a=True, name="agreement2")
	raw_weights_round2 = tf.add(raw_weights, agreement, name="raw_weight_round_22")
	# Use softmax to calculate c=softmax(b).
	routing_weights_round2 = tf.nn.softmax(raw_weights_round2, dim=2, name="routing_weights_round_22")
	# Multiply and sum
	weighted_prediction_round2 = tf.multiply(routing_weights_round2, caps3_predicted,
											 name="weighted_prediction_round_22")
	weighted_sum_round2 = tf.reduce_sum(weighted_prediction_round2, axis=1, keep_dims=True,
										name="weighted_sum_round_22")
	caps2_output_2 = squash(weighted_sum_round2, axis=-2, name="caps2_output_round_22")

	# Third round
	caps2_output_3_tiled = tf.tile(caps2_output_2, [1, caps2_caps, 1, 1, 1], name="caps2_2nd_output_tiled2")
	# Update the new b_i,j
	agreement = tf.matmul(caps3_predicted, caps2_output_3_tiled, transpose_a=True, name="agreement2")
	raw_weights_round3 = tf.add(raw_weights_round2, agreement, name="raw_weight_round_32")
	# Use softmax to calculate c=softmax(b).
	routing_weights_round3 = tf.nn.softmax(raw_weights_round3, dim=2, name="routing_weights_round_22")
	# Multiply and sum
	weighted_prediction_round3 = tf.multiply(routing_weights_round3, caps3_predicted,
											 name="weighted_prediction_round_22")
	weighted_sum_round3 = tf.reduce_sum(weighted_prediction_round3, axis=1, keep_dims=True,
										name="weighted_sum_round_22")
	caps2_output_3 = squash(weighted_sum_round3, axis=-2, name="caps2_output_round_32")

	return caps2_output_3


def CapsNet_3(X):
	print(X.shape[0])
	X = tf.reshape(X, [-1, patch_size, patch_size, num_band])
	# First layer, convolutional.
	conv1_params = {
		"filters": 64,
		"kernel_size": 3,
		"strides": 1,
		"padding": "valid",
		"activation": tf.nn.relu,
		"name": "conv1"
	}
	# Use 256 3*3 filters to extract features in the first conv layer.
	conv1 = tf.layers.conv2d(X, **conv1_params)

	conv1_params_2 = {
		"filters": 128,
		"kernel_size": 3,
		"strides": 1,
		"padding": "same",
		"activation": tf.nn.relu,
		"name": "conv1_1"
	}
	conv1_1 = tf.layers.conv2d(conv1, **conv1_params_2)

	conv1_params_3 = {
		"filters": 256,
		"kernel_size": 3,
		"strides": 1,
		"padding": "same",
		"activation": tf.nn.relu,
		"name": "conv1_2"
	}
	conv1_2 = tf.layers.conv2d(conv1_1, **conv1_params_3)

	# Primary layer. Contains a convolution layer and the first cpasule layer.\
	# We extract 32 features and each feature will be casted to 6*6 capsules, whose dimension is [8].
	# FIXME: The caps1_caps scalar should be modified to fit into different size of data sets.
	caps1_maps = 32
	caps1_caps = caps1_maps * 3 * 3
	caps1_dims = 8
	conv2_params = {
		"filters": caps1_maps * caps1_dims,
		"kernel_size": 3,
		"strides": 2,
		"padding": "valid",
		"activation": tf.nn.relu,
		"name": "conv2"
	}
	conv2 = tf.layers.conv2d(conv1_2, **conv2_params)
	caps1_raw = tf.reshape(conv2, [-1, caps1_caps, caps1_dims], name="caps1_raw")
	# caps1_caps=tf.shape(caps1_raw)[1]
	# Squash the capsules.
	caps1_output = squash(caps1_raw, name="caps1_output")

	# Digit layer
	# 10 capsule in the digit layer and each capsule outputs a vector with dimension of 16.
	caps2_caps = num_classes
	caps2_dims = 16
	init_sigma = 0.1
	w_init = tf.random_normal(
		shape=(1, caps1_caps, caps2_caps, caps2_dims, caps1_dims),
		stddev=init_sigma, dtype=tf.float32, name="W_init"
	)
	W = tf.Variable(w_init, name="W")
	# Tile the W to fit to the batch.
	batch_size = tf.shape(X)[0]
	w_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")
	# To use the tf.matmul() function, we have to make the Primary layer's dimension fit to the W_tile tensor.
	caps1_output_expanded = tf.expand_dims(caps1_output, -1, name="caps1_output_expanded1")
	caps1_output_expanded = tf.expand_dims(caps1_output_expanded, 2, name="caps1_output_expanded2")
	caps1_output_tiled = tf.tile(caps1_output_expanded, [1, 1, caps2_caps, 1, 1], name="caps1_output_tiled")
	caps2_predicted = tf.matmul(w_tiled, caps1_output_tiled, name="caps2_predicted")

	# Dynamic routing.
	# FIXME: There may be something wrong...
	# Initialize b_i,j
	raw_weights = tf.zeros([batch_size, caps1_caps, caps2_caps, 1, 1], dtype=np.float32, name="raw_weights")

	# First round.
	# Use softmax to calculate c=softmax(b).
	routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")
	# Multiply and sum
	weighted_prediction = tf.multiply(routing_weights, caps2_predicted, name="weighted_prediction")
	weighted_sum = tf.reduce_sum(weighted_prediction, axis=1, keep_dims=True, name="weighted_sum")
	caps2_output_1 = squash(weighted_sum, axis=-2, name="caps2_output_round_1")

	# Second round
	caps2_output_1_tiled = tf.tile(caps2_output_1, [1, caps1_caps, 1, 1, 1], name="caps2_1st_output_tiled")
	# Update the new b_i,j
	agreement = tf.matmul(caps2_predicted, caps2_output_1_tiled, transpose_a=True, name="agreement")
	raw_weights_round2 = tf.add(raw_weights, agreement, name="raw_weight_round_2")
	# Use softmax to calculate c=softmax(b).
	routing_weights_round2 = tf.nn.softmax(raw_weights_round2, dim=2, name="routing_weights_round_2")
	# Multiply and sum
	weighted_prediction_round2 = tf.multiply(routing_weights_round2, caps2_predicted,
											 name="weighted_prediction_round_2")
	weighted_sum_round2 = tf.reduce_sum(weighted_prediction_round2, axis=1, keep_dims=True, name="weighted_sum_round_2")
	caps2_output_2 = squash(weighted_sum_round2, axis=-2, name="caps2_output_round_2")

	# Third round
	caps2_output_3_tiled = tf.tile(caps2_output_2, [1, caps1_caps, 1, 1, 1], name="caps2_2nd_output_tiled")
	# Update the new b_i,j
	agreement = tf.matmul(caps2_predicted, caps2_output_3_tiled, transpose_a=True, name="agreement")
	raw_weights_round3 = tf.add(raw_weights_round2, agreement, name="raw_weight_round_3")
	# Use softmax to calculate c=softmax(b).
	routing_weights_round3 = tf.nn.softmax(raw_weights_round3, dim=2, name="routing_weights_round_2")
	# Multiply and sum
	weighted_prediction_round3 = tf.multiply(routing_weights_round3, caps2_predicted,
											 name="weighted_prediction_round_2")
	weighted_sum_round3 = tf.reduce_sum(weighted_prediction_round3, axis=1, keep_dims=True, name="weighted_sum_round_2")
	caps2_output_3 = squash(weighted_sum_round3, axis=-2, name="caps2_output_round_3")

	return caps2_output_3


def reconstruct(capsOutput, mask_with_labels, X, y, y_pred, Labels, outputDimension):
	# A normal 3-layer fully connected neuron network.
	# Mask
	# mask_with_labels = tf.placeholder_with_default(False, shape=(), name="mask_with_labels")
	reconstruction_target = tf.cond(mask_with_labels, lambda: y, lambda: y_pred, name="reconstruction_target")
	reconstruction_mask = tf.one_hot(reconstruction_target, depth=Labels, name="reconstruction_mask")
	reconstruction_mask_reshaped = tf.reshape(
		reconstruction_mask, [-1, 1, Labels, 1, 1],
		name="reconstruction_mask_reshaped")
	capsOutput_masked = tf.multiply(
		capsOutput, reconstruction_mask_reshaped,
		name="caps2_output_masked")
	decoder_input = tf.reshape(capsOutput_masked, [-1, Labels * outputDimension])

	# Decoder
	# Tow relu and a sigmoid
	n_hidden1 = 512
	n_hidden2 = 1024
	n_output = 28 * 28

	with tf.name_scope("decoder"):
		hidden1 = tf.layers.dense(decoder_input, n_hidden1, activation=tf.nn.relu, name="hidden1")
		hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
		decoder_output = tf.layers.dense(hidden2, n_output, activation=tf.nn.sigmoid, name="decoder_output")

	# Reconstruction loss.
	X_flat = tf.reshape(X, [-1, n_output], name="X_flat")
	squared_difference = tf.square(X_flat - decoder_output, name="squared_difference")
	reconstruction_loss = tf.reduce_mean(squared_difference, name="reconstruction_loss")

	return reconstruction_loss, decoder_output
