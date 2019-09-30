import tensorflow as tf
import numpy as np
from utils import squash, patch_size
# from tensorflow.contrib import slim
import capslayer as cl

num_band = 220  # paviaU 103
num_classes = 16


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

	# Primary layer. Contains a convolution layer and the first cpasule layer.\
	# We extract 32 features and each feature will be casted to 6*6 capsules, whose dimension is [8].
	# FIXME: The caps1_caps scalar should be modified to fit into different size of data sets.
	caps1_maps = 32
	caps1_caps = caps1_maps
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
	# caps1_caps=np.prod(tf.squeeze(conv2.shape[1:4]))
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


def CapsNetWithPoolingAndBN(X):
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

	# Primary layer. Contains a convolution layer and the first cpasule layer.\
	# We extract 32 features and each feature will be casted to 6*6 capsules, whose dimension is [8].
	# FIXME: The caps1_caps scalar should be modified to fit into different size of data sets.
	caps1_maps = 32
	caps1_caps = caps1_maps
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
	# caps1_caps=np.prod(tf.squeeze(conv2.shape[1:4]))
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


def caps_net(x):
	net = tf.reshape(x, [-1, patch_size, patch_size, num_band])
	conv1 = tf.layers.conv2d(
		net,
		filters=32,
		kernel_size=3,
		strides=1,
		padding="VALID",
		activation=tf.nn.relu,
		name="convLayer"
	)

	convCaps, activation = cl.layers.primaryCaps(
		conv1,
		filters=32,
		kernel_size=3,
		strides=1,
		out_caps_dims=[8, 1],
		method="logistic"
	)

	n_input = np.prod(cl.shape(convCaps)[1:4])
	convCaps = tf.reshape(convCaps, shape=[-1, n_input, 8, 1])
	activation = tf.reshape(activation, shape=[-1, n_input])

	rt_poses, rt_probs = cl.layers.dense(
		convCaps,
		activation,
		num_outputs=num_classes,
		out_caps_dims=[16, 1],
		routing_method="DynamicRouting"
	)
	return rt_probs


def caps_net_mod(x):
	net = tf.reshape(x, [-1, patch_size, patch_size, num_band])
	conv1 = tf.layers.conv2d(
		net,
		filters=100,
		kernel_size=3,
		strides=1,
		padding="same",
		activation=tf.nn.relu,
		name="convLayer"
	)
	conv1 = tf.layers.max_pooling2d(conv1, 2, strides=2, padding="same")

	conv2 = tf.layers.conv2d(
		conv1,
		filters=300,
		kernel_size=3,
		padding="same",
		activation=tf.nn.relu
	)
	conv2 = tf.layers.max_pooling2d(conv2, 2, strides=2, padding="same")

	convCaps, activation = cl.layers.primaryCaps(
		conv2,
		filters=64,
		kernel_size=3,
		strides=1,
		out_caps_dims=[8, 1],
		method="logistic"
	)

	n_input = np.prod(cl.shape(convCaps)[1:4])
	convCaps = tf.reshape(convCaps, shape=[-1, n_input, 8, 1])
	activation = tf.reshape(activation, shape=[-1, n_input])

	rt_poses, rt_probs = cl.layers.dense(
		convCaps,
		activation,
		num_outputs=num_classes,
		out_caps_dims=[16, 1],
		routing_method="DynamicRouting"
	)
	return rt_probs
