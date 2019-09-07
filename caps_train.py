# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 16:21:13 2017
@author: Xiangyong Cao
This code is modified based on https://github.com/KGPML/Hyperspectral
"""

from __future__ import print_function
import tensorflow as tf
import HSI_Data_Preparation
from HSI_Data_Preparation import num_train, Band, All_data, TrainIndex, TestIndex, Height, Width
from utils import patch_size, Post_Processing, safe_norm, squash
import numpy as np
import os
import scipy.io
# from cnn_model import conv_net
import time
from caps_model import CapsNet
import argparse


def compute_prob_map():
	# Obtain the probabilistic map
	num_all = len(All_data['patch'])
	times = 20
	num_each_time = int(num_all / times)
	res_num = num_all - times * num_each_time
	Num_Each_File = num_each_time * np.ones((1, times), dtype=int)
	Num_Each_File = Num_Each_File[0]
	Num_Each_File[times - 1] = Num_Each_File[times - 1] + res_num
	start = 0
	prob_map = np.zeros((1, n_classes))
	for i in range(times):
		feed_x = np.reshape(np.asarray(All_data['patch'][start:start + Num_Each_File[i]]), (-1, n_input))
		temp = sess.run(softmax_output, feed_dict={x: feed_x})
		prob_map = np.concatenate((prob_map, temp), axis=0)
		start += Num_Each_File[i]
	
	prob_map = np.delete(prob_map, (0), axis=0)
	return prob_map


start_time = time.time()

# Import HSI data
# import pickle
#
# pkfile = open("./saved_data/training.pkl", "rb")
# Training_data = pickle.load(pkfile)
# pkfile = open("./saved_data/test.pkl", "rb")
# Test_data = pickle.load(pkfile)

Training_data, Test_data = HSI_Data_Preparation.Prepare_data()
n_input = Band * patch_size * patch_size

Training_data['train_patch'] = np.transpose(Training_data['train_patch'], (0, 2, 3, 1))
Test_data['test_patch'] = np.transpose(Test_data['test_patch'], (0, 2, 3, 1))

Training_data['train_patch'] = np.reshape(Training_data['train_patch'], (-1, n_input))
Test_data['test_patch'] = np.reshape(Test_data['test_patch'], (-1, n_input))

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 10
display_step = 500
n_classes = 16

# tf Graph input
# x = tf.placeholder("float", [None, n_input])
# y = tf.placeholder("float", [None, n_classes])

x = tf.placeholder(shape=[None, n_input], dtype=tf.float32, name="X")
y = tf.placeholder(shape=[None,n_classes], dtype=tf.int64, name="y")

# construct model.
capsOutput = CapsNet(x)

# Calculate the probability.
y_prob = safe_norm(capsOutput, axis=-2, name="y_prob")
pred=y_prob
softmax_output = tf.nn.softmax(tf.squeeze(y_prob))
# Choose the predicted one.
y_prob_argmax = tf.argmax(y_prob, axis=2, name="y_predicted_argmax")
y_pred = tf.squeeze(y_prob_argmax, axis=[1, 2], name="y_pred")

# margin loss
Labels = n_classes
outputDimension = 16

# TODO: Modulize the loss function and the reconstruction process.
# Training preparation
# Define the loss function. Here we use the margin loss.
mPlus = 0.9
mMinus = 0.1
lambda_ = 0.5

# TODO: These two pramaters can be changed to fit into new datasets.

yy=y
y=tf.argmax(y,axis=-1)
T = tf.one_hot(y, depth=Labels, name="T")
y=yy
# T=y
capsOutput_norm = safe_norm(capsOutput, axis=-2, keep_dims=True, name="capsule_output_norm")

present_error_raw = tf.square(tf.maximum(0., mPlus - capsOutput_norm), name="present_error_raw")
present_error = tf.reshape(present_error_raw, shape=(-1, n_classes), name="present_error")
absent_error_raw = tf.square(tf.maximum(0., capsOutput_norm - mMinus), name="absent_error_raw")
absent_error = tf.reshape(absent_error_raw, shape=(-1, n_classes), name="absent_error")

L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error, name="L")
margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")
cost=margin_loss

'''
# Construct model
pred = conv_net(x)
softmax_output = tf.nn.softmax(pred)

# Define loss and optimizer
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
cost = tf.reduce_mean(cross_entropy)
'''

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Define accuracy
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

predict_test_label = tf.argmax(pred, 1)

# Initializing the variables
init = tf.global_variables_initializer()

x_test, y_test = Test_data['test_patch'], Test_data['test_labels']
y_test_scalar = np.argmax(y_test, 1) + 1
x_train, y_train = Training_data['train_patch'], Training_data['train_labels']

with tf.Session() as sess:
	sess.run(init)
	idx = np.random.choice(num_train, size=batch_size, replace=False)
	# Use the random index to select random images and labels.
	batch_x = Training_data['train_patch'][idx, :]
	batch_y = Training_data['train_labels'][idx, :]
	print()
	print("============")
	print(sess.run(y_prob,feed_dict={x:batch_x,y:batch_y}).shape)
	print("===========")
	print()

# Launch the graph
with tf.Session() as sess:
	sess.run(init)
	# Training cycle
	for iteration in range(training_iters):
		idx = np.random.choice(num_train, size=batch_size, replace=False)
		# Use the random index to select random images and labels.
		batch_x = Training_data['train_patch'][idx, :]
		batch_y = Training_data['train_labels'][idx, :]
		# Run optimization op (backprop) and cost op (to get loss value)
		_, batch_cost, train_acc = sess.run([optimizer, cost, accuracy],
											feed_dict={x: batch_x, y: batch_y})
		# Display logs per epoch step
		if iteration % 100 == 0:
			print("Iteraion", '%04d,' % (iteration), \
				  "Batch cost=%.4f," % (batch_cost), \
				  "Training Accuracy=%.4f" % (train_acc))
		if iteration % 1000 == 0:
			print('Training Data Eval: Training Accuracy = %.4f' % sess.run(accuracy, \
																			feed_dict={x: x_train, y: y_train}))
			print('Test Data Eval: Test Accuracy = %.4f' % sess.run(accuracy, \
																	feed_dict={x: x_test, y: y_test}))
	print("Optimization Finished!")
	
	# Test model
	print("The Final Test Accuracy is :", sess.run(accuracy, feed_dict={x: x_test, y: y_test}))
	
	# Obtain the probabilistic map
	All_data['patch'] = np.transpose(All_data['patch'], (0, 2, 3, 1))
	num_all = len(All_data['patch'])
	times = 20
	num_each_time = int(num_all / times)
	res_num = num_all - times * num_each_time
	Num_Each_File = num_each_time * np.ones((1, times), dtype=int)
	Num_Each_File = Num_Each_File[0]
	Num_Each_File[times - 1] = Num_Each_File[times - 1] + res_num
	start = 0
	prob_map = np.zeros((1, n_classes))
	for i in range(times):
		feed_x = np.reshape(np.asarray(All_data['patch'][start:start + Num_Each_File[i]]), (-1, n_input))
		temp = sess.run(softmax_output, feed_dict={x: feed_x})
		prob_map = np.concatenate((prob_map, temp), axis=0)
		start += Num_Each_File[i]
	
	prob_map = np.delete(prob_map, (0), axis=0)
	
	# MRF
	prob_map = compute_prob_map()
	Seg_Label, seg_Label, seg_accuracy = Post_Processing(prob_map, Height, Width, n_classes, y_test_scalar, TestIndex)
	
	print('The shape of prob_map is (%d,%d)' % (prob_map.shape[0], prob_map.shape[1]))
	DATA_PATH = os.getcwd()
	file_name = 'prob_map.mat'
	prob = {}
	prob['prob_map'] = prob_map
	scipy.io.savemat(os.path.join(DATA_PATH, file_name), prob)
	
	train_ind = {}
	train_ind['TrainIndex'] = TrainIndex
	scipy.io.savemat(os.path.join(DATA_PATH, 'TrainIndex.mat'), train_ind)
	
	test_ind = {}
	test_ind['TestIndex'] = TestIndex
	scipy.io.savemat(os.path.join(DATA_PATH, 'TestIndex.mat'), test_ind)
	
	end_time = time.time()
	print('The elapsed time is %.2f' % (end_time - start_time))
