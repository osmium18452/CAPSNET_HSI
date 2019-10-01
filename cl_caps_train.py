from __future__ import print_function
import tensorflow as tf
import HSI_Data_Preparation
from HSI_Data_Preparation import num_train, Band, All_data, TrainIndex, TestIndex, Height, Width, num_test
from utils import patch_size, Post_Processing, safe_norm, squash
import numpy as np
import os
import scipy.io
# from cnn_model import conv_net
import time
import capslayer as cl
from caps_model import caps_net, caps_net_mod
import argparse
import pickle


def normalize(rt):
	ans = np.zeros((1, rt.shape[1]))
	for item in rt:
		sum = 0.
		for i in item:
			sum += i
		ans = np.concatenate((ans, np.reshape(item / sum, (1, -1))), axis=0)
	rtn = np.delete(ans, (0), axis=0)
	return rtn


parser = argparse.ArgumentParser(description="Capsule Network on MNIST")
parser.add_argument("-e", "--epochs", default=100, type=int,
					help="The number of epochs you want to train.")
parser.add_argument("-g", "--gpu", default="0", type=str,
					help="Which gpu(s) you want to use.")
parser.add_argument("-d", "--directory", default="./saved_model",
					help="The directory you want to save your model.")
parser.add_argument("-b", "--batch", default=100, type=int,
					help="Set batch size.")
parser.add_argument("-m", "--model", default=1, type=int,
					help="Use which model to train and predict.")
parser.add_argument("-r", "--restore", default=False,
					help="Restore the trained model or not. True or False")
parser.add_argument("-c", "--cost", default="margin",
					help="Use margin loss or cross entropy as loss function. 'margin' for margin loss or 'cross' for cross entropy.")
parser.add_argument("-l", "--lr", default=0.0001, type=float,
					help="learning rate")
parser.add_argument("-t", "--predict_times", default=400, type=int,
					help="times you want to predict. smaller times causes bigger batches.")
parser.add_argument("-o", "--optimizer", default="adam", type=str,
					help="use adam optimizer or sgd optmizer")
parser.add_argument("-p", "--predict_batch", default=100, type=int,
					help="predict batch size")
args = parser.parse_args()

print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

start_time = time.time()

Training_data, Test_data = HSI_Data_Preparation.Prepare_data()
n_input = Band * patch_size * patch_size

Training_data['train_patch'] = np.transpose(Training_data['train_patch'], (0, 2, 3, 1))
Test_data['test_patch'] = np.transpose(Test_data['test_patch'], (0, 2, 3, 1))

Training_data['train_patch'] = np.reshape(Training_data['train_patch'], (-1, n_input))
Test_data['test_patch'] = np.reshape(Test_data['test_patch'], (-1, n_input))

# Parameters
learning_rate = args.lr
batch_size = args.batch
display_step = 500
n_classes = 16

x = tf.placeholder(shape=[None, n_input], dtype=tf.float32, name="X")
y = tf.placeholder(shape=[None, n_classes], dtype=tf.float32, name="y")

# construct model.
if args.model == 1:
	pred = caps_net(x)
else:
	pred = caps_net_mod(x)

margin_loss = cl.losses.margin_loss(y, pred)
cost = tf.reduce_mean(margin_loss)

if args.optimizer == "adam":
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	print("adam used")
else:
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
	print("sgd used")
# Define accuracy
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

predict_test_label = tf.argmax(pred, 1)

# Initializing the variables
init = tf.global_variables_initializer()

x_test, y_test = Test_data['test_patch'], Test_data['test_labels']
y_test_scalar = np.argmax(y_test, 1) + 1
x_train, y_train = Training_data['train_patch'], Training_data['train_labels']

if not os.path.exists(args.directory):
	os.makedirs(args.directory)
save_path = os.path.join(args.directory, "saved_model.ckpt")

# Launch the graph
saver = tf.train.Saver()
with tf.Session() as sess:
	if args.restore and os.path.exists(args.directory):
		saver.restore(sess, save_path)
		print()
		print("model restored.")
		print()
	else:
		sess.run(init)
	# Training cycle
	
	for epoch in range(args.epochs):
		
		# mix the training data.
		if epoch % 5 == 0:
			permutation = np.random.permutation(Training_data["train_patch"].shape[0])
			Training_data["train_patch"] = Training_data["train_patch"][permutation, :]
			Training_data["train_labels"] = Training_data["train_labels"][permutation, :]
			permutation = np.random.permutation(Test_data["test_patch"].shape[0])
			Test_data["test_patch"] = Test_data["test_patch"][permutation, :]
			Test_data["test_labels"] = Test_data["test_labels"][permutation, :]
			print("randomized")
		
		iters = num_train // batch_size
		for iter in range(iters):
			batch_x = Training_data['train_patch'][iter * batch_size:(iter + 1) * batch_size, :]
			batch_y = Training_data['train_labels'][iter * batch_size:(iter + 1) * batch_size, :]
			_, batch_cost, train_acc = sess.run([optimizer, cost, accuracy], feed_dict={x: batch_x, y: batch_y})
			print(
				"\repochs:{:3d}  batch:{:4d}/{:4d}  accuracy:{:.6f}  cost:{:.6f}".format(epoch + 1, iter + 1, iters,
																						 train_acc,
																						 batch_cost), end="")
		
		if num_train % batch_size != 0:
			batch_x = Training_data['train_patch'][iters * batch_size:, :]
			batch_y = Training_data['train_labels'][iters * batch_size:, :]
			_, batch_cost, train_acc = sess.run([optimizer, cost, accuracy], feed_dict={x: batch_x, y: batch_y})
		# print(
		# 	"\repochs:{:3d}  batch:{:4d}/{:4d}  accuracy:{:.6f}  cost:{:.6f}".format(epoch, iters, iters, train_acc,
		# 																			 batch_cost), end="")
		saver.save(sess, save_path=save_path)
		print("\nmodel saved")
		
		idx = np.random.choice(num_test, size=batch_size, replace=False)
		# Use the random index to select random images and labels.
		test_batch_x = Test_data['test_patch'][idx, :]
		test_batch_y = Test_data['test_labels'][idx, :]
		ac, cs, pr, ory = sess.run([accuracy, cost, pred, y], feed_dict={x: test_batch_x, y: test_batch_y})
		print('Test Data Eval: Test Accuracy = %.4f, Test Cost =%.4f' % (ac, cs))
		
		pr = normalize(pr)
		
		for ii in pr[1]:
			print("%.6f" % ii, end=" ")
		print()
		for ii in ory[1]:
			print("%8d" % ii, end=" ")
		print()
	
	print("optimization finished!")

with tf.Session() as sess:
	saver.restore(sess, save_path)
	
	print("==========training data===========")
	arr, rst = sess.run([pred, y],
						feed_dict={x: Training_data["train_patch"][0:20, :], y: Training_data["train_labels"][0:20, :]})
	
	arr = normalize(arr)
	
	for i in range(20):
		for item in arr[i]:
			print("%.6f" % item, end=" ")
		print()
		for item in rst[i]:
			print("%8d" % item, end=" ")
		print()
	print()
	
	print("==========test data===========")
	# print(Test_data["test_patch"])
	arr, rst = sess.run([pred, y],
						feed_dict={x: Test_data["test_patch"][0:20, :], y: Test_data["test_labels"][0:20, :]})
	for i in range(20):
		for item in arr[i]:
			print("%.6f" % item, end=" ")
		print()
		for item in rst[i]:
			print("%8d" % item, end=" ")
		print()
	print()
	
	# Obtain the probabilistic map
	num_all = len(All_data["patch"])
	pred_times = num_all // args.predict_batch
	prob_map = np.zeros((1, n_classes))
	for i in range(pred_times):
		feedx = np.reshape(np.asarray(All_data["patch"][i * args.predict_batch:(i + 1) * args.predict_batch]),
						   (-1, n_input))
		# print(feedx[1])
		temp = sess.run(pred, feed_dict={x: feedx})
		
		"""for itm in temp[0]:
			print("%.6lf" % itm, end=" ")
		print()
		print(All_data["labels"][i * args.predict_batch])"""
		
		prob_map = np.concatenate((prob_map, temp), axis=0)
	if num_all % args.predict_batch != 0:
		feedx = np.reshape(np.asarray(All_data["patch"][pred_times * args.predict_batch:]), (-1, n_input))
		# feedx = np.asarray(All_data["patch"][pred_times * args.predict_batch:], (-1, n_input))
		temp = sess.run(pred, feed_dict={x: feedx})
		prob_map = np.concatenate((prob_map, temp), axis=0)
	
	"""
	num_all = len(All_data['patch'])
	times = args.predict_times
	num_each_time = int(num_all / times)
	res_num = num_all - times * num_each_time
	Num_Each_File = num_each_time * np.ones((1, times), dtype=int)
	Num_Each_File = Num_Each_File[0]
	Num_Each_File[times - 1] = Num_Each_File[times - 1] + res_num
	start = 0
	prob_map = np.zeros((1, n_classes))
	for i in range(times):
		feed_x = np.reshape(np.asarray(All_data['patch'][start:start + Num_Each_File[i]]), (-1, n_input))
		temp = sess.run(pred, feed_dict={x: feed_x})
		prob_map = np.concatenate((prob_map, temp), axis=0)
		start += Num_Each_File[i]
	"""
	
	prob_map = np.delete(prob_map, (0), axis=0)
	
	# MRF
	DATA_PATH = os.path.join(os.getcwd(), args.directory)
	
	f = open(os.path.join(DATA_PATH, "101data.txt"), "w+")
	for items in prob_map:
		for i in items:
			print("%.6f" % i, file=f, end=" ")
		print(file=f)
	f.close()
	
	print('The shape of prob_map is (%d,%d)' % (prob_map.shape[0], prob_map.shape[1]))
	arr_test = np.zeros((1, patch_size * patch_size * 220))
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
	
	f = open(os.path.join(DATA_PATH, "./105label.txt"), "w+")
	for items in All_data["labels"]:
		print(items, file=f)
	f.close()
	
	Seg_Label, seg_Label, seg_accuracy = Post_Processing(prob_map, Height, Width, n_classes, y_test_scalar, TestIndex)
	print("The Final Seg Accuracy is :", seg_accuracy)
