from resnet import inception_resnet_v2
import tensorflow as tf
import pandas
import sys
sys.path.append("..")
from utils import addis as util
import numpy as np
from addis_ababa_dataset import AddisAbaba
from datasets import *
import tensorflow.contrib.slim as slim


# TO DO:
# 
# Remove -888 and other values on Addis Ababa dataset
# Obtain the binary features and continuous features once
# Write get sattelite images function
# Save model

def build_placeholders(data):
	y_binary = tf.placeholder(tf.float32, shape=(None, data.num_binary_features()))
	y_continuous = tf.placeholder(tf.float32, shape=(None, data.num_continuous_features()))
	x = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))

	return {'x': x,  'y_binary': y_binary, 'y_continuous': y_continuous}

def build_resnet(placeholders, checkpoint_name):
	logits, end_points = inception_resnet_v2(placeholders['x'])
	end_points['final_features'] = end_points['PreLogitsFlatten']

	return end_points

def build_predictions(end_points, data):
	end_points['logits_binary'] = slim.fully_connected(end_points['final_features'], data.num_binary_features(), activation_fn=None, scope='Logits_Binary')
	end_points['logits_continuous'] = slim.fully_connected(end_points['final_features'], data.num_continuous_features(), activation_fn=None, scope='Logits_Continuous')

	end_points['predictions_binary'] = tf.nn.sigmoid(end_points['logits_binary'])
	end_points['predictions_continuous'] = end_points['logits_continuous']

	return (end_points['predictions_binary'], end_points['predictions_continuous'])

def build_loss(end_points, placeholders, binary_weighting):
	end_points['loss_binary'] = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=placeholders['y_binary'], logits=end_points['logits_binary']))
	end_points['loss_continuous'] = tf.nn.l2_loss(end_points['logits_continuous'] - placeholders['y_continuous'])

	end_points['loss_total'] = binary_weighting*end_points['loss_binary'] + (1-binary_weighting)*end_points['loss_continuous']

	return end_points['loss_total']

def build_accuracy(end_points, placeholders):

	rounded_predictions = tf.round(end_points['predictions_binary'])
	correct_predictions = tf.equal(placeholders['y_binary'], rounded_predictions)
	correct_predictions_float = tf.cast(correct_predictions, tf.float32)
	num_correct_predictions = tf.reduce_sum(correct_predictions_float)
	num_predictions = tf.cast(tf.shape(correct_predictions)[0], tf.float32)
	end_points['acc'] = num_correct_predictions / num_predictions

	return end_points['acc']


def build_optimizer(end_points, lr):
	end_points['train_op'] = tf.train.AdamOptimizer(lr).minimize(end_points['loss_total'])

	return end_points['train_op']

if __name__ == "__main__":
	checkpoint_name = "checkpoints/inception_resnet_v2_2016_08_30.ckpt"
	file_name = "../addis_ababa/Addis_processed.csv"
	lr = 0.01
	binary_loss_weighting = 0.5
	batch_size = 128
	num_epochs = 5
	print_every = 5

	data = AddisAbaba(file_name)

	tf.reset_default_graph()
	placeholders = build_placeholders(data)
	end_points = build_resnet(placeholders, checkpoint_name)
	build_predictions(end_points, data)
	build_loss(end_points, placeholders, binary_loss_weighting)
	acc = build_accuracy(end_points, placeholders)
	train_op = build_optimizer(end_points, lr)

	sess = tf.Session()
	saver = tf.train.Saver()
	saver.restore(sess, checkpoint_name)
	sess.run(tf.initialize_all_variables())

	for epoch in range(num_epochs):
		for iteration in range(data.num_batches(batch_size)):
			x_batch = data.get_x_batch(iteration)
			y_binary_batch, y_continuous_batch = data.get_y_batch(iteration)

			feed_dict = {placeholders['x']: x_batch, placeholders['y_binary']: y_binary_batch, placeholders['y_continuous']: y_continuous_batch}
			if iteration % print_every == 0: 
				_, acc_, loss_binary_, loss_continuous_ = sess.run([train_op, acc, end_points['loss_binary'], end_points['loss_continuous']], feed_dict = feed_dict)
				print ("Epoch %d, Iteration %d, Loss_binary %.2f, Accuracy %.2f, Loss_reg %.2f" % (epoch, iteration, loss_binary_, acc_, loss_continuous_))
			else:
				sess.run(train_op, feed_dict = feed_dict)

		x_test = data.get_x_test()
		y_binary_test, y_continuous_test = data.get_y_test()

		feed_dict = {placeholders['x']: x_test, placeholders['y_binary']: y_binary_test, placeholders['y_continuous']: y_continuous_test}
		acc_, loss_binary_, loss_continuous_ = sess.run([acc, end_points['loss_binary'], end_points['loss_continuous']], feed_dict = feed_dict)
		print ("Testing: >>> Epoch %d, Loss %.2f, Accuracy %.2f" % (epoch, loss_, acc_))



