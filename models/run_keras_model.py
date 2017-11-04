import pandas
import sys
sys.path.append("..")
from utils import addis as util
import numpy as np
from addis_ababa_dataset import AddisAbaba
from afrobarometer_dataset import Afrobarometer
from datasets import *
import os
import csv

from sklearn.metrics import f1_score
from scipy.cluster.vq import whiten

from sklearn.preprocessing import scale

import numpy as np
import keras

import keras.backend as K

from keras import optimizers
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50, preprocess_input

#### TO DO
# Test accuracy
# Print out
###########

def all_acc(y_true, y_pred):
	equals = np.equal(y_true, np.round(y_pred)).astype(float)
	accuracies = np.mean(equals, axis=0)

	return accuracies

def all_f1(y_true, y_pred):

	y_pred_rounded = np.round(y_pred)

	scores = []
	for i in range(y_true.shape[1]):
		scores.append(f1_score(y_true[:, i], y_pred_rounded[:, i]))

	return np.array(scores)

def preprocess(x):
	# x = np.resize(x, (x.shape[0], 224, 224, 5))

	a = x[:, :, :, :3]
	#b = (a - np.mean(a)) / 256

	a = a - a.mean(axis=0)
	a = a / np.sqrt((a ** 2).sum(axis=1))[:,None]

	return a

class DataGenerator():

	def __init__(self, data, binary = True, continuous = True):
		self.data = data
		self.binary = True
		self.continuous = False

	def train_generate(self):
		while 1:
			for i in range(self.data.num_train_batches()):
				x_train_batch = preprocess(self.data.get_x_batch(i))
				y_train_binary_batch, y_train_continuous_batch = self.data.get_y_batch(i)

				output_dict = {}
				if self.binary: output_dict['binary'] = y_train_binary_batch
				if self.continuous: output_dict['continuous'] = y_train_continuous_batch

				yield x_train_batch, output_dict

	def eval_generate(self):
		while 1:
			for i in range(self.data.num_test_batches()):
				x_test_batch = preprocess(self.data.get_x_test_batch(i))
				y_test_binary_batch, y_test_continuous_batch = self.data.get_y_test_batch(i)

				output_dict = {}
				if self.binary: output_dict['binary'] = y_test_binary_batch
				if self.continuous: output_dict['continuous'] = y_test_continuous_batch

				yield x_test_batch, output_dict

class MetricsCallback(keras.callbacks.Callback):

	def __init__(self, model, data):
		self.model = model
		self.data = data
		self.accuracies = []
		self.f1_scores = []

	def on_epoch_end(self, epoch, logs=None):
		print "On epoch end!"
		data_generator = DataGenerator(self.data, continuous=False)
		y_preds = self.model.predict_generator(DataGenerator(data, continuous = False).train_generate(), self.data.num_train_batches())
		y_true = self.data.y_binary[:y_preds.shape[0]]

		self.accuracies.append(all_acc(y_true, y_preds))
		self.f1_scores.append(all_f1(y_true, y_preds))

		# print ("Accuracy mean: %f, F1 score mean: %f" % (np.mean(accuracies), np.mean(f1_scores)))

def train_on_binary(data, output_filename):
	model = ResNet50(include_top=False, weights='imagenet',
	                 pooling='max', input_shape=input_shape)

	binary_pred = Dense(data.num_binary_features(), activation='sigmoid', name = 'binary')(model.layers[-1].output)
	model = Model(input=model.input, output=binary_pred)
	model.compile(loss=keras.losses.binary_crossentropy,
	              optimizer=keras.optimizers.Adam(lr=learning_rate, decay=0.005))

	metrics = MetricsCallback(model, data)
	model.fit_generator(DataGenerator(data, continuous = False).train_generate(), data.num_train_batches(), callbacks = [metrics], epochs = epochs)
	#score = model.evaluate_generator(DataGenerator(data, continuous = False).eval_generate(), data.num_test_batches())
	
	myfile = open(output_filename, 'w')
	with myfile:
		writer = csv.writer(myfile)
		writer.writerow(util.binary_features)
		writer.writerow(['accuracies'])
		writer.writerows(metrics.accuracies)
		writer.writerow(['f1_scores'])
		writer.writerows(metrics.f1_scores)
		writer.writerow(['score'])
		# writer.writerow([score])

if __name__ == "__main__":
	batch_size = 32
	epochs = 7
	input_shape = (224, 224, 3)
	learning_rate = 0.00001

	file_name = "../Afrobarometer_R6.csv"
	data = Afrobarometer(file_name, batch_size)

	train_on_binary(data, "test_final_last.csv")


	batch_size = 32
	epochs = 5
	input_shape = (224, 224, 3)
	learning_rate = 0.0001

	file_name = "../Afrobarometer_R6.csv"
	data = Afrobarometer(file_name, batch_size)

	train_on_binary(data, "test2.csv")


	batch_size = 32
	epochs = 4
	input_shape = (224, 224, 3)
	learning_rate = 0.0005

	file_name = "../Afrobarometer_R6.csv"
	data = Afrobarometer(file_name, batch_size)

	train_on_binary(data, "test3.csv")
