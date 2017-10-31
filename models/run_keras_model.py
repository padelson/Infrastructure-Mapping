import pandas
import sys
sys.path.append("..")
from utils import addis as util
import numpy as np
from addis_ababa_dataset import AddisAbaba
from datasets import *
import os

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
# Get accuracy metrics and print them out for each category (in a CSV file)
# Get F1 score and print them out for each category
# Normalize
###########

def all_acc(y_true, y_pred):
	equals = np.equal(y_true, y_pred).astype(float)
	accuracies = np.mean(equals, axis=0)
	mean_accuracy = np.mean(accuracies)

	return mean_accuracy

def preprocess(x):
	x = np.resize(x, (x.shape[0], 224, 224, 5))
	return x[:, :, :, :3]

class DataGenerator():

	def __init__(self, data, binary = True, continuous = True):
		self.data = data
		self.num_train_batches = data.num_train_batches()
		self.num_test_batches = data.num_test_batches()
		self.binary = True
		self.continuous = True

	def train_generate(self):
		for i in range(self.num_train_batches):
			x_train_batch = preprocess(data.get_x_batch(i))

			y_train_binary_batch, y_train_continuous_batch = data.get_y_batch(i)

			output_dict = {}
			if self.binary: output_dict['binary'] = y_train_binary_batch
			if self.continuous: output_dict['continuous'] = y_train_continuous_batch

			yield x_train_batch, output_dict

	def eval_generator(self):
		for i in range(self.num_test_batches):
			x_test_batch = preprocess(data.get_x_test_batch(i))
			y_test_binary_batch, y_test_continuous_batch = data.get_y_test_batch(i)

			output_dict = {}
			if self.binary: output_dict['binary'] = y_test_binary_batch
			if self.continuous: output_dict['continuous'] = y_test_continuous_batch

			yield x_test_batch, output_dict

class MetricsCallback(keras.callbacks.Callback):

	def __init__(self, model, data, generator):
		self.model = model
		self.data = data
		self.data_generator = generator

	def on_epoch_end(self, epoch, logs=None):
		preds = model.predict_generator(self.data_generator, data_generator.num_batches)
		print preds.shape

batch_size = 2
epochs = 2
input_shape = (224, 224, 3)

file_name = "../Addis_data_processed.csv"
data = AddisAbaba(file_name, batch_size)

def do_binary():
	model = ResNet50(include_top=False, weights='imagenet',
	                 pooling='max', input_shape=input_shape)

	binary_pred = Dense(data.num_binary_features(), activation='sigmoid', name = 'binary')(model.layers[-1].output)
	model = Model(input=model.input, output=binary_pred)
	model.compile(loss=keras.losses.binary_crossentropy,
	              optimizer=keras.optimizers.Adam(lr=0.0001, decay=0.005))

	data_generator = DataGenerator(data, continuous = False)

	metrics = MetricsCallback(model, data, data_generator)

	model.fit_generator(data_generator.train_generate(), data_generator.num_train_batches, callbacks=[metrics], epochs = 2)

def do_all():
	model = ResNet50(include_top=False, weights='imagenet',
	                 pooling='max', input_shape=input_shape)

	binary_pred = Dense(data.num_binary_features(), activation='sigmoid', name = 'binary')(model.layers[-1].output)
	binary_continuous = Dense(data.num_continuous_features(), activation='linear', name='continuous')(model.layers[-1].output)
	model = Model(input=model.input, output=[binary_pred, binary_continuous])
	model.compile(loss={'binary': keras.losses.binary_crossentropy, 'continuous': keras.losses.mean_squared_error},
	              optimizer=keras.optimizers.Adam(lr=0.0001, decay=0.005),
	              metrics = {'binary': binary_acc})

	data_generator = DataGenerator(data)

	model.fit_generator(data_generator.generate(), data_generator.num_train_batches, epochs = 2)

if __name__ == "__main__":
	do_binary()