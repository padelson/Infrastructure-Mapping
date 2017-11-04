import pandas
import sys
sys.path.append("..")
from utils import addis as util
import numpy as np
from addis_ababa_dataset import AddisAbaba
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
from keras.applications.vgg16 import VGG16

file_name = "../Addis_data_processed.csv"

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

	a = x[:, :, :, 2:5]
	#b = (a - np.mean(a)) / 256

	a = a - a.mean(axis=0)
	a = a / np.sqrt((a ** 2).sum(axis=1))[:,None]

	return a*10

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
		self.losses = []
		self.preds = []

	def on_epoch_end(self, epoch, logs=None):
		data_generator = DataGenerator(self.data, continuous=False)
		y_preds = self.model.predict_generator(DataGenerator(data, continuous = False).train_generate(), self.data.num_train_batches())
		y_true = self.data.y_binary[:y_preds.shape[0], self.data.which_features]

		self.accuracies.append(all_acc(y_true, y_preds))
		self.f1_scores.append(all_f1(y_true, y_preds))
		self.losses.append(logs['loss'])
		self.preds.append(y_preds)

		# print ("Accuracy mean: %f, F1 score mean: %f" % (np.mean(accuracies), np.mean(f1_scores)))

def train_on_binary(data, output_filename='test.csv', epochs = 3):
	model = VGG16(include_top=False, weights='imagenet',
	                 pooling='max', input_shape=(224, 224, 3))

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
		writer.writerow(map (lambda x: util.binary_features[x], data.which_features))
		# writer.writerow(['losses'])
		# writer.writerows(metrics.losses)
		writer.writerow(['accuracies'])
		writer.writerows(metrics.accuracies)
		writer.writerow(['f1_scores'])
		writer.writerows(metrics.f1_scores)
		writer.writerow(['predictions'])

		writer.writerow([data.num_binary_features() * ""] + ["Epoch %d" % i if i % (data.num_binary_features()) == 0 else '' for i in range(epochs * (data.num_binary_features()))])

		predictions = np.array(metrics.preds)
		predictions_formatted = np.hstack(tuple(predictions))

		y_true = data.y_binary[:data.num_train, data.which_features]

		predictions_formatted = np.hstack((y_true, predictions_formatted))

		writer.writerow(map(lambda x: util.binary_features[x], data.which_features) * epochs)
		for i in range(predictions_formatted.shape[0]):
			writer.writerow(predictions_formatted[i].tolist())


if __name__ == "__main__":
	batch_size = 2
	learning_rate = 0.0001
	num_examples = 20
	epochs = 2
	which_features = [52]

	data = AddisAbaba(file_name, batch_size, num_examples, which_features)

	train_on_binary(data, "test.csv", epochs = epochs)





