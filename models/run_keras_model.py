import pandas
import sys
sys.path.append("..")
from utils import addis as util
import numpy as np
from addis_ababa_dataset import AddisAbaba
from datasets import *

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

def binary_acc(y_true, y_pred):
	a = y_true - y_pred
	b = K.abs(a)
	c = K.mean(b)
	print K.shape(y_true[0][0])
	print K.shape(y_pred)

	return c

def all_acc(y_true, y_pred):
	equals = np.equal(y_true, y_pred).astype(float)
	accuracies = np.mean(equals, axis=0)
	mean_accuracy = np.mean(accuracies)

	return mean_accuracy





file_name = "../Addis_data_processed.csv"
data = AddisAbaba(file_name)
batch_size = 5
epochs = 1
input_shape = (224, 224, 3)

model = ResNet50(include_top=False, weights='imagenet',
                 pooling='max', input_shape=input_shape)

binary_pred = Dense(data.num_binary_features(), activation='sigmoid', name = 'binary')(model.layers[-1].output)
binary_continuous = Dense(data.num_continuous_features(), activation='linear', name='continuous')(model.layers[-1].output)
model = Model(input=model.input, output=[binary_pred, binary_continuous])
model.compile(loss={'binary': keras.losses.binary_crossentropy, 'continuous': keras.losses.mean_squared_error},
              optimizer=keras.optimizers.Adam(lr=0.0001, decay=0.005),
              metrics = {'binary': binary_acc})

x_train = np.random.rand(15, 224, 224, 3)
y_train_binary = data.y_binary[:15]
y_train_continuous = data.y_continuous[:15]

model.fit(x_train, {'binary': y_train_binary, 'continuous': y_train_continuous},
          batch_size=batch_size,
          epochs=epochs)

preds = model.predict(x_train)

print all_acc(preds[0], y_train_binary)


