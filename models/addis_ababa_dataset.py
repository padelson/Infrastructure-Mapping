#The basic idea here is that x_batch wants to read  in .npy files as needed instead of storing all the data in one big matrix too big for memory
#the ugly stuff at top basically defines all the path names as needed
#change data_source and sat as needed to run on different x data.

import sys
sys.path.append("")
from utils import addis as util
from datasets import Dataset
import pandas
import numpy as np
import os
import scipy.misc as M

filename_dict ={'addis_s1' : 's1_median_addis_multiband_500x500_','addis_l8' : 'l8_median_addis_multiband_500x500_', 'addis_skysat' : 'skysat_median_addis_multiband_500x500_', 'afro_s1' : 's1_median_afrobarometer_multiband_500x500_', 'afro_l8': 'l8_median_afrobarometer_multiband_500x500_'}
filetail = ".0.npy"
pathname = "/mnt/mounted_bucket/saved_npy/"
num_files = {'addis' : 3591, 'afro' : 7022}
data_source = 'addis'
data_len = num_files[data_source]
sat = 's1'
batch_source = pathname + filename_dict[data_source + '_' + sat]

class AddisAbaba(Dataset):
	def __init__(self, filename, train_test_split = 0.8):
                
		data = pandas.read_csv(filename)

		self.y_binary = data[util.binary_features].values
		self.y_continuous = data[util.continuous_features].values

		self.num_ids = num_files['addis']
		self.training_size = self.num_ids * 0.8
        
	def create_splits(self):
		[self.y_binary_train, self.y_continuous_train, self.x_train] = map(lambda x: x[:self.split_value], [self.y_binary, self.y_continuous, self.x])
		[self.y_binary_test, self.y_continuous_test, self.x_test] = map(lambda x: x[self.split_value:], [self.y_binary, self.y_continuous, self.x])

	def num_batches(self, batch_size):
		self.num_batches = self.training_size / batch_size
		self.batch_size = batch_size
		return self.num_batches

	def get_x_batch(self, iteration):
		curr_id = iteration*self.batch_size + 1 #everything is 1 indexed
		x_batch = []
		for i in range(curr_id, curr_id+self.batch_size):
			if i > data_len:
				break
			if os.path.exists(batch_source+str(i)+filetail):
				x_batch.append(np.load(batch_source+str(i)+filetail)) #loads npy file
			elif os.path.exists(batch_source+str(i)+".npy"+filetail):
				x_batch.append(np.load(batch_source+str(i)+".npy"+filetail))
			else:
				print batch_source+str(i)+".npy"+filetail
				raise Exception("Sattelite image %d not found!" % i)
		return np.array(x_batch) #didn't want to mess up stacking. Is this cheating?

	def get_y_batch(self, iteration):
		return self.y_binary[self.batch_size * iteration : self.batch_size*(iteration+1)], self.y_continuous[self.batch_size * iteration : self.batch_size * (iteration+1)]

		# if (iteration == self.num_batches-1):
		# 	return self.y_binary[self.batch_size * iteration :], self.y_continuous[self.batch_size * iteration :]
		# else:
		# 	return self.y_binary[self.batch_size * iteration : self.batch_size * (iteration + 1)], self.y_continuous[self.batch_size * iteration : self.batch_size * (iteration + 1)]

	def num_binary_features(self):
		return len(util.binary_features)

	def num_continuous_features(self):
		return len(util.continuous_features)

	def get_x_test(self):
		return self.x_test

	def get_y_test(self):
		return self.y_binary_test, self.y_continuous_test
