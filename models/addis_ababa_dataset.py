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
sat = 's1'
data_len = num_files[data_source]
batch_source = pathname + filename_dict[data_source + '_' + sat]

class AddisAbaba(Dataset):
	def __init__(self, filename, batch_size, train_test_split = 0.9):
                
		data = pandas.read_csv(filename)

		self.y_binary = data[util.binary_features].values
		self.y_continuous = data[util.continuous_features].values

		self.num_ids = num_files['addis']
		self.num_train = int(self.num_ids * train_test_split)
		self.num_test = self.num_ids - self.num_train

		self.batch_size = batch_size
        
	def num_binary_features(self):
		return len(util.binary_features)

	def num_continuous_features(self):
		return len(util.continuous_features)

	def num_train_batches(self):
		num_train_batches = self.num_train / self.batch_size
		return num_train_batches

	def num_test_batches(self):
		num_test_batches = self.num_test / self.batch_size
		return num_test_batches

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
		return np.array(x_batch)

	def get_y_batch(self, iteration):
		return self.y_binary[self.batch_size * iteration : self.batch_size*(iteration+1)], self.y_continuous[self.batch_size * iteration : self.batch_size * (iteration+1)]

	def get_x_test_batch(self, iteration):
		x_batch = []
		num_test = self.num_ids - self.num_train
		curr_id = self.num_train + iteration * self.batch_size + 1 # All is 1 indexed

		for i in range(curr_id, curr_id + self.batch_size):
			if i > data_len:
				break
			if os.path.exists(batch_source+str(i)+filetail):
				x_batch.append(np.load(batch_source+str(i)+filetail)) #loads npy file
			elif os.path.exists(batch_source+str(i)+".npy"+filetail):
				x_batch.append(np.load(batch_source+str(i)+".npy"+filetail))
			else:
				print batch_source+str(i)+".npy"+filetail
				raise Exception("Sattelite image %d not found!" % i)
		return np.array(x_batch)

	def get_y_test_batch(self, iteration):
		return 	self.y_binary[self.num_train + self.batch_size * iteration : self.num_train + self.batch_size*(iteration+1)], self.y_continuous[self.num_train + self.batch_size * iteration : self.num_train + self.batch_size * (iteration+1)]
