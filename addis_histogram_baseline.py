# Single User Classification 
# Constructs a unique classifier for each user and evaluates a sample of comments 
# to determine which came from the user and which didn't

from collections import defaultdict
import numpy as np
import random
import os
import pandas as pd
from utils import histogram as util
from utils import addis
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

verbose = True
nbins = 100
nbands = 4
data = pd.read_csv('Addis_data_processed.csv')
path = '/mnt/mounted_bucket/saved_img/s1'
predictor = 'svm'

def get_all_histograms(read=True):
	if read:
		return joblib.load('s1_histograms.pkl')
	all_features = np.zeros((len(data), nbins * nbands))
	done = 0
	for f in os.listdir(path):
		if 's1' in f:
			index = f[f.rfind('_') + 1:f.find('.0.npy')]
			index = index.replace('.npy', '')
			i = int(index) - 1
			arr = np.load(path + '/' + f)
			all_features[i, :] = util.convert_array(arr, bins=nbins)
			done += 1
		if done % 100 == 0:
			print 'done',done
	joblib.dump(all_features, 's1_histograms.pkl', compress=True)
	return all_features

def get_predictor():
	if predictor == 'logreg':
		return LogisticRegression()
	if predictor == 'svm':
		return LinearSVC(dual=False)

def main():
	print 'transforming histograms'
	all_features = get_all_histograms()
	split = int(len(data) * 0.8)
	x_train, x_test = all_features[:split], all_features[split:]
	print "loaded %d training, %d testing samples" % (len(x_train), len(x_test))
	all_results_file = open('s1_histograms_results_' + predictor + '.txt', 'w')
	all_results = {}
	for col in addis.binary_features:
		print col
		all_vals = data[col]
		y_train = all_vals[:split]
		y_test = all_vals[split:]
		
		if np.max(y_train) == 0:
			print "all 0 column!"
			continue
		reg = get_predictor()
		print "Training model"
		reg.fit(x_train, y_train)
		predictions = reg.predict(x_test)
		precision, recall, f1, support = precision_recall_fscore_support(y_test, predictions, average='binary')
		if verbose:
			print "\tF1:%f, Precision:%f, Recall:%f" % (f1, precision,recall)
		all_results_file.write(col + ": \tF1:%f, Precision:%f, Recall:%f\n" % (f1, precision,recall))
		all_results[col] = (f1, precision, recall)
	joblib.dump(all_results, 's1_histogram_results_' + predictor + '.pkl', compress=True)

if __name__ == '__main__':
	main()
