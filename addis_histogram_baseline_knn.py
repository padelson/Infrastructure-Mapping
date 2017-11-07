# Single User Classification 
# Constructs a unique classifier for each user and evaluates a sample of comments 
# to determine which came from the user and which didn't
from __future__ import division

from collections import defaultdict
import numpy as np
import random
import os
import pandas as pd
from utils import addis
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib

verbose = True
nbins = 100
nbands = 4
data = pd.read_csv('Addis_data_processed.csv')
satellite = 's1'
path = '/mnt/mounted_bucket/saved_img/' + satellite

def predict_all(names):
	all_features = data[['lat', 'long']]
	split = int(len(data) * 0.8)
	x_train, x_test = all_features[:split], all_features[split:]
	print "loaded %d training, %d testing samples" % (len(x_train), len(x_test))
	all_results = {}
	for col in names:
		all_vals = data[col]
		y_train = all_vals[:split]
		y_test = all_vals[split:]
		
		if np.max(y_train) == 0:
			print "all 0 column!"
			continue
		print "Training model"
		reg = KNeighborsClassifier()
		reg.fit(x_train, y_train)
		predictions = reg.predict(x_test)
		train_predict = reg.predict(x_train)
		precision, recall, f1, support = precision_recall_fscore_support(y_train, train_predict, average='binary')
		print "Training\tF1:%f, Precision:%f, Recall:%f" % (f1, precision,recall)
		precision, recall, f1, support = precision_recall_fscore_support(y_test, predictions, average='binary')
		print "Testing\tF1:%f, Precision:%f, Recall:%f" % (f1, precision,recall)
	joblib.dump(all_results, satellite + '_histogram_results_' + predictor + '.pkl', compress=True)
	# joblib.dump(all_results, 'logreg_test.pkl')

def main():
	predict_all(addis.binary_features)
if __name__ == '__main__':
	main()
