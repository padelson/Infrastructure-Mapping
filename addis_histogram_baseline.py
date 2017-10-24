# Single User Classification 
# Constructs a unique classifier for each user and evaluates a sample of comments 
# to determine which came from the user and which didn't

from collections import defaultdict
import numpy as np
import random
import os
import pandas as pd
from utils import histogram as util
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression

verbose = True
data = pd.read_csv('Addis_data_processed.csv')
path = 'data-copy/saved_npy'

def get_all_histograms():
	all_features = []
	for f in os.listdir(path):
		if 's1' in f:
			arr = np.load('path' + '/' + f)
			all_features.append(util.convert_array(arr))
	return all_features

def main():
	all_features = get_all_histograms()
	all_vals = data['toilet_shared_val_YES']

	split = int(len(all_vals) * 0.8)
	x_train, y_train = all_features[:split], all_vals[:split]
	x_test, y_test = all_features[split:], all_vals[split:]

	reg = LogisticRegression()

	reg.fit(x_train, y_train)
	predictions = reg.predict(x_test)
	precision,recall,fbeta_score,support = precision_recall_fscore_support(y_test, predictions, average='binary')
	if verbose:
		print "\tPrecision:%f\n\tRecall:%f" % (precision,recall)

if __name__ == '__main__':
	main()