import pandas
import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import util

def initialize_data(datafile):
	data = pandas.read_csv('Addis_data.csv')
	return data

def change_feature_names(data):
	data.columns = util.NEW_NAMES
	

def get_assign_list(func, feature_dict, data):
	rows = [data[i:i+1] for i in range(len(data))]
	mapped_column = map(func, rows)
	return mapped_column

def create_columns(data, feature_types):
	dont_do = {}

	print "Starting mode F and O"

	for feature_dict in util.feature_dicts:
		print "Mode F/O: %s" % feature_dict['orig_name']
		if 'o' in feature_dict['mode']:
			if feature_dict['orig_name'] == 'bl_dw42':
				func = lambda x: 0 if x['bl_dw41'].values == 2 else x['bl_dw42'].values[0]
				data = data.assign(feat = get_assign_list(func, feature_dict, data))
				data = data.rename(columns = {'feat': feature_dict['new_name']})

			if feature_dict['orig_name'] == 'bl_dw63':
				func = lambda x: 0 if x['bl_dw56'].values == 2 else x['bl_dw63'].values[0]
				data = data.assign(feat = get_assign_list(func, feature_dict, data))
				data = data.rename(columns = {'feat': feature_dict['new_name']})

			if feature_dict['orig_name'] == 'bl_dw64':
				func = lambda x: 0 if x['bl_dw63'].values == 0 else x['bl_dw63'].values[0]
				data = data.assign(feat = get_assign_list(func, feature_dict, data))
				data = data.rename(columns = {'feat': feature_dict['new_name']})

			if feature_dict['orig_name'] == 'bl_sd46':
				func = lambda x: 150 if x['bl_dw45'].values == -888 else x['bl_dw63'].values[0]
				data = data.assign(feat = get_assign_list(func, feature_dict, data))
				data = data.rename(columns = {'feat': feature_dict['new_name']})

			feature_types[1].append(feature_dict['new_name'])

		if 'f' not in feature_dict['mode']: continue
		if 'b' in feature_dict['mode']:
			for value in range(feature_dict['num_values']):
				func = lambda x: 1 if (x[feature_dict['follows_feature']].values == feature_dict['follows_value']) and (x[feature_dict['orig_name']].values == value + 1) else 0
				data = data.assign(feat = get_assign_list(func, feature_dict, data))
				name = "%s_val%d_when_%s_val%d" % (feature_dict['new_name'], value+1, feature_dict['follows_feature'], feature_dict['follows_value'])
				data = data.rename(columns = {'feat': name})

				feature_types[0].append(name)

				if feature_dict['orig_name'] in dont_do.keys():
					dont_do[feature_dict['follows_feature']].append(feature_dict['follows_value'])
				else:
					dont_do[feature_dict['follows_feature']] = [feature_dict['follows_value']]

		if 'y' in feature_dict['mode']:
			for value in range(2):
				func = lambda x: 1 if x[feature_dict['follows_feature']].values == feature_dict['follows_value'] and x[feature_dict['orig_name']].values == value + 1 else 0
				data = data.assign(feat = get_assign_list(func, feature_dict, data))
				name = "%s_val%s_when_%s_val%d" % (feature_dict['new_name'], "YES" if value == 0 else "NO", feature_dict['follows_feature'], feature_dict['follows_value'])
				data = data.rename(columns = {'feat': name})	
				feature_types[0].append(name)		

				if feature_dict['orig_name'] in dont_do.keys():
					dont_do[feature_dict['follows_feature']].append(feature_dict['follows_value'])
				else:
					dont_do[feature_dict['follows_feature']] = [feature_dict['follows_value']]

	print "Finished mode F and O"
	print "Starting all modes"

	for feature_dict in util.feature_dicts:
		print "Mode All: %s" % feature_dict['orig_name']

		if 'f' in feature_dict['mode'] or 'd' in feature_dict['mode'] or 'o' in feature_dict['mode']: 
			data = data.drop(feature_dict['orig_name'], axis=1)
			continue

		if 'b' in feature_dict['mode']:
			for value in range(feature_dict['num_values']):
				if (feature_dict['orig_name'] in dont_do.keys() and value+1 in dont_do[feature_dict['orig_name']]): continue

				func = lambda x: 1 if x[feature_dict['orig_name']].values == value + 1 else 0
				data = data.assign(feat = get_assign_list(func, feature_dict, data))
				name = "%s_val%d" % (feature_dict['new_name'], value + 1)
				data = data.rename(columns = {'feat': name})
				feature_types[0].append(name)

		if 'y' in feature_dict['mode']:
			for value in range(2):
				if (feature_dict['orig_name'] in dont_do.keys() and value+1 in dont_do[feature_dict['orig_name']]): continue

				func = lambda x: 1 if x[feature_dict['orig_name']].values == value + 1 else 0
				data = data.assign(feat = get_assign_list(func, feature_dict, data))
				name = "%s_val_%s" % (feature_dict['new_name'], "YES" if value == 0 else "NO")
				data = data.rename(columns = {'feat': name})
				feature_types[0].append(name)

		if 'c' in feature_dict['mode']:
			data = data.rename(columns = {feature_dict['orig_name']: feature_dict['new_name']})
			feature_types[1].append(feature_dict['new_name'])

			continue

		if 'r' in feature_dict['mode']:
			func = lambda x: 1 if np.isnan(x[feature_dict['orig_name']].values)[0] else 0
			data = data.assign(feat = get_assign_list(func, feature_dict, data))
			name = "%s_val_NaN" % feature_dict['new_name']
			data = data.rename(columns = {'feat': name})
			feature_types[0].append(name)

			for value in range(feature_dict['num_values']):
				func = lambda x: 0 if (np.isnan(x[feature_dict['orig_name']]).values[0] or x[feature_dict['orig_name']].values != value + 1 or x[feature_dict['orig_name']].values == -888) else 1
				data = data.assign(feat = get_assign_list(func, feature_dict, data))
				name = "%s_val_%d" % (feature_dict['new_name'], value + 1)
				data = data.rename(columns = {'feat': name})
				feature_types[0].append(name)

		data = data.drop(feature_dict['orig_name'], axis=1)

	print "Finished!"

	return data

def run_pipeline():
	data = initialize_data('Addis_data.csv')
	binary_features, continuous_features = [], []
	data = create_columns(data, (binary_features, continuous_features))
	
	return data

def save_as_csv(data, filename):
	data.to_csv(filename)

if __name__ == "__main__":
	data = run_pipeline()
	to_csv(data, "Addis_data_processed.csv")
