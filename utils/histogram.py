import numpy as np

def convert_array(arr, bins=100, min_val=1, max_val=1000):
	'''
	Given an (x, y, z) shaped 3-d numpy array,
	computes a histogram along the z-axis and
	returns a (z, bins) shaped array, where each row
	corresponds to the histogram for that band.
	'''
	hist = np.zeros((arr.shape[2], bins), dtype=np.int)
	for band in range(arr.shape[2]):
		hist[band] = np.histogram(arr[:, :, band], bins=bins, range=(min_val, max_val))[0]
	return hist.flatten()