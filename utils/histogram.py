import numpy as np
import matplotlib.pyplot as plt

def convert_array(arr, bins=100, min_val=-25, max_val=10):
	'''
	Given an (x, y, z) shaped 3-d numpy array,
	computes a histogram along the z-axis and
	returns a (z, bins) shaped array, where each row
	corresponds to the histogram for that band.

	for landsat: min=1, max=1000
	for sentinel: min=-25, max=10
	'''
	hist = np.zeros((arr.shape[2], bins), dtype=np.int)
	for band in range(arr.shape[2]):
		if min_val < 0 and band == 2: # strangely, entire band in s1 will be same value sometimes
			continue
		hist[band] = np.histogram(arr[:, :, band], bins=bins, range=(min_val, max_val))[0]
	if min_val < 0:
		hist = hist[[0, 1, 3, 4], :]
	return hist.flatten()

def show_histograms(filename):
	img = np.load(filename)
	for band in range(img.shape[2]):
		arr = img[:, :, band].flatten()
		plt.hist(arr, bins=100, density=True)
		plt.xlabel('reflectance')
		plt.ylabel('proportion')
		plt.title('band_' + str(band) + '_' + filename.replace('.npy', '.png'))
		plt.show()