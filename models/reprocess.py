import os
import numpy as np

num_files = {'addis' : 3591,
             'afro' : 7022,
             'afro_asc' : 7022, 
             'afro_des' : 7022}

# Original Files
sat = 's1'
data_source = 'afro_des'  #'addis'
filename_dict ={'addis_s1' : 's1_median_addis_multiband_500x500_','addis_l8' : 'l8_median_addis_multiband_500x500_', 'addis_skysat' : 'skysat_median_addis_multiband_500x500_', 'afro_asc_s1' : 's1_median_afrobarometer_multiband_ascending_500x500_', 'afro_des_s1' : 's1_median_afrobarometer_multiband_descending_500x500_', 'afro_l8': 'l8_median_afrobarometer_multiband_500x500_'}
filetail = ".0.npy"
pathname = "/mnt/mounted_bucket/afro_s1_npy_copy/"
#print(pathname)
#print(data_source+'_' + sat)
#print("="*10)
batch_source = pathname + filename_dict[data_source + '_' + sat]

print(batch_source)

# raise NotImplemented("===break===!")
 
# New files
folder_name = "center_cropped"
resolution = "224"
new_pathname = "/home/mark_chen_jq/git-infrastructure/%s_%s_%s" % (data_source, sat, folder_name)
new_batch_source = "%s/%s_median_%s_multiband_%sx%s_" % (new_pathname, sat, data_source, resolution, resolution)
new_filetail = ".npy"


reset = False

def resize(arr):
	return np.resize(arr, (224, 224, 5))

def crop(arr):
	offset_from = (500-224)//2
	offset_to = 500 - offset_from

	return arr[offset_from:offset_to, offset_from:offset_to, 0:3]

func = crop

if not os.path.exists(new_pathname):
	os.mkdir(new_pathname)

for i in range(1, num_files[data_source] + 1):
	if i % 100 == 0: print "At %d" % i

	if not reset and os.path.exists(new_batch_source + str(i-1) + new_filetail):
		continue

	if os.path.exists(batch_source+str(i)+filetail):
		arr = np.load(batch_source+str(i)+filetail) #loads npy file
		arr = func(arr)
		np.save(new_batch_source + str(i-1) + new_filetail, arr)
	elif os.path.exists(batch_source+str(i)+".npy"+filetail):
		arr = np.load(batch_source+str(i)+".npy"+filetail)
		arr = func(arr)
		np.save(new_batch_source + str(i-1) + new_filetail, arr)
	else:
		print("Sattelite image %d not found!" % i)

