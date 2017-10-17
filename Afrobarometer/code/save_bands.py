import csv
import numpy as np
import matplotlib.pyplot as plt
import gdal

with open("/mnt/mounted_bucket/Afrobarometer_R6.csv", 'r') as f:
    survey = list(csv.reader(f, delimiter= ","))
areas = [];
for i in range(1,len(survey)):
    item = survey[i];
    if item[0] == 'Tanzania' or item[0] == 'Uganda' or item[0] == 'Kenya' or item[0] == 'Burundi' or item[0] == 'Rwanda':
        areas.append(item[-1]);
#areas = [item[1] if item[1] is 'Tanzania' or if item[1] is 'Kenya' or if item[1] is 'Uganda' or if item#[1] is 'Burundi' or if item[1] 'Rwanda' for item in survey[1:]];
print (len(areas))

landsat_path = '/mnt/mounted_bucket/l8_median_afrobarometer_multiband_500x500_';

def read(tif_path, H,W):
    gdal_dataset = gdal.Open(tif_path)
    x_size, y_size = gdal_dataset.RasterXSize, gdal_dataset.RasterYSize
    gdal_result = gdal_dataset.ReadAsArray((x_size-W)//2, (y_size-H)//2, W,H)

    return np.transpose(gdal_result, (1,2,0))

i = 0

log = ""
for i in range(0, len(areas)):
    tifpath = landsat_path+areas[i]+'.0.tif';
    try:
        img = read(tifpath, 500, 500)
        path = 'saved_images/img_'+areas[i];
        np.save(path, img)
    except:
        log = log + 'failed: ' + areas[i] + '\n'

text_file = open("log.txt", "w");
text_file.write(log);
text_file.close;
        
