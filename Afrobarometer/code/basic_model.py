import csv
import numpy as np
import gdal
import os
import sys
sys.path.append("../../models")
from logistic_regression import *

with open("/mnt/mounted_bucket/Afrobarometer_R6.csv", 'r') as f:
    survey = list(csv.reader(f, delimiter= ","))

ids = []
X = []

for filename in os.listdir('saved_images'):
    id = int(filename[4:-4])
    arr = np.load("saved_images/"+filename)
    #print (np.mean(arr, axis = 0))
    X.append(np.mean(np.mean(arr, axis = 0), axis = 0))
    ids.append(id)

X = np.array(X)
Y = []
for i in range(0, len(ids)):
    id = ids[i]
    item = survey[id]
    Y.append(int(item[19])) #19 corresponds to electricity
Y = np.array(Y)
print(np.shape(X[0]))
lm = logreg_model(X.shape[1])
score_train, score_test = train_model(lm, X,Y, 5, 20)
print (score_train)
print (score_test)
