# Implementation of logistic regresion in python using scikit learn for comparison

import h5py
from scipy.sparse import coo_matrix
from sklearn.linear_model import LogisticRegression
import numpy as np
input_file = "SST1.hdf5"
f = h5py.File(input_file, 'r')



coordinates = []
input_data = f['train_input']
output_data = f['train_output']

nfeatures = f['nfeatures'][0]
print nfeatures
print input_data.shape

for n in range(input_data.shape[0]):
	for item in input_data[n]:
		index = item - 1
		if index > 0:
			coordinates.append([n, index])
		if index == 0:
			break
print "Done adding vals"
coordinates = np.array(coordinates)
sparse_mat = coo_matrix((np.ones(coordinates.shape[0]), (coordinates[:, 0], coordinates[:, 1])))
print sparse_mat.shape
print "Beginning to fit logistic regression classifier"
logreg = LogisticRegression(C=1e5, multi_class='multinomial', solver='newton-cg')
logreg.fit(sparse_mat, output_data)





