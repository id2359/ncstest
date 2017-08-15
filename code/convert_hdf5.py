import h5py
import numpy as np
from sklearn.model_selection import train_test_split as tts

data = np.genfromtxt("../data/processed.txt",delimiter=",")

features, labels = data[:,0:3], data[:,4]

data_file = h5py.File('../data/train.hdf5', 'w')
data_file.create_dataset('train', data=data)
data_file.close()

test_data_file = h5py.File('../data/test.hdf5', 'w')
test_data_file.create_dataset('test', data=test_data)
test_data_file.close()

