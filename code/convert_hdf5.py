import h5py
import numpy as np
from sklearn.model_selection import train_test_split as tts

data = np.genfromtxt("./data/processed.txt",delimiter=",")

features, labels = data[:,0:3], data[:,4]
X_train, X_test, y_train, y_test = tts(features,
                                       labels,
                                       test_size=0.3,
                                       random_state=42)


                  

data_file = h5py.File('./data/train.hdf5', 'w')
data_file.create_dataset('data',
                         X_train.shape,
                         data=X_train,
                         dtype="f8")

data_file.create_dataset('labels',
                         y_train.shape,
                         data=y_train,
                         dtype="i8")


data_file.close()

test_data_file = h5py.File('./data/test.hdf5', 'w')
test_data_file.create_dataset('data',
                              X_test.shape,
                              data=X_test,
                              dtype="f8")

test_data_file.create_dataset('labels',
                         y_test.shape,
                         data=y_test,
                         dtype="i8")

test_data_file.close()

