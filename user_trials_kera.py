from keras.models import Sequential
from keras.layers.core import Dense, Activation
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import keras.optimizers as optimizer
from keras.utils.np_utils import to_categorical

angry_bird_no_outlier = "angrybirds_30users_removed_outlier_exValues.csv"
angry_bird_with_outlier = "angry_bird_data.csv"
featurelen = 17
num_classes = 2

def model():
	logits = np.loadtxt(open(angry_bird_no_outlier, "rb"), delimiter = ",", skiprows = 1, usecols = range(featurelen))
	labels = np.loadtxt(open(angry_bird_no_outlier, "rb"), delimiter = ",", skiprows = 1, usecols = (-1,), dtype = np.uint8)

	labels = [x - 22 for x in labels]

	encoder = LabelEncoder()
	encoder.fit(labels)
	encoded_Y = encoder.transform(labels)
	labels_onehot = to_categorical(encoded_Y, num_classes=30)

	x_train, x_test, y_train, y_test = train_test_split(logits, labels_onehot, test_size = 0.25, random_state = 42, shuffle = True)

	model = Sequential()
	model.add(Dense(units = 100, activation = 'sigmoid', input_dim = x_train.shape[1], use_bias = True))
	model.add(Dense(units = 100, activation = 'sigmoid', use_bias = True))
	model.add(Dense(units = 100, activation = 'sigmoid', use_bias = True))
	model.add(Dense(units = len(labels_onehot[-1]), activation = 'softmax'))


	model.compile(loss = 'categorical_crossentropy', optimizer = optimizer.Adam())

	model.fit(x = x_train, y = y_train, validation_data = (x_test, y_test), verbose = 2, epochs = 2000, batch_size = 1000, shuffle = True)

	pred = model.predict(x_test)
	pred = np.argmax(pred, axis = 1)
	y_compare = np.argmax(y_test, axis = 1)
	score = metrics.accuracy_score(y_compare, pred)
	print("Final accuracy: {}".format(score *100) + " %")

model()