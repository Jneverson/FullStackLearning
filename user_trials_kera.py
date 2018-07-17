from keras.models import Sequential
from keras.layers.core import Dense, Activation
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import keras.optimizers as optimizer
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout
from keras.layers import BatchNormalization
import itertools as it
from pathlib import Path

featurelen = 17

def prepare_data(featurecols = range(featurelen)):

	temp_logits = []
	temp_labels = []
	logits = None
	labels = None

	numfiles = int(input("How many training sets are you using?: "))
	if numfiles > 1:
		user_string = input("Enter the names of the files you would like to merge\nseperated by a comma with no spaces: ")
		files = user_string.split(",")
		for item in files:
			temp_logits.append(np.loadtxt(open(item, "rb"), delimiter = ",", skiprows = 1, usecols = featurecols))
			temp_labels.append(np.loadtxt(open(item, "rb"), delimiter = ",", skiprows = 1, usecols = (-1,), dtype = np.uint8))

		for i in range(len(temp_logits) - 1):
			logits = np.vstack([temp_logits[i], temp_logits[i + 1]])
			labels = np.hstack([temp_labels[i], temp_labels[i + 1]])
	else:
		file = input("Enter the name of your file: ")
		logits = np.loadtxt(open(file, "rb"), delimiter = ",", skiprows = 1, usecols = featurecols)
		labels = np.loadtxt(open(file, "rb"), delimiter = ",", skiprows = 1, usecols = (-1,), dtype = np.uint8)

	return logits, labels

def test_combination(combination):
	col_list = list(range(featurelen))
	col_combinations = list(it.combinations(col_list, combination))
	print("Total Cycle count: " + str(len(col_combinations)))

	for item in col_combinations:
		print("Cycle " + str(col_combinations.index(item) + 1) + " of " + str(col_combinations) + " starting...")
		file.write("Column " + str(item[0]) + " and " + str(item[1]) + " only\n")
		file.write(model(featurecols = item))
		file.write("\n")
		print("Cycle " + str(col_combinations.index(item) + 1) + " of " + str(col_combinations) + " ending...")

def test_single():
	for i in range(featurelen):
		print("Cycle " + str(i + 1)  + " of " + str(featurelen) + " starting...")
		file.write("Column " + str(i) + " only\n")
		file.write(model(featurecols = (i,)))
		file.write("\n")
		print("Cycle " + str(i + 1)  + " of " + str(featurelen) + " ending...")

def fisher_extraction_test(file):
	col_list = list(range(featurelen))
	fisher_list = [6,0,5,1,3,15,7,8,9,12,14,4,11,16,2,13,10]

	file.write("\n")
	file.write("Intended test by Professor Regarding Fisher Score:")
	file.write("\n")
	file.write("Fisher Score Decreasing Order Extraction Test:")
	file.write("\n")
	file.write("------------------------------------------")
	file.write("\n")

	for i in fisher_list:
		if len(col_list) <= 1:
			break
		print("Cycle " + str(fisher_list.index(i) + 1)  + " of " + str(featurelen - 1) + " starting...")
		file.write("Column " + str(i) + " extracted\n")
		col_list.remove(i)
		file.write(model(featurecols = col_list))
		file.write("\n")
		print("Cycle " + str(fisher_list.index(i) + 1)  + " of " + str(featurelen - 1) + " ending...")
	
	fisher_list.remove(col_list[0])
	# fisher_list.reverse() No need to reverse, readding first what was taken out first
	file.write("Fisher Score Re-addition according to decreasing order test\n")
	for i in fisher_list:
		print("Cycle " + str(fisher_list.index(i) + 1)  + " of " + str(featurelen - 1) + " starting...")
		file.write("Column " + str(i) + " added\n")
		col_list.append(i)
		file.write(model(featurecols = col_list))
		file.write("\n")
		print("Cycle " + str(fisher_list.index(i) + 1)  + " of " + str(featurelen - 1) + " ending...")

def feature_validity_testing(): 
	# Destination File for Configurations
	# file = open('/home/infinity/Documents/Neural_Research/model_configurations/nn_feature_configurations.txt', 'a+')
	file = open(str(Path.home()) + 'nn_feature_configurations.txt', 'a+')
	'''
	#Base Test, All:------------------------------------------------------------------------------------
	file.write("All Features:\n" + model())
	file.write("\n")

	#Individual Feature Testing:-------------------------------------------------------------------------------------
	# test_single()

	#Combination Testing------------------------------------------------------------------------------
	test_combination(combination = 2)
	'''	
	#Fisher Extraction Test
	fisher_extraction_test(file)

	file.close()

def model(featurecols = range(featurelen)):
	logits = np.loadtxt(open('angrybirds.csv', "rb"), delimiter = ",", skiprows = 1, usecols = featurecols) 

	labels = np.loadtxt(open('angrybirds.csv', "rb"), delimiter = ",", skiprows = 1, usecols = (-1,), dtype = np.uint8)
	# logits, labels = prepare_data(featurecols = featurecols)
	labels = [x - 22 for x in labels]

	encoder = LabelEncoder()
	encoder.fit(labels)
	encoded_Y = encoder.transform(labels)
	labels_onehot = to_categorical(encoded_Y, num_classes=30)

	x_train, x_test, y_train, y_test = train_test_split(logits, labels_onehot, test_size = 0.25, random_state = 42, shuffle = True)

	model = Sequential()
	#Determine whether user is trying to enter a tuple representing a single column or multiple columns as another type
	if(len(featurecols) == 1):
		model.add(Dense(units = 1000, activation = 'sigmoid', input_dim = 1, use_bias = True, bias_initializer = "random_uniform"))
	else: 
		model.add(Dense(units = 1000, activation = 'sigmoid', input_dim = x_train.shape[1], use_bias = True, bias_initializer = "random_uniform"))
	model.add(Dense(units = 750, activation = 'sigmoid', use_bias = True, bias_initializer = "random_uniform"))
	model.add(Dense(units = 500, activation = 'sigmoid', use_bias = True, bias_initializer = "random_uniform"))
	model.add(Dense(units = 250, activation = 'sigmoid', use_bias = True, bias_initializer = "random_uniform"))
	model.add(Dense(units = len(labels_onehot[-1]), activation = 'softmax'))

	model.compile(loss = 'categorical_crossentropy', optimizer = optimizer.Adam())
	model.fit(x = x_train, y = y_train, validation_data = (x_test, y_test), verbose = 2, epochs = 200, batch_size = 1000, shuffle = True)
	pred = model.predict(x_test)
	pred = np.argmax(pred, axis = 1)
	y_compare = np.argmax(y_test, axis = 1)
	score = metrics.accuracy_score(y_compare, pred)
	# print("Final accuracy: {}".format(score *100) + " %")
	return "Final accuracy: {}+".format(score * 100) + " %"

def model_with_batch_normalization(featurecols = range(featurelen)):

	# logits = np.loadtxt(open(angry_bird_no_outlier, "rb"), delimiter = ",", skiprows = 1, usecols = featurecols) 
	# labels = np.loadtxt(open(angry_bird_no_outlier, "rb"), delimiter = ",", skiprows = 1, usecols = (-1,), dtype = np.uint8)
	logits, labels = prepare_data()
	labels = [x - 22 for x in labels]

	encoder = LabelEncoder()
	encoder.fit(labels)
	encoded_Y = encoder.transform(labels)
	labels_onehot = to_categorical(encoded_Y, num_classes=30)

	x_train, x_test, y_train, y_test = train_test_split(logits, labels_onehot, test_size = 0.25, random_state = 42, shuffle = True)


	model = Sequential()
	#Determine whether user is trying to enter a tuple representing a single column or multiple columns as another type
	if(len(featurecols) == 1):
		model.add(Dense(units = 1000, activation = 'sigmoid', input_dim = 1, use_bias = False))
	else: 
		model.add(Dense(units = 1000, activation = 'sigmoid', input_dim = x_train.shape[1], use_bias = False))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())
	model.add(Dense(units = 750, activation = 'sigmoid', use_bias = False))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())
	model.add(Dense(units = 500, activation = 'sigmoid', use_bias = False))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())
	model.add(Dense(units = 250, activation = 'sigmoid', use_bias = False))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())
	model.add(Dense(units = len(labels_onehot[-1]), activation = 'softmax'))


	model.compile(loss = 'categorical_crossentropy', optimizer = optimizer.Adam())

	model.fit(x = x_train, y = y_train, validation_data = (x_test, y_test), verbose = 2, epochs = 100, batch_size = 1000, shuffle = True)

	pred = model.predict(x_test)
	pred = np.argmax(pred, axis = 1)
	y_compare = np.argmax(y_test, axis = 1)
	score = metrics.accuracy_score(y_compare, pred)
	# print("Final accuracy: {}".format(score *100) + " %")
	return "Final accuracy: {}".format(score * 100) + " %"

# print(model())
# prepare_data()
feature_validity_testing()
# print(model_with_batch_normalization())+








