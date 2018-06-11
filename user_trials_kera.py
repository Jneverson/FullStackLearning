from keras.models import Sequential
from keras.layers.core import Dense, Activation
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import keras.optimizers as optimizer
# from sklearn import metrics
# import os
featurelen = 17
num_classes = 2


logits = np.loadtxt(open("angry_bird_data.csv", "rb"), delimiter = ",", skiprows = 1, usecols = range(featurelen))
labels = np.loadtxt(open("angry_bird_data.csv", "rb"), delimiter = ",", skiprows = 1, usecols = (-1,))

labels = [x - 22 for x in labels]
# num_classes = 2\
# data =  np.loadtxt(open("angry_bird_data.csv", "rb"), delimiter = ",", skiprows = 1)

# labels = []				
# logits = [] #features

# for item in data:
# 	item = item.flatten()  #May be redundant
# 	labels.append(int(item[-1])) #Take User ID as Label

# 	temp = []
# 	for x in item[0:-1]: #Up to and not including the last element
# 		temp.append(float(x))
# 	logits.append(temp)


# x, y = to_xy(features, labels)


labels_np = np.array(labels).astype(dtype = np.uint8)
print(len(labels_np))

#Convert the int numpy array into a one-hot matrix

labels_onehot = (np.arange(30) == labels_np[:, None]).astype(np.uint8) 
print(labels_onehot[-1])

# print([x for x in labels_onehot[-1]])

'''
encoder = LaberEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(Y)
#Convert integers to one_hot format
dummy_y = np_utils.to_categorical(encoded_Y)  ####IMPORTANT FOR REPORT

'''

x_train, x_test, y_train, y_test = train_test_split(logits, labels_onehot, test_size = 0.30, random_state = 42, shuffle = True)

print(type(x_train))
dir(x_train)
print(x_train.shape[1])
model = Sequential()
model.add(Dense(units = 100, activation = 'relu', input_dim = x_train.shape[1]))
model.add(Dense(units = 100, activation = 'relu'))
model.add(Dense(units = 100, activation = 'relu'))
model.add(Dense(units = len(labels_onehot[-1]), activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = optimizer.Adam())

# batch_size = 5147
model.fit(x = x_train, y = y_train, validation_data = (x_test, y_test), verbose = 0, epochs = 1000, batch_size = 1000)
#Correcting the batch_size improved the accuracy to approx. 6 % for 10 epochs)

pred = model.predict(x_test)
print("Pred 1 : " + str(pred))
pred = np.argmax(pred, axis = 1)
print("Pred 2 : " + str(pred))
y_compare = np.argmax(y_test, axis = 1)
print(str(y_compare))
score = metrics.accuracy_score(y_compare, pred)
print("Final accuracy: {}".format(score))