from keras.models import Sequential
from keras.layers.core import Dense, Activation
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import metrics
# from sklearn import metrics
# import os
featurelen = 17
num_classes = 2


logits = np.loadtxt(open("angry_bird_data.csv", "rb"), delimiter = ",", skiprows = 1, usecols = range(featurelen))
labels = np.loadtxt(open("angry_bird_data.csv", "rb"), delimiter = ",", skiprows = 1, usecols = (-1,))

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

#Convert the int numpy array into a one-hot matrix
labels_onehot = (np.arange(51) == labels_np[:, None]).astype(np.uint8) 

x_train, x_test, y_train, y_test = train_test_split(logits, labels_onehot, test_size = 0.25, random_state = 42)
model = Sequential()
model.add(Dense(units = 500, activation = 'relu', input_dim = x_train.shape[1]))
model.add(Dense(units = 500, activation = 'relu'))
model.add(Dense(units = 500, activation = 'relu'))
model.add(Dense(units = 51, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = "adam")


model.fit(x_train, y_train, validation_data = (x_test, y_test), verbose = 0, epochs = 10, batch_size = 5147)
#Correcting the batch_size improved the accuracy to approx. 6 %


pred = model.predict(x_test)
pred = np.argmax(pred, axis = 1)
y_compare = np.argmax(y_test, axis = 1)
score = metrics.accuracy_score(y_compare, pred)
print("Final accuracy: {}".format(score))