import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

#fix random seed for reproductibility
seed = 7
numpy.random.seed(seed)


#Since output variable contains string, it is easiest to load the data using pandas.
# We can then split the attributes (columns) into input variables (x) and output (y)

#load dataset
dataframe = pandas.read_csv("iris.csv", header = None)
dataset = dataframe.values

X = dataset[:,0:4].astype(float)
Y = dataset[:, 4]

#ENCODE OUTPUT VARIABLE


#encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
#Convert integers to one_hot format
dummy_y = np_utils.to_categorical(encoded_Y)  ####IMPORTANT FOR REPORT


# Iris-setosa,	Iris-versicolor,	Iris-virginica
# 1,		0,			0
# 0,		1, 			0
# 0, 		0, 			1  


#Defining The Neural Network Model
# 4 inputs -> [8 hidden nodes] -> 3 outputs

#Baseline Model

def baseline_model():
	#create model
	model = Sequential()
	model.add(Dense(8, input_dim = 4, activation = 'relu'))
	model.add(Dense(3, activation = "softmax"))
	#Compile Model
	model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	return model


estimator = KerasClassifier(build_fn = baseline_model, epochs = 200, batch_size = 5, verbose = 0)

#Evaluate Model with k-Fold Cross Validation

kfold = KFold(n_splits = 10, shuffle = True, random_state = seed)

results = cross_val_score(estimator, X, dummy_y, cv = kfold)
print("Baseline: %.2f%%)" % (results.mean() * 100, results.std() * 100))
