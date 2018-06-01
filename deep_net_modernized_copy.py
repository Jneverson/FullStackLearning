import tensorflow as tf
#Data is already in good format (from Voris)

''' 
Feed Forward Neural Network
-----------------------------
input > weight > hidden layer 1 (activation function) > weights > hidden l 2
(activation function) > weights > output layer

compare output to intended output > cost function (cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer....SGD, AdaGrad)

backpropogation

feed forward + backprop = epoch


'''
# tensrflow.models.official.mnist
# from tensorflow.models.official.mnist import dataset
# from tensorflow.examples.tutorials.mnist import input_data # (deprecated?)???
# tf.data ???
from tensorflow.examples.tutorials.mnist import input_data

#Above needs to change to import data from format Voris has it in


mnist = input_data.read_data_sets("MNIST_data", one_hot = True) 
'''tf.one_hot = True''' 

#one_hot : one is on, rest are of, good for multi class networks

#10 classes, 0 - 9
'''
You may want, 0 = 0, 1 = 1, 2 = 2, Instead due to one_hot
0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0,0]
'''

#3 Hidden Layers, 500 Nodes each (Customize based on data being used)
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10


'''
#Batches of 100 features at a time,
 feed them through network, 
 then manipulate the weights and 
 move on to the next batch
 e.g 1000 images at a time if analyzing image data
 (Customize based on # of features we have) 
'''
batch_size = 100 

#Matrix: height x width
#28 x 28
#flattened out to have no height, but extend 784 pixels long in this example (string of vals)
x = tf.placeholder('float', [None, 784 ]) #Input Data (type, [height, width])
#Because matrix shape was specified, if I attempt to feed in data not of this shape,
#Tensor flow will throw an error

y = tf.placeholder('float')
#Tensorflow automatically handles weight optimization

def neural_network_model(data):
	#Dictionary Mapping string 'weights' and string 'biases'
	 # to corresponding values for first hidden layer
	#Creates one giant tensor (array) of your weights
	#Weight Matrix : 784 x n_nodes_hl1 (500)
	# Biases are size of the number of nodes, bias connects to every node (all 500)
	#Bias: If all input data is 0, no neuron would fire, not always ideal, bias
	#just adds a value to that 0 so some neurons can still fire even if input val is 0
	hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
							'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

	#Come up with some type of dynamic for loop to iterate through a variable 
	# number of hidden layers, for example you wanted 100 layers


	#Input is number of nodes in hidden layer1 multiplied by number of nodes in hidden layer 2
	hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
							'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

	#HIdden Layer 2, Hidden Layer 3
	hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
							'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
							'biases': tf.Variable(tf.random_normal([n_classes]))}


	'''
	tf.random_normal returns tensor of specified shape filled with random normal values
	tf.random_normal(shape,*default values*...)  shape = [height, width] ; Matrix = height x width 
	(shape of matrix)
	'''

	
	# (input_data * weights) + biases (Actual Model)

	#Actual Layer 1 (Takes Data as input)
	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1) #Rectified Linear (Threshold Function) or (Activation Function)

	#Takes output of layer 1 as input
	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2) #Rectified Linear (Threshold Function) or (Activation Function)

	#Takes output of layer 2 as input
	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3) #Rectified Linear (Threshold Function) or (Activation Function)

	output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

	return output

#Previously built up computation Model and Neural Network Model, 
#Now we want to specify how we want to run data through the model in the session

# -------------------------------------------------------------------------------------

def train_neural_network(x):
	prediction = neural_network_model(x)

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = prediction, labels = y))
# tf.nn.softmax_cross_entropy_with_logits()

#Using Cross Entropy with logits as cost function (calculate the difference of the prediction
# we got and the known label we have ) --> outputed in one_hot format (can be any shape u want)
	
	#Goal to minimize the cost, create optimizer

#AdamOptimizer has a parameter called learning_rate and is defaulted to 0.001
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	#Cycles of feedforward + backprop
	num_epochs = 10

	with tf.Session() as sess:

		# Initalizes Variables and begins session
		# sess.run(tf.initialize_all_variables())
		tf.global_variables_initializer().run()

		for epoch in range(num_epochs): #Training Loop for network
			epoch_loss = 0
			# Total # of samples / batch size ::: Tells us how many times we need to cycle
			# Based on Batch Size we chose
			# mnist.train accesses the training data
			for _ in range(int(mnist.train.num_examples/batch_size)):
				# Chunks through data set for you, in reality you have to build a 
				# function yourself to do this (outside tensorflow)
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				# c is cost
				#Tensorflow knows from this to modify weights
				_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
				epoch_loss += c
			print('Epoch', epoch + 1, 'completed out of', num_epochs, 'loss:', epoch_loss)

			# tf.argmax returns index of max val in array
			# Hoping that the values at corresponding index are the same, both one_hots
			# tf.equal tells us whether they are identical
			correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))

			accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
			print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


train_neural_network(x)