import tensorflow as tf
import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data

data =  np.loadtxt(open("angry_bird_data.csv", "rb"), delimiter = ",", skiprows = 1)

labels = []				
logits = [] #features
#num_classes = 29

train_data = data[:-1][:int(len(data) * 0.8)]
train_labels = digits[-1][:int(len(data) * 0.8)]
valid_images = digits[0][int(len(data[0]) * 0.8):]
valid_labels = digits[1][int(len(data[0]) * 0.8):]


'''

for item in data:
	item = item.flatten()  #May be redundant
	labels.append(int(item[-1])) #Take User ID as Label

	temp = []
	for x in item[0:-1]: #Up to and not including the last element
		temp.append(float(x))
	logits.append(temp)

NUM_LABELS = np.amax(labels) - np.amin(labels) # 29 different users from 22 - 51
BATCH_SIZE = len(logits) / 12     #Batch Size = 5147 Total of 12 Batches

#Create Test Set

print(labels[0]) #User ID: #22
print(len(logits[0])) #Features corresponding to User ID: #22

#Convert the array of float arrays into a numpy float matrix
logits_np = np.matrix(logits).astype(dtype = np.float32)

#convert the array of int labels into a numpy array
labels_np = np.array(labels).astype(dtype = np.uint8)

#Convert the int numpy array into a one-hot matrix
labels_onehot = (np.arange(51) == labels_np[:, None]).astype(np.uint8)   #Needs to be able to compare up to the 51st user
print(labels_onehot[0])

print(labels_onehot[8155])


#we want to shuffle data later on "random.shuffle(array)"
#array[:,0] means we want all the 0th elements in the array, returns new array with just those elements
def main(argv = None):

	x = tf.placeholder('float', shape = [None, len(logits[0])])
	y = tf.placeholder('float', shape = [None, 29])    #Remember you gave the one hot a depth of 51



'''








'''
one hot formatting user tensorflow
# list = [1, 2 ,3]

# with tf.Session() as sess:
# 	print(sess.run(tf.one_hot(list, 4, dtype = tf.int32)))



'''




# n_nodes_hl1 = 500
# n_nodes_hl2 = 500
# n_nodes_hl3 = 500

# n_classes = 10


# batch_size = 100 

# x = tf.placeholder('float', [None, 784 ]) 

# y = tf.placeholder('float')


# def neural_network_model(data):
# 	hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
# 							'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

# 	hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
# 							'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
# 	hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
# 							'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

# 	output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
# 							'biases': tf.Variable(tf.random_normal([n_classes]))}

# 	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
# 	l1 = tf.nn.relu(l1) 

# 	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
# 	l2 = tf.nn.relu(l2) 
# 	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
# 	l3 = tf.nn.relu(l3)

# 	output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

# 	return output

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = prediction, labels = y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	num_epochs = 10

	with tf.Session() as sess:
		tf.global_variables_initializer().run()

		for epoch in range(num_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
				epoch_loss += c
			print('Epoch', epoch + 1, 'completed out of', num_epochs, 'loss:', epoch_loss)
			correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))

			accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
			print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


train_neural_network(x)   -----> init_data()