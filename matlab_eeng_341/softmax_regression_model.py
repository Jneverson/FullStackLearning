from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

import tensorflow as tf 
sess = tf.InteractiveSession()

#Softmax Regression Model with a multilayer convolutional network

x = tf.placeholder(tf.float32, shape = [None, 784])
y_ = tf.placeholder(tf.float32, shape = [None, 10])

# [None, 784] => Dimensionality of a single flattened 28 x 28 pixel MNIST image
 # (None indicates first dimensionm corresponding to batch size, can be any size)

 # output y_ will be a 2d tensor where each row is a one-hot dimensional vector
 # indicating which digit class (zero through nine) the corresponsing MNIST image belongs to


#Variable is a value that lives inside the computation graph, it can be used and modified
 # by computation (in background)

 # 784 Input features, 10 outputs
W = tf.Variable(tf.zeros([784, 10])) 

 #10 Dimensional Vector because there are 10 classes
b = tf.Variable(tf.zeros([10]))


#Variables must be initialized within the session before they can be used
sess.run(tf.global_variables_initializer())

#Regression Model
y = tf.matmul(x, W) + b

#Loss Function: Indicates how bad the model's prediction was on a single example
#It is the cross-entropy between the target and the softmax activation function applied to
# models prediction

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, 
		logits = y))
#tf.nn.softmax... internally applies the softmax on the omdels unnormalized 
# prediction and sums across all classes and tf.reduce_mean takes the average over sumss


#Steepest Gradient Descent
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for _ in range(1000):
	batch = mnist.train.next_batch(100)
	# Use "feed dict" to replace placeholder tensors x and y_
	train_step.run(feed_dict = {x: batch[0], y_: batch[1]})


	# tf.argmax(y,1) is the label our model thinks is most likely for each input
	# tf.argmax(y_,1) is the true label
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# Gives us a list of booleans, to determine what fraction are correct we cast to floating point numbers
# then take mean... [True, False, Trie, True] would become [1, 0, 1, 1] equating to 0.75

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(str(accuracy.eval(feed_dict = {x: mnist.test.images, y_: mnist.test.labels}) * 100) + " %")

#92 % Accuracy