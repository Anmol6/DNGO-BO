import tensorflow as tf
import numpy as np

'''
Train the basis model 

	Args: 
		xin : input placeholder
		out : output placeholder
		x_train : X training data
		y_train : Y training data
		n_epochs : number of training epochs
		sess : tensorflow Session to run training

	
'''



def train(xin, out, x_train,y_train, n_epochs, sess):
    
    Y_train = tf.placeholder(tf.float32, shape=[None,1], name="ytrain")
    MSEloss = tf.reduce_mean(tf.square(Y_train-out))
    train = tf.train.AdamOptimizer().minimize(MSEloss)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)     

    for i in range(n_epochs):  
        sess.run(train, feed_dict = {xin:x_train,Y_train: y_train})





#TODO: ADD more ways to do this

'''
Args:
	ranges: a D X 2 numpy array of lower and upper bounds for each dimension
	type: type of sampling to perform
	num_points: integer, specifying number of points to generate
Returns: 
	startingx: a num_points X D numpy array of starting points

'''
def generatestartx(num_points, ranges, type = "uni_rand"):
	if (type == "uni_rand"):
		d = ranges.shape[0]

		startingx = np.empty((num_points,d))
		for i in range(num_points):
			for j in range(d):
				startingx[i,j] = ranges[j,0]+(ranges[j,1]-ranges[j,0])*np.random.rand() 
		return startingx

