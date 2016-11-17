

import numpy as np
import tensorflow as tf

'''
3-layer nnet - last layer is used as basis for bayesian linear inference

'''


def model():
	
	dim_input = 1
	n_hidden1 = 10
	n_hidden2 = 35
	n_hidden3 = 30
	dim_output = 1
	
	x_in = tf.placeholder(tf.float32, shape = [None,1],name="plzwork")
	#x_inn = tf.Variable(x_in, validate_shape=False)
	x_inn = tf.Variable([[1.0]],name = "wrtin")
	#x_inn = x_in
	var_op = tf.assign(x_inn, x_in, validate_shape = False)
	#var_op = x_inn.assign(x_in,validate_shape=False)
	
	w1 = tf.Variable(tf.truncated_normal([dim_input,n_hidden1]))
	b1 = tf.Variable(tf.zeros([n_hidden1]))
	l1 = tf.nn.sigmoid(tf.matmul(x_inn,w1) + b1)
	
	w2 = tf.Variable(tf.truncated_normal([n_hidden1,n_hidden2]))
	b2 = tf.Variable(tf.zeros([n_hidden2]))
	l2 = tf.nn.sigmoid(tf.matmul(l1,w2) + b2)
	
	w3 = tf.Variable(tf.truncated_normal([n_hidden2,n_hidden3]))
	b3 = tf.Variable(tf.zeros([n_hidden3]))
	l3 = tf.nn.sigmoid(tf.matmul(l2,w3) + b3)
	
	w4 = tf.Variable(tf.truncated_normal([n_hidden3,dim_output]))
	b4 = tf.Variable(tf.zeros([dim_output]))
	y_pred = tf.matmul(l3,w4) 
	
  
	basis = l3
	params = w4
	
	return x_in, x_inn, y_pred, l3,var_op

'''
This learns the basis function needed for bayesian inference

	Args:
		xin - placeholder for data input
		out - output tensor of the nnet
		x_train - numpy array of training data(input), shape = (-1,1)
		y_train - numpy array of training data(output), shape = (-1,1)
		n_epochs - number of epochs for training, an integer
		sess - current tensorflow Session

'''
	
def train(xin, out, x_train,y_train, n_epochs, sess,var_op):
	
	Y_train = tf.placeholder(tf.float32, shape=[1,1], name="yyy")
	MSEloss = tf.reduce_mean(tf.square(Y_train-out))
	train = tf.train.AdamOptimizer().minimize(MSEloss)
	init_op = tf.initialize_all_variables()
	sess.run(init_op,feed_dict = {xin:[[1.0]]})
	l = len(x_train)
	for i in range(n_epochs):        
		sess.run(train, feed_dict = {xin:[np.reshape(x_train,(-1,1))[i%l]], Y_train: [np.reshape(y_train,(-1,1))[i%l]]})



'''
Class for performing bayesian inference on the data to model the unkown objective function

'''

class Bayesian():

	#dim(map_params) = (1,x_features)
	#dim(A) = (numtrain, x_features), dim(xtest) = (x_features, 1), dim(ytrain) = (numtrain,1)
	#dim(Av) = (x_features, x_features), dim(lambda_pos)=(1)
	
	
	'''
	Instantiates the class and computes 
	
		Args:
			basis - reference to the basis function, a tensorflow Tensor with shape = (basis dimensions, 1)
			xplaceholder - tensorflow placeholder, to provide input data
			xinn - input data in the form of a tf.Variable. Acquisiton function is optimized wrt this variable
			xtrain - numpy array of known function inputs, shape = (num_samples, num_dimensions)
			ytrain - numpy array of known corresponding function outputs to xtrain, shape = (num_samples,1)
			a - prior on weight variance(hyperparameter), a float
			b - prior on (hyperparameter), a float
			
	'''
	
	def __init__(self, basis, xplaceholder, xinn, xtrain, ytrain,a,b, sess, var_op):

		self._xinn = xinn
		self._a = a
		self._b = b
		self._sess = sess
		self._basis = basis
		self._xplaceholder = xplaceholder
		iden = tf.Variable(initial_value = np.identity(30))
		iden =  tf.cast(iden, tf.float64)
		self._objjj = tf.cast(tf.Variable(1.0),tf.float64)

		#sess.run(tf.initialize_all_variables())
		self._A = np.zeros((len(xtrain), 30))
		for i in range(len(xtrain)):
			self._A[i,:] = sess.run(basis, feed_dict = {xplaceholder:[xtrain[i]]})

		#_,self._A = sess.run((var_op,basis), feed_dict = {xplaceholder:xtrain})  
		self._A = tf.cast(self._A, tf.float64)
	
		self._Av = tf.matrix_inverse(self._a*tf.matmul(tf.transpose(self._A),self._A) + (self._b)*(self._b)*iden)
		self._A = tf.cast(self._A, tf.float64)
		self._Av = tf.cast(self._Av, tf.float64)

		self._inte = tf.matmul(self._Av, tf.transpose(self._A))
		#print(tf.shape(ytrain))
		self._map_params = tf.matmul(self._inte,ytrain)   #(30x1)(6x1)

		self._xtest = self._basis
		self._xtest = tf.cast(self._xtest, tf.float64)
		s1 = tf.matmul(self._xtest, self._Av)
		self._phiK = tf.matmul(self._Av, tf.transpose(self._xtest)) 
		self._sig =  (1/self._a) + tf.matmul(s1, tf.transpose(self._xtest))
		self._mu_pos = tf.matmul(self._a*self._xtest, self._map_params)        
		self._sample_min = np.min(xtrain) 

	'''
	Builds the posterior distribution function.
	note: function can be changed to return the posterior function at point xtest
	
	'''    

	def posterior(self):        
		self._xtest = self._basis
		self._xtest = tf.cast(self._xtest, tf.float64)
		s1 = tf.matmul(self._xtest, self._Av)
		self._phiK = tf.matmul(self._Av, tf.transpose(self._xtest)) 
		self._sig =  (1/self._a) + tf.matmul(s1, tf.transpose(self._xtest))
		self._mu_pos = tf.matmul(self._a*self._xtest, self._map_params)
		
		
	
	'''
	Optimizes the acquisiton function, maximizing the Expected Improvement criterion
	Args:
		trialx - 1x1 numpy array, specifying point to start optimization
	Returns:
		1x1 numpy array, specifying optimal point that maximizes expected improvemet
	
	'''
	
	def optimize_acquisition(self,trialx):
		#self._xtest=trialx
		dist = tf.contrib.distributions.Normal(mu = self._mu_pos[0], sigma=self._sig[0])
		objective_acq = tf.convert_to_tensor(dist.cdf(tf.cast(self._sample_min,tf.float64)))#+2.0 #TODO: add expectation
		opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
		#train_op = opt.minimize(objective_acq,var_list = [self._xinn])
		self._sess.run(tf.initialize_all_variables())
		return self._sess.run(objective_acq, feed_dict = {self._xplaceholder: trialx})        
	
	'''
		set the sample minimum based on the known data points
	'''
	def set_sample_min(self,minn):
		self._sample_min = minn
		


'''
objective function, used for experimental purposes

'''
def objective(x):
	y = x**2+x**3;
	return y 

#train data
x_train = np.array([-7.0,-6.0,-5.0,0.0,3.0,7.0])
np.random.shuffle(x_train)
y_train = objective(x_train)
x_train = np.reshape(x_train, (-1,1))
y_train = np.reshape(y_train, (-1,1))

#test data
xtest = np.linspace(-12.0,12.0,10)
ytest = objective(xtest)
xtest = np.reshape(xtest,(-1,1))
ytest = np.reshape(ytest,(-1,1))



def main():
	with tf.Session() as sess:
		xplace, xinn, out, basis,var_op = model()
		train(xplace,out,x_train,y_train,1000, sess,var_op)
		sess.run(var_op,feed_dict = {xplace:[[1.00]]})
		#print(sess.run(basis, feed_dict = {xplace:x_train}))  
		bayes = Bayesian(basis, xplace, xinn, x_train, y_train, 8,0.4, sess,var_op)
		#sess.run(var_op,feed_dict = {xplace:[[1.00]]})

		#bayes.posterior()
		k = bayes.optimize_acquisition([[1.0]])
		print(k)



		



if __name__ == "__main__":
	main()