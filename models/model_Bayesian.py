import tensorflow as tf
import numpy as np
from .acq_funcs import acquisition_function

class Bayesian():

    #dim(map_params) = (1,x_features)
    #dim(A) = (numtrain, x_features), dim(xtest) = (x_features, 1), dim(ytrain) = (numtrain,1)
    #dim(Av) = (x_features, x_features), dim(lambda_pos)=(1)
    
    
    '''
    Instantiates the class and builts the posterior function
    
        Args:
            basis - reference to the basis function, a tensorflow Tensor with shape = (basis dimensions, 1)
            xplaceholder - tensorflow placeholder, to provide input data
            xinn - input data in the form of a tf.Variable. Acquisiton function is optimized wrt this variable
            xtrain - numpy array of known function inputs, shape = (num_samples, num_dimensions)
            ytrain - numpy array of known corresponding function outputs to xtrain, shape = (num_samples,1)
            a - prior on weight variance(hyperparameter), a float
            b - prior on (hyperparameter), a float
            
    '''
    
    def __init__(self, basis, xplaceholder, xinn, xtrain, ytrain,basis_dim, a,b, sess, var_op,acq_type = 'EI'):

        self._xinn = xinn
        self._a = a
        self._b = b
        self._sess = sess
        self._basis = basis
        self._xplaceholder = xplaceholder
        self._var_op = var_op

        iden = np.identity(basis_dim) 
        iden =  tf.cast(iden, tf.float64)
    
        self._A = np.zeros((len(xtrain), basis_dim))
        for i in range(len(xtrain)):
            self._sess.run(self._var_op, feed_dict = {self._xplaceholder: [xtrain[i]]}) 
            self._A[i,:] = sess.run(basis, feed_dict = {self._xplaceholder:[xtrain[i]]})

        self._A = tf.cast(self._A, tf.float64)
    
        self._Av = tf.matrix_inverse(self._a*tf.matmul(tf.transpose(self._A),self._A) + (self._b)*(self._b)*iden)
        self._A = tf.cast(self._A, tf.float64)
        self._Av = tf.cast(self._Av, tf.float64)

        self._inte = tf.matmul(self._Av, tf.transpose(self._A))

        self._map_params = tf.matmul(self._inte,ytrain)   #(30x1)(6x1)

        self._basis = tf.cast(self._basis, tf.float64)
        self._s1 = tf.matmul(self._basis, self._Av)
        self._phiK = tf.matmul(self._Av, tf.transpose(self._basis)) 
        self._sig =  (1/self._a) + tf.matmul(self._s1, tf.transpose(self._basis))
        self._mu_pos = self._a*tf.matmul(self._basis, self._map_params)     #self._a*   
        self._sample_min = np.min(ytrain) 

        self._acq = acquisition_function(acq_type)
   
        
    
    '''
    Optimizes the acquisiton function, maximizing the Expected Improvement criterion
    Args:
        trialx - 1x1 numpy array, specifying point to start optimization
    Returns:
        1x1 numpy array, specifying optimal point that maximizes expected improvemet
    
    '''
    
    def optimize_acquisition(self,trialx,opt_iters):

        ei = self._acq.eval(self._mu_pos, self._sig, self._sample_min,self._sess)

        self._objective_acq = tf.convert_to_tensor(tf.cast(ei,tf.float64)) 
        temp_vars = set(tf.global_variables())
        opt = tf.train.AdamOptimizer(epsilon = 1e-4)
        train_op = opt.minimize(self._objective_acq,var_list = [self._xinn])
        self._sess.run(tf.variables_initializer(set(tf.global_variables()) - temp_vars))
        self._sess.run(self._var_op, feed_dict = {self._xplaceholder: trialx})
        grad1 = self._sess.run(tf.gradients(self._objective_acq, [self._xinn]))       
        #print("ei value:")
        #print(self._sess.run(ei))
        for h in range(opt_iters):
            self._sess.run(train_op)
        grad2 = self._sess.run(tf.gradients(self._objective_acq, [self._xinn]))
        return (grad1,grad2,self._sess.run(self._xinn))
        #self._sess.run(np.random)
        #return self._sess.run(ei)#,feed_dict = {self._xplaceholder: trialx})        
    
    '''
        set the sample minimum based on the known data points
    '''
    def set_sample_min(self,minn):
        self._sample_min = minn
    def predict(self,trialx):
        self._sess.run(self._var_op, feed_dict = {self._xplaceholder:trialx})
        return self._sess.run((self._mu_pos[0],self._sig[0]))
    def get_ei(self, trialx):
        self._sess.run(self._var_op, feed_dict = {self._xplaceholder:trialx})
        return 1.0*self._sess.run(self._objective_acq)
        