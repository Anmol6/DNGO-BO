import tensorflow as tf 
'''
Acquisition functions to determine next sampling point

'''
class acquisition_function():

    def __init__(self, func_type = 'EI'):
        self._type = func_type

    def eval(self,mu,sigma, min_val,sess,scaling_factor = -1000.0,BETA = 100):
        if (self._type == 'EI'):
            return self.EI(mu,sigma,min_val,sess,scaling_factor)
        if(self._type == 'UCB'):
            return self.UCB(mu,sigma,BETA)

    def EI(self,mu, sigma, min_val,sess, scaling_factor):
        '''
        Z = scaling_factor * (mu - min_val) / sigma
        Z = tf.cast(Z,tf.float64)
        dist = tf.contrib.distributions.Normal(loc=0., scale=1.)
        '''
        dist = tf.contrib.distributions.Normal(loc=tf.cast([0.],tf.float64), scale=tf.cast([1.],tf.float64))
        gamma_ = ( [min_val] - mu[0] )/(sigma[0])
        gamma = tf.cast(gamma_[0], tf.float64)
        
        #print("minval:")
        #print(str(min_val))
        
        print("sig")
        print(sess.run((sigma[0])))

        print ("mu")
        print(sess.run(mu[0]))
        
        #EI_ = [min_val] - mu[0] + 1e4*sigma[0]#([min_val]-mu[0])* dist.cdf(gamma) + ((sigma[0]) * dist.prob(gamma))
        
        EI_ = ([min_val]-mu[0])* dist.cdf(gamma) + ((sigma[0]) * dist.prob(gamma))
        EI = scaling_factor*EI_
        
        print("EIVAL ")
        print(sess.run(EI))
        
        return tf.cast(EI, tf.float64)

    def UCB(self,mu, sig, BETA = 1000):
        return mu - BETA*sig
