import tensorflow as tf
import numpy as np


'''
Specifies model used as basis function 
for linear input-output mapping
'''


class BasisModel():

    '''
    initializes model specified here
    '''
    def __init__(self,dim_in,train_mode = True ):       

    
        self._dim_input = dim_in
        dim_output = 1
        n_hidden1 = 20
        n_hidden2 = 40
        n_hidden3 = 30 
        self._basis_dim = n_hidden3
        self._train_mode = train_mode


        if(train_mode == True):
            self._x_in = tf.placeholder(tf.float32, shape = [None,self._dim_input],name="plzwork1")
            self._x_inn = self._x_in

        else:
            with tf.name_scope("trainable_basis_model"):
                self._x_in = tf.placeholder(tf.float32, shape = [1,self._dim_input],name="plzwork2")
                init_val  = np.random.normal(size = (1,self._dim_input))
                self._x_inn = tf.Variable(init_val,name = "wrtin",dtype= tf.float32)
                self._var_op = tf.assign(ref = self._x_inn, value=self._x_in, validate_shape = False)



        self._w1 = tf.Variable(tf.truncated_normal([self._dim_input,n_hidden1]))
        self._b1 = tf.Variable(tf.zeros([n_hidden1]))
        l1 = tf.nn.sigmoid(tf.matmul(self._x_inn,self._w1) + self._b1)
        
        self._w2 = tf.Variable(tf.truncated_normal([n_hidden1,n_hidden2]))
        self._b2 = tf.Variable(tf.zeros([n_hidden2]))
        l2 = tf.nn.sigmoid(tf.matmul(l1,self._w2) + self._b2)
        
        self._w3 = tf.Variable(tf.truncated_normal([n_hidden2,n_hidden3]))
        self._b3 = tf.Variable(tf.zeros([n_hidden3]))
        self._l3 = tf.nn.sigmoid(tf.matmul(l2,self._w3) + self._b3) #this output is the basis function
        
        self._w4 = tf.Variable(tf.truncated_normal([n_hidden3,dim_output]))
        self._b4 = tf.Variable(tf.zeros([dim_output]))
        self._y_pred = tf.matmul(self._l3,self._w4)  + self._b4
        
            #Saver object for saving nnet parameters
        self._save_params = tf.train.Saver({"w1":self._w1, "b1": self._b1,"w2":self._w2, "b2": self._b2,"w3":self._w3, "b3": self._b3, "w4":self._w4, "b4": self._b4})
          
        
    def get_params(self):
        if(self._train_mode):
            return self._x_in, self._y_pred
        else:
            return self._x_in, self._x_inn, self._y_pred, self._l3, self._var_op, self._basis_dim
    def save_model(self,sess,path):
        save_path = self._save_params.save(sess, path + 'model.ckpt')
        return save_path    
    def load_model(self, sess, path):
        self._save_params.restore(sess, path)
    
    
    

