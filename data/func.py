import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

'''
    Evaluates function at given input

    args: 
        input: numpy array of n x d 

    returns:
        output: numpy array n x 1

'''

def obj(inp):   
    # Import MNIST data


    # Set parameters
    learning_rate = inp[0]#0.01
    training_iteration = inp[1]#30
    batch_size = inp[2]#100
    display_step = 2

    # TF graph input
    x = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28=784
    y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes

    # Create a model

    # Set model weights
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    with tf.name_scope("Wx_b") as scope:
        # Construct a linear model
        model = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax


    # More name scopes will clean up graph representation
    with tf.name_scope("cost_function") as scope:
        # Minimize error using cross entropy
        # Cross entropy
        cost_function = -tf.reduce_sum(y*tf.log(model))

    with tf.name_scope("train") as scope:
        # Gradient descent
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

    # Initializing the variables
    init = tf.initialize_all_variables()


    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        
        # Set the logs writer to the folder /tmp/tensorflow_logs

        # Training cycle
        for iteration in range(training_iteration):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                # Fit training using batch data
                sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
                # Compute the average loss
                avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys})/total_batch
                
            # Display logs per iteration step
            #if iteration % display_step == 0:
                #print "Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost)


        # Test the model
        predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
        acc = accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
        
        error = 1.0 - acc
        print ("error:", error)
        return error