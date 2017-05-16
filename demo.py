import numpy as np
import tensorflow as tf
import time
import gc
import getopt
import sys
import matplotlib.pyplot as plt

from models.model_basis import BasisModel 
from models.model_Bayesian import Bayesian
from data.func_demo import obj
from helper_fns import train 
from helper_fns import generatestartx

gc.disable() #disable garbage collector

logs_path = "tmp/debug"






def main(argv):
    #default
    basis_training_epochs = 5000
    total_iters = 6 #NUM data points to collect
    num_startx = 4 #number of starting exploration points
    startx_lowrange = -5 #should be chosen according to problem domain
    startx_highrange = 5
    opt_iters = 400
    bayes_alpha = 0.2
    bayes_beta = 4.0
    model_name = "nothingrn"
    acq_type = 'EI'
    save_dir = "/home/anmol/projects/dngo/models/saved_models/"
    X_dir = ('data/X.npy')
    y_dir = ('data/y.npy')

    opts, args = getopt.getopt(argv, "te:ti:ns:oe:ba:bb:at:sd:sx:sy:mn:")
    for opt, arg in opts:
        if (opt == "-te"):
            basis_training_epochs = arg
        elif (opt == "-ti"):
            total_iters = arg

        elif (opt == "-ns"):
            num_startx = arg 
        elif (opt == "-oe"):
            opt_iters = arg
        elif(opt == "-ba"): 
            bayes_alpha = arg

        elif (opt == "-bb"):
            bayes_beta = arg
        elif (opt == "-at"):
            acq_type = arg
        elif (opt == "-sd"):
            save_dir = arg

        elif (opt == "-sx"):
            X_dir = arg
        elif (opt == "-sy"):
            y_dir = arg
        elif(opt == "-mn"):
            model_name = arg
            #Load data
    x_train = np.load(X_dir)
    dim_inp = x_train.shape[1]    
    y_train = np.load(y_dir)

    
    sess = tf.Session() #use default session

    #Instantiate basis model for training
    m1 = BasisModel(dim_in=dim_inp, train_mode = True)
    x_place,out_t = m1.get_params()
    train(x_place, out_t, x_train, y_train, basis_training_epochs, sess) #FIND BETTER WAY TO DO INitIALIZATION
    save_dir = m1.save_model(sess, save_dir)
    #Instatiante basis model for optimizing EI criterion 
    m2 = BasisModel(dim_in=dim_inp, train_mode=False)
    
    
    
    for iter in range(total_iters):
        
        x_train = np.reshape(x_train,(-1,dim_inp))
        y_train = np.reshape(y_train,(-1,1))

        train(x_place, out_t, x_train, y_train,basis_training_epochs, sess)
       
        m2.load_model(sess, save_dir)
        xplace_2, x_var, y_hat, basis, var_op, basis_dim = m2.get_params()
        bayes = Bayesian(basis, xplace_2, x_var, x_train, y_train,basis_dim, bayes_alpha,bayes_beta, sess, var_op,acq_type) 
        startx = generatestartx(num_startx, np.array([[-6.0, 6.0]])) 
        #THIS NEEDS TO BE SPECIFIED BY USER IN SCRIPT#np.random.random_integers(startx_lowrange, startx_highrange, (num_startx, dim_inp)) + np.random.rand()/2.0

        if(iter>0):          
            startx[-1,:] = next_xsample 
        
        ei_min = 990000000 
        
        #try all starting points for optimization -- use the one with the largest EI for sampling
        start_time2 = time.time()
        for start_iter in range(num_startx):            
            grad1,grad2, candidate_x = bayes.optimize_acquisition(np.reshape(startx[start_iter,:],(-1,dim_inp)),opt_iters)
            candidate_ei_minus = bayes.get_ei(candidate_x)

            if (candidate_ei_minus<ei_min and not(candidate_x in x_train)): #and (not(candidate_x[0,0] in x_train[:,0] and candidate_x[0,1] in x_train[:,1]))):
                gradbest1 = grad1
                gradbest2  = grad2
                ei_min = candidate_ei_minus
                next_xsample = candidate_x
                best_iter = start_iter
        print('time for EI opt ' + str(time.time() - start_time2))


        print ("grads before and after"  + "\n")
        print(gradbest1)
        print (gradbest2)
            


    
        next_ysample = obj(next_xsample)
   
      
        #bestsofarx[iter,:] = next_xsample
        #bestsofary[iter] = next_ysample
        x_train = np.append(x_train, next_xsample, axis = 0)
        y_train = np.append(y_train, next_ysample)
        print("Iteration " + str(iter) + '\n')
        print("before optimization:" + str(startx[best_iter,:]))
        print("optimal x: " + str(next_xsample))
        print("optimal y: " + str(next_ysample))

        print("best so far: " + str(np.min(y_train)))
        print('\n')
        #print("total time taken this loop: ", time.time() - start_time)

    np.save(save_dir + model_name + "x_train.npy",x_train)
    np.save(save_dir + model_name + "y_train.npy", y_train)

    #teststuff 1D only !!
    numpoints = 500
    xtest = np.reshape(np.linspace(-7,7,numpoints),(numpoints,1))
    ypreds = np.empty((xtest.shape[0],1))
    eis = np.empty((xtest.shape[0],1))
    sigs = np.empty((xtest.shape[0],1))
    ytest = obj(xtest)
    for i in range(numpoints):        
        eis[i] = bayes.get_ei(np.reshape([xtest[i]],(1,1)))
        ypreds[i],sigs[i]  = bayes.predict(np.reshape([xtest[i]],(1,1)))
    sctplot = plt.figure()
    plt.scatter(xtest,ytest)
    plt.scatter(xtest,ypreds)
    plt.show()
    
if (__name__ == "__main__"):
    main(sys.argv[1:])
    

