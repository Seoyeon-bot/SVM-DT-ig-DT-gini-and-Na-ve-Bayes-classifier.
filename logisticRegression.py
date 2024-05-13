###############################     INSTALLATION/PREP     ################
# This is a code template for logistic regression using stochastic gradient ascent to be completed by you 
# in CSI 431/531 @ UAlbany
#


###############################     IMPORTS     ##########################
import numpy as np
import pandas as pd
import math as mt
from numpy import linalg as li
import matplotlib.pyplot as plt


###############################     FUNCTION DEFINITOPNS   ##########################

#  use the logistic function (𝜃) to convert 𝑓(x) to 𝜋(x) numerical to probablity of categorical value 
def logisticFunction(z):
  etha_z = 0; 
  etha_z = 1/ (1 + np.exp(-z))
  return etha_z 

"""
Receives data point x and coefficient parameter values w 
Returns the predicted label yhat, according to logistic regression.
"""
def predict(x, w):
  # y_hat = predict(xi,w) # predict label for datapoints.  
    list_of_yhat = [] ; # store list of y hat y1 hat , y2 hat ,,, 
    for xi in x :  
       z = np.dot(w.T, xi) # do product of w^t * xi
       yhat = logisticFunction(z) # convert f(x) to pi(x) by calling logisticfunction() 
       list_of_yhat.append(yhat)
 
    return  list_of_yhat
      
    
"""
Receives data point (x), data label (y), and coefficient parameter values (w) 
Computes and returns the gradient of negative log likelihood at point (x)
"""
def gradient(x, y, w):
  # for logisticBegression_SGA gradient dleta(tildaw, tilda_xi) = (yi - etha(tildaW_transpo * tilda_xi)) * tilda_xi 
  # this gradient gives gradeint acent in random point xi 
  z = np.dot(w.T, x) # w.T is ugmented w transpo,  x is xi 
  y_hat = logisticFunction(z)  # etha(tildaW_transpo * tilda_xi)
  difference  = y - y_hat  # yi (actual) - y_hat ( prediect)
  gradient = difference * x 
  return gradient # return delta ( ugmented W, ugmented xi)


"""
Receives the predicted labels (y_hat), and the actual data labels (y)
Computes and returns the cross-entropy loss
"""
# used Logistic Regression ppt slide 10th formula to code cross_entropy
def cross_entropy(y_hat, y):
        # delat J(w) = X^t * (yhat - y) 
    N = len(y)
    error = 0
   
    for i in range(N):
        # ex) if y = 1 case 
        if np.all( y_hat[i] == 0):  # used .all() based on recommendation on terminal to fix error
            prop1 = 0
        else:
            prop1 = y[i] * np.log(1 / y_hat[i])
        
        if np.all(1 - y_hat[i] == 0):
            prop2 = 0
        else:
            prop2 = (1 - y[i]) * np.log(1 / (1 - y_hat[i]))
       
        error += prop1 + prop2  # sum over
    cross_ent =- error/N 
    return cross_ent
  



"""
Receives data set (X), dataset labels (y), learning rate (step size) (psi), stopping criterion (for change in norm of ws) (epsilon), and maximum number of epochs (max_epochs)
Computes and returns the vector of coefficient parameters (w), and the list of cross-entropy losses for all epochs (cross_ent)
"""
def logisticRegression_SGA(X, y, psi, epsilon, max_epochs):
    """
      TODO
      NOTE: remember to either shuffle the data points or select data points randomly in each internal iteration.
      NOTE: stopping criterion: stop iterating if norm of change in w (norm(w-w_old)) is less than epsilon, or the number of epochs (iterations over the whole dataset) is more than maximum number of epochs
    """ 
    # x : input datset , y : lables , psi : learning reate ( step size ) 
    # epsilon : stopping creiterion, max_epochs : maximum number of iteration. 
    X_transpo = X.T 
    attributes = X_transpo.shape[1] # ex) x1 column size. (d by 1 )
    w = np.zeros(attributes)
    
    cross_ent = [] 
    
    for count in range(max_epochs):  # for break point. 
        # get random indicies xi 
        np.random.shuffle(X_transpo)
        for i in range(X_transpo.shape[1]):  # Randomize data order
            xi = X_transpo[i]
         
            slope = gradient(xi, y, w) # compute gradient at xi 
            w_old = np.copy(w)  # make copy of tildaW transpo. 
            w = w +(psi * slope)  # learing rate * gradient
        
            y_hat = predict(xi,w) # predict label for datapoints. 
            cross_entropy_loss = cross_entropy(y_hat, y) # calculate cross entropy lose 
            cross_ent.append(cross_entropy_loss) # commpuate 
            
            if np.linalg.norm(w - w_old) < epsilon or count > max_epochs:  # stopping criterion
                # we did enough iteration or difference between w - wold is small -> stop 
                return w, cross_ent 
        
    return w,cross_ent
  
  
if __name__ == '__main__':  
    ## initializations and config
    psi=0.1 # learning rate or step size
    epsilon = 10 # used in SGA's stopping criterion to define the minimum change in norm of w
    max_epochs = 8 # used in SGA's stopping criterion to define the maximum number of epochs (iterations) over the whole dataset
    
    ## loading the data
    df_train = pd.read_csv("cancer-data-train.csv", header=None)
    df_test = pd.read_csv("cancer-data-test.csv", header=None)
    
    ## split into features and labels (firstcolum - before y colum), y colum 
    X_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1].astype("category").cat.codes #Convert string labels to numeric
    X_test, y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1].astype("category").cat.codes
    
    ## augmenting train data with 1 (X0) [x0,x1,x2 ,,, xd]
    X_train.insert(0,'',1)
    X_test.insert(0,'',1)
    
    X_train = X_train.to_numpy(); 
    y_train = y_train.to_numpy(); 
    ## learning logistic regression parameters
    [w, cross_ent] = logisticRegression_SGA(X_train, y_train, psi, epsilon, max_epochs)
    
    print("w : \n", w )
    print("cross ent : \n" , cross_ent)
    
    ## plotting the cross-entropy across epochs to see how it changes
    plt.plot(cross_ent, 'o', color='black');
    plt.xlabel('etha')
    plt.ylabel('cross entropy loss value')
    plt.title('cross entropy loss vs etha ( with psi=0.1, epsilon=10 max_epochs=8) ')
    plt.grid(True) 
    plt.show(); 
    
    
    """
      TODO: calculate and print the average cross-entropy error for training data (cross-entropy error/number of data points)
    """
    # get each colum from X_train dataset 
    # get each column from X_train dataset 
    print("w size : ", w.shape)
    
    list_of_train_y_hat = [] 
    for xi in X_train: 
        train_y_hat = predict(xi, w)
        list_of_train_y_hat.append(train_y_hat)
        
    print("size of train y hat : ", len(list_of_train_y_hat))
    print("size of y train : ", len(y_train))
    
    train_y_hat_array = np.array(list_of_train_y_hat)
    train_loss = cross_entropy(train_y_hat_array, y_train) 
    print("Average cross-entropy error for training data:", train_loss)
    
    """
      TODO: predict the labels for test data points using the learned w's
    """
    X_test = X_test.to_numpy(); 
    y_test = y_test.to_numpy(); 
    
    list_of_test_y_hat =[]
    for xi in X_test: 
      test_y_hat = predict(xi,  w)
      list_of_test_y_hat.append(test_y_hat)
    
    """
      TODO: calculate and print the average cross-entropy error for testing data (cross-entropy error/number of data points)
    """
    
    test_y_hat_array = np.array(list_of_test_y_hat)
    test_loss = cross_entropy(test_y_hat_array  ,  y_test)
    print("Average cross-entropy error for testing data:", test_loss)
    print("Finished First Task.\n")
    
 ###########################################################################################################
    """_test part 2 - (d) test with  different psi and epsilon values. 
    """
  
    ## initializations and config
    new_psi=0.14 # learning rate or step size
    new_epsilon = 8 # used in SGA's stopping criterion to define the minimum change in norm of w
    new_max_epochs = 8 # used in SGA's stopping criterion to define the maximum number of epochs (iterations) over the whole dataset
  
    ## learning logistic regression parameters
    [w, cross_ent] = logisticRegression_SGA(X_train, y_train, new_psi, new_epsilon, new_max_epochs)
    
    ## plotting the cross-entropy across epochs to see how it changes
    plt.plot(cross_ent, 'o', color='blue');  # ciatation! 
    plt.xlabel('etha')
    plt.ylabel('cross entropy loss value')
    plt.title('cross entropy loss vs etha ( with psi=0.14, epsilon=8 max_epochs=8) ')
    plt.grid(True) 
    plt.show(); 
    
    list_of_train_y_hat = [] 
    for xi in X_train: 
        train_y_hat = predict(xi, w)
        list_of_train_y_hat.append(train_y_hat)
        
    print("size of train y hat : ", len(list_of_train_y_hat))
    print("size of y train : ", len(y_train))
    
    train_y_hat_array = np.array(list_of_train_y_hat)
    train_loss = cross_entropy(train_y_hat_array, y_train) 
    print("Average cross-entropy error for training data:", train_loss)
    
    """
      TODO: predict the labels for test data points using the learned w's
    """
    
    list_of_test_y_hat =[]
    for xi in X_test: 
      test_y_hat = predict(xi,  w)
      list_of_test_y_hat.append(test_y_hat)
    
    """
      TODO: calculate and print the average cross-entropy error for testing data (cross-entropy error/number of data points)
    """
    
    test_y_hat_array = np.array(list_of_test_y_hat)
    test_loss = cross_entropy(test_y_hat_array  ,  y_test)
    print("Average cross-entropy error for testing data:", test_loss)
    
    
    
    
