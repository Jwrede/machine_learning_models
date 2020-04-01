import numpy as np
import pandas as pd
import math
from scipy import optimize

class machine_learning_model:
    
    features = 0
    featuresSize = 0
    labels = 0
    labelSize = 0
    theta = 0
    
    def __init__(self, features, labels):
        #converts a label Dataframe to a Series for compatibility
        if isinstance(labels, pd.DataFrame):
            labels = labels.iloc[:,0]
            
        #converts a pandas dataframe to a numpy array, inserts a column of ones to the features dataframe
        #and initializes mu, sigma and theta to a matrix of different dimensions with just zeros
        if isinstance(features, pd.DataFrame) or isinstance(labels, pd.DataFrame):
            features = features.to_numpy(dtype = "float"); labels = labels.to_numpy(dtype = "float")[np.newaxis].T
        self.labels = labels
        self.labelSize = labels.size
                
        #inserts a column of ones to the features dataframe
        self.features = np.insert(features, [0], np.ones((self.labelSize,1)), axis = 1)
        self.featureSize = self.features[0].size
        
        self.theta = np.zeros((self.featureSize,1))

class linearRegression(machine_learning_model):
    mu = 0
    sigma = 0
    _featuresNormalized = False
    
    def __init__(self, features, labels):
        super().__init__(features, labels)
    
        
    def normalizeFeatures(self):
        self.mu = np.zeros(self.featureSize)
        self.sigma = np.zeros(self.featureSize)
        #normalizes all features to values between -1 and 1 to speed up the gradient descent
        #methode
        
        #sets columns of mu to the feature columns mean and sigmas columns to its standard deviations
        self.mu = np.mean(self.features, axis=0)
        self.sigma = np.std(self.features, axis=0)
        
        #normalizes the feature columns
        for i in range(1, self.featureSize):
            self.features[:,i] = (self.features[:,i] - self.mu[i]) / self.sigma[i]
        
        #indicates that the features got normalized for the prediction methode 
        self._featuresNormalized = True

    def computeCost(self, regularization_parameter = 0):
        #computes how cost efficient theta is and how far off it is from the actual result,
        #usefull for checking if the gradient descents convergences
        tempMatrix = np.dot(self.features,self.theta) - self.labels
        theta_reg = self.theta.copy()
        theta_reg[0,0] = 0
        term = np.dot(tempMatrix.T, tempMatrix)
        regularization_term = regularization_parameter * np.dot(theta_reg.T, theta_reg )
        return (1/(2*self.labelSize) * (term + regularization_term))[0][0]
            
    def gradientDescent(self, alpha, num_iterations, regularization_parameter = 0):
        self.theta = np.zeros((self.featureSize,1))
        #finds the right values for theta
        for i in range(1, num_iterations):
            tempMatrix = np.dot(self.features,self.theta) - self.labels
            tempMatrix = np.dot(self.features.T, tempMatrix)
            self.theta = self.theta * (1 - alpha * (regularization_parameter / self.labelSize)) - (alpha / self.labelSize) * tempMatrix
        
        
    def normalEquation(self):
        self.theta = np.zeros((self.featureSize,1))
        
        #finds the right values for theta, doesn't need feature normalization, slow if features > 10000
        
        XTXinverse = np.linalg.inv(np.dot(self.features.T, self.features))
        self.theta = np.dot(XTXinverse, np.dot(self.features.T, self.labels))
    
    def predict(self, X):
        try:
            #converts X into a numpy array
            if isinstance(X, list):
                X = np.array(X)[np.newaxis][0]
            if X.size == self.featureSize-1:
                #normalizes the values to predict, then predicts
                if self._featuresNormalized:
                    scaledMatrix = np.ones((1, self.featureSize))[0]
                    for i in range(0, X.size):
                        scaledMatrix[i+1] = (X[i] - self.mu[i+1]) / self.sigma[i+1]
                    
                    return (np.dot(scaledMatrix,self.theta))[0]
                
                #predicts if the features are not normalized
                else:
                    X = np.insert(X, [0], 1)
                    return (np.dot(X, self.theta))[0]

            
            else:
                print("insert the right amount of features")
        except:
            print("only insert 1D numpy arrays or lists")


class logisticRegression(machine_learning_model):
    '''
    __________________________________________________________________________
        
    This is a classifier model, use it predict 2 different classes
    __________________________________________________________________________

    '''
        
    def __init__(self, features, labels):
        super().__init__(features, labels)
        
        
    def _sigmoid(self, X, theta):
        '''
        __________________________________________________________________________
        
        Returns the sigmoid of the scaler/matrix x
        __________________________________________________________________________

        '''
        z = np.dot(X, self.theta)
    
        return 1.0 / ( 1.0 + np.exp(-z))
    
    def computeCost(self, regularization_parameter = 0):
        '''
        __________________________________________________________________________
        
        Computes the cost of theta, use the regularization_parameter to regularize
        if your have to many features
        __________________________________________________________________________

        '''
        h = self._sigmoid(self.features, self.theta)
        theta_reg = self.theta.copy()
        theta_reg[0,0] = 0
        term_1 = np.dot(-self.labels.T, np.log(h))
        term_2 =  np.dot((1 - self.labels).T, np.log(1 - h))
        regularization_term = regularization_parameter / (2 * self.labelSize) * np.dot(theta_reg.T, theta_reg)
        return (1 / self.labelSize * (term_1 - term_2) + regularization_term)[0][0]
    
    def gradientDescent(self, alpha, num_iterations = 20000, regularization_parameter = 0):
        '''
        __________________________________________________________________________
        
        Finds the optimal values for theta, needs a high number of iterations,
        num_iterations < 20,000 is recommended
        __________________________________________________________________________

        '''
        self.theta = np.zeros((self.featureSize,1))
        #finds the right values for theta
        cost_history = []
        
        for i in range(1, num_iterations):
            h = self._sigmoid(self.features, self.theta)
            error = h - self.labels
            grad = np.dot(self.features.T, error)
            regularization_term = (regularization_parameter / self.labelSize) * self.theta
            self.theta = self.theta - alpha * ((1 / self.labelSize) * (grad  + regularization_term))
            cost_history.append(self.computeCost(regularization_parameter))
        return cost_history
        
    def predict(self, X):
        '''
        __________________________________________________________________________
        
        predicts the class of the set of values for X
        __________________________________________________________________________

        '''
        if isinstance(X,list):
            X = np.array([X])
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype = "float")
        X = np.insert(X, [0], np.ones([X[:,0].size,1]), axis = 1)
        prediction = self._sigmoid(X, self.theta)
        return np.where(prediction > 0.5,1,0)
        