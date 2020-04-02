import numpy as np
import pandas as pd
import math
from scipy import optimize

class machine_learning_model:
    
    features = 0
    features_size = 0
    labels = 0
    label_size = 0
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
        self.label_size = labels.size
                
        #inserts a column of ones to the features dataframe
        self.features = np.insert(features, [0], np.ones((self.label_size,1)), axis = 1)
        self.features_size = self.features[0].size
        
        self.theta = np.zeros((self.features_size,1))
    
            
        
    def _sigmoid(self, X):
        '''
        __________________________________________________________________________
        
        Returns the sigmoid of the scaler/matrix x
        __________________________________________________________________________

        '''
        X @ self.theta
    
        return 1.0 / ( 1.0 + np.exp(-z))

class linear_regression(machine_learning_model):
    mu = 0
    sigma = 0
    _features_normalized = False
    
    def __init__(self, features, labels):
        super().__init__(features, labels)
    
        
    def normalize_features(self):
        self.mu = np.zeros(self.features_size)
        self.sigma = np.zeros(self.features_size)
        #normalizes all features to values between -1 and 1 to speed up the gradient descent
        #methode
        
        #sets columns of mu to the feature columns mean and sigmas columns to its standard deviations
        self.mu = np.mean(self.features, axis=0)
        self.sigma = np.std(self.features, axis=0)
        
        #normalizes the feature columns
        for i in range(1, self.features_size):
            self.features[:,i] = (self.features[:,i] - self.mu[i]) / self.sigma[i]
        
        #indicates that the features got normalized for the prediction methode 
        self._features_normalized = True

    def compute_cost(self, regularization_parameter = 0):
        #computes how cost efficient theta is and how far off it is from the actual result,
        #usefull for checking if the gradient descents convergences
        temp_matrix = (self.features @ self.theta) - self.labels
        theta_reg = self.theta.copy()
        theta_reg[0,0] = 0
        term = temp_matrix.T @ temp_matrix
        regularization_term = regularization_parameter * (theta_reg.T @ theta_reg )
        return (1/(2*self.label_size) * (term + regularization_term))[0][0]
            
    def gradient_descent(self, alpha, num_iterations, regularization_parameter = 0):
        self.theta = np.zeros((self.features_size,1))
        #finds the right values for theta
        for i in range(1, num_iterations):
            temp_matrix = (self.features @ self.theta) - self.labels
            temp_matrix = self.features.T @ temp_matrix
            self.theta = self.theta * (1 - alpha * (regularization_parameter / self.label_size)) - (alpha / self.label_size) * temp_matrix
        
        
    def normal_equation(self):
        self.theta = np.zeros((self.features_size,1))
        
        #finds the right values for theta, doesn't need feature normalization, slow if features > 10000
        
        XTXinverse = np.linalg.inv(self.features.T @ self.features)
        self.theta = XTXinverse @ (self.features.T @ self.labels)
    
    def predict(self, X):
        try:
            #converts X into a numpy array
            if isinstance(X, list):
                X = np.array(X)[np.newaxis][0]
            if X.size == self.features_size-1:
                #normalizes the values to predict, then predicts
                if self._features_normalized:
                    scaledMatrix = np.ones((1, self.features_size))[0]
                    for i in range(0, X.size):
                        scaledMatrix[i+1] = (X[i] - self.mu[i+1]) / self.sigma[i+1]
                    
                    return (scaledMatrix @ self.theta)[0]
                
                #predicts if the features are not normalized
                else:
                    X = np.insert(X, [0], 1)
                    return (X @ self.theta)[0]

            
            else:
                print("insert the right amount of features")
        except:
            print("only insert 1D numpy arrays or lists")


class logistic_regression(machine_learning_model):
    '''
    __________________________________________________________________________
        
    This is a classifier model, use it predict 2 different classes
    __________________________________________________________________________

    '''
        
    def __init__(self, features, labels):
        super().__init__(features, labels)
    
    def compute_cost(self, regularization_parameter = 0):
        '''
        __________________________________________________________________________
        
        Computes the cost of theta, use the regularization_parameter to regularize
        if your have to many features
        __________________________________________________________________________

        '''
        h = self._sigmoid(self.features, self.theta)
        theta_reg = self.theta.copy()
        theta_reg[0,0] = 0
        term_1 = -self.labels.T @ np.log(h)
        term_2 =  (1 - self.labels).T @ np.log(1 - h)
        regularization_term = regularization_parameter / (2 * self.label_size) * (theta_reg.T @ theta_reg)
        return (1 / self.label_size * (term_1 - term_2) + regularization_term)[0][0]
    
    def gradient_descent(self, alpha, num_iterations = 20000, regularization_parameter = 0):
        '''
        __________________________________________________________________________
        
        Finds the optimal values for theta, needs a high number of iterations,
        num_iterations < 20,000 is recommended
        __________________________________________________________________________

        '''
        self.theta = np.zeros((self.features_size,1))
        #finds the right values for theta
        cost_history = []
        
        for i in range(1, num_iterations):
            h = self._sigmoid(self.features, self.theta)
            error = h - self.labels
            grad = self.features.T @ error
            regularization_term = (regularization_parameter / self.label_size) * self.theta
            self.theta = self.theta - alpha * ((1 / self.label_size) * (grad  + regularization_term))
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

class multi_class_classifier(logistic_regression):
    def __init__(self, features, labels):
        super().__init__(features, labels)
    
    def one_vs_all(self, regularization_parameter = 0):
        pass