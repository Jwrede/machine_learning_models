import numpy as np
import pandas as pd
import math
from scipy import optimize

class machine_learning_models:
    
    features = 0
    num_features = 0
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
        self.num_features = self.features[0].size
        
        self.theta = np.zeros((self.num_features,1))
    
            
        
    def _sigmoid(self, X):
        '''
        __________________________________________________________________________
        
        Returns the sigmoid of the scaler/matrix x
        __________________________________________________________________________

        '''
        z = X @ self.theta
    
        return 1.0 / ( 1.0 + np.exp(-z))
