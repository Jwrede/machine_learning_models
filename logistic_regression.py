import numpy as np

from machine_learning_models import machine_learning_models

class logistic_regression(machine_learning_models):
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
        h = self._sigmoid(self.features)
        theta_reg = self.theta.copy()
        theta_reg[0,0] = 0
        term_1 = -self.labels * np.log(h)
        term_2 =  (1 - self.labels) * np.log(1 - h)
        regularization_term = regularization_parameter / (2 * self.label_size) * (theta_reg.T @ theta_reg)
        return (1 / self.label_size * (term_1 - term_2) + regularization_term)[0][0]
    
    def gradient_descent(self, alpha, num_iterations = 20000, regularization_parameter = 0):
        '''
        __________________________________________________________________________
        
        Finds the optimal values for theta, needs a high number of iterations,
        num_iterations < 20,000 is recommended
        __________________________________________________________________________

        '''
        self.theta = np.zeros((self.num_features,1))
        #finds the right values for theta
        cost_history = []
        
        for i in range(1, num_iterations):
            h = self._sigmoid(self.features)
            error = h - self.labels
            grad = self.features.T @ error
            regularization_term = (regularization_parameter / self.label_size) * self.theta
            self.theta = self.theta - alpha * ((1 / self.label_size) * (grad  + regularization_term))
            cost_history.append(self.compute_cost(regularization_parameter))
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

