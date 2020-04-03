import numpy as np

from machine_learning_models import machine_learning_models

class multi_class_classifier(machine_learning_models):
    
    all_theta = []
    
    def __init__(self, features, labels):
        super().__init__(features, labels)
        
    def _compute_cost(self, regularization_parameter = 0):
        '''
        __________________________________________________________________________
        
        Computes the cost of theta, use the regularization_parameter to regularize
        if your have to many features
        __________________________________________________________________________

        '''
        h = self._sigmoid(self.features)
        theta_reg = self.theta.copy()
        theta_reg[0] = np.zeros((1, theta_reg[0].size))
        term_1 = -self.labels * np.log(h)
        term_2 =  (1 - self.labels) * np.log(1 - h)
        regularization_term = regularization_parameter / (2 * self.label_size) * sum(self.theta[1:] ** 2)
        return (1 / self.label_size * sum(term_1 - term_2) + regularization_term)[0]
        
    def _gradient_descent(self, y, alpha, num_iterations = 20000, regularization_parameter = 0):
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
            error = h - y
            grad = self.features.T @ error
            regularization_term = (regularization_parameter / self.label_size) * self.theta
            self.theta = self.theta - alpha * ((1 / self.label_size) * (grad  + regularization_term))
            cost_history.append(self._compute_cost(regularization_parameter))
        return cost_history
    
    def one_vs_all(self, num_classes, alpha, num_iterations, regularization_parameter = 0):
        self.all_theta = []
        all_J = []
        
        for i in range(1, num_classes + 1):
            J_history = self._gradient_descent(np.where(self.labels == i, 1, 0), alpha, num_iterations, regularization_parameter)
            self.all_theta.extend(self.theta)
            all_J.extend(J_history)
        self.all_theta = np.array(self.all_theta).reshape(num_classes, self.num_features)
        return all_J
    
    def predict(self, X):
        m = X.shape[0]
        X = np.hstack((np.ones((m, 1)), X))
        
        prediction = X @ self.all_theta.T
        return np.argmax(prediction, axis = 1) + 1
