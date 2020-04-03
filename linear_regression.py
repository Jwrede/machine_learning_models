import numpy as np

class linear_regression(machine_learning_models):
    mu = 0
    sigma = 0
    _features_normalized = False
    
    def __init__(self, features, labels):
        super().__init__(features, labels)
    
        
    def normalize_features(self):
        self.mu = np.zeros(self.num_features)
        self.sigma = np.zeros(self.num_features)
        #normalizes all features to values between -1 and 1 to speed up the gradient descent
        #methode
        
        #sets columns of mu to the feature columns mean and sigmas columns to its standard deviations
        self.mu = np.mean(self.features, axis=0)
        self.sigma = np.std(self.features, axis=0)
        
        #normalizes the feature columns
        for i in range(1, self.num_features):
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
        self.theta = np.zeros((self.num_features,1))
        #finds the right values for theta
        for i in range(1, num_iterations):
            temp_matrix = (self.features @ self.theta) - self.labels
            temp_matrix = self.features.T @ temp_matrix
            self.theta = self.theta * (1 - alpha * (regularization_parameter / self.label_size)) - (alpha / self.label_size) * temp_matrix
        
        
    def normal_equation(self):
        self.theta = np.zeros((self.num_features,1))
        
        #finds the right values for theta, doesn't need feature normalization, slow if features > 10000
        
        XTXinverse = np.linalg.inv(self.features.T @ self.features)
        self.theta = XTXinverse @ (self.features.T @ self.labels)
    
    def predict(self, X):
        try:
            #converts X into a numpy array
            if isinstance(X, list):
                X = np.array(X)[np.newaxis][0]
            if X.size == self.num_features-1:
                #normalizes the values to predict, then predicts
                if self._features_normalized:
                    scaledMatrix = np.ones((1, self.num_features))[0]
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


