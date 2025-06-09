from __future__ import print_function, division
import numpy as np
import math
from utils import normalize, polynomial_features

class l1_regularization():
    """Regularization for Lasso regression"""
    def __init__(self, alpha):
        self.alpha = alpha

        def __call__(self, w):
            return self.alpha * np.linalg.norm(w, ord=1)
        
        def grad(self, w):
            return self.alpha * np.sign(w)
        
class l2_regularization():
    def __init__(self, alpha):
        self.alpha = alpha
        
    def __call__(self, w):
        return self.alpha * 0.5 * np.linalg.norm(w)**2

    def grad(self, w):
        return self.alpha * w

class l1_l2_regularization():
    """ Regularization for Elastic Net Regression """
    def __init__(self, alpha, l1_ratio=0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def __call__(self, w):
        l1_contr = self.l1_ratio * np.linalg.norm(w)
        l2_contr = (1 - self.l1_ratio) * 0.5 * w.T.dot(w) 
        return self.alpha * (l1_contr + l2_contr)

    def grad(self, w):
        l1_contr = self.l1_ratio * np.sign(w)
        l2_contr = (1 - self.l1_ratio) * w
        return self.alpha * (l1_contr + l2_contr) 
    

class Regression:
    """
    Base regression models

    Models relationship between scalar depedendent variable y and independent variables X.

    Parameters:
    n_iterations: float
    number of iterations the algorithm will tune the weights for. 

    learning_rate: float
    Step length to use when updating the weights
    """

    def __init__(self, n_iterations, learning_rate):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def initialize_weights(self, n_features):
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features, ))

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.training_errors = []
        self.initialize_weights(n_features = X.shape[1])

        for i in range(self.n_iterations):
            y_pred = X.dot(self.w)

            mse = np.mean(0.5 ** (y-y_pred)**2 + self.regularization(self.w))

            self.training_errors.append(mse)

            grad_w = -(y-y_pred).dot(X) + self.regularization.grad(self.w)

            self.w -= self.learning_rate * grad_w

    def predict(self,X):
        X = np.insert(X,0,1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred 
    
class LinearRegression(Regression):

    def __init__(self, n_iterations=100, learning_rate=0.001, gradient_descent=True):
        self.gradient_descent = gradient_descent
        self.regularization = lambda x: 0
        self.regularization.grad  = lambda x: 0
        super(LinearRegression, self).__init__(n_iterations=n_iterations,
                                            learning_rate=learning_rate)
    def fit(self, X, y):

        if not self.gradient_descent:

            # w = X.T.dot(X) ^ -1 .dot(X.T).dot(y) is the analytical solution 
            X = np.insert(X,0,1,axis=1)
            U, S, V = np.linalg.svd(X.T.dot(X))
            S = np.diag(S)
            X_sq_reg_inv = V.dot(np.linalg.pinv(S)).dot(U.T)
            self.w = X_sq_reg_inv.dot(X.T).dot(y)
        else: 
            super(LinearRegression, self).fit(X,y)

class LassoRegression(Regression):
    """
    l1 regularized regression

    """

    def __init__(self, degree, reg_factor, n_iterations = 3000, learning_rate=0.01):

        self.degree = degree
        self.regularization = l1_regularization(alpha = reg_factor)
        super(LassoRegression, self).__init__(n_iterations, learning_rate)

    def fit(self, X, y):
        X = normalize(polynomial_features(X,degree=self.degree))
        super(LassoRegression, self).fit(X,y)


class PolynomialRegression(Regression):
    """
        The step length that will be used when updating the weights.
    """
    def __init__(self, degree, n_iterations=3000, learning_rate=0.001):
        self.degree = degree
        # No regularization
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super(PolynomialRegression, self).__init__(n_iterations=n_iterations,
                                                learning_rate=learning_rate)

    def fit(self, X, y):
        X = polynomial_features(X, degree=self.degree)
        super(PolynomialRegression, self).fit(X, y)

    def predict(self, X):
        X = polynomial_features(X, degree=self.degree)
        return super(PolynomialRegression, self).predict(X)    
    
class RidgeRegression(Regression):
    """
    -----------
    reg_factor: float
        The factor that will determine the amount of regularization and feature
        shrinkage. 
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """
    def __init__(self, reg_factor, n_iterations=1000, learning_rate=0.001):
        self.regularization = l2_regularization(alpha=reg_factor)
        super(RidgeRegression, self).__init__(n_iterations, 
                                            learning_rate)



class ElasticNet(Regression):
    """ Regression where a combination of l1 and l2 regularization are used. The
    ratio of their contributions are set with the 'l1_ratio' parameter.
    
    """
    def __init__(self, degree=1, reg_factor=0.05, l1_ratio=0.5, n_iterations=3000, 
                learning_rate=0.01):
        self.degree = degree
        self.regularization = l1_l2_regularization(alpha=reg_factor, l1_ratio=l1_ratio)
        super(ElasticNet, self).__init__(n_iterations, 
                                        learning_rate)

    def fit(self, X, y):
        X = normalize(polynomial_features(X, degree=self.degree))
        super(ElasticNet, self).fit(X, y)

    def predict(self, X):
        X = normalize(polynomial_features(X, degree=self.degree))
        return super(ElasticNet, self).predict(X)
    

    