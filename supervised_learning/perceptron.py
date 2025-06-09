from __future__ import print_function, division
import math
import numpy as np

# Import helper functions
from utils import train_test_split, to_categorical, normalize, accuracy_score
from deep_learning.activation_functions import Sigmoid, ReLU, SoftPlus, LeakyReLU, TanH, ELU
from deep_learning.loss_functions import CrossEntropy, SquareLoss
from utils import Plot
from utils.misc import bar_widgets
import progressbar


class Perceptron():

    def __init__(self, n_iterations = 20000, activation_function=Sigmoid, loss=SquareLoss, learning_rate=0.01):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.loss = loss()
        self.activation_func = activation_function()
        self.progressbar = progressbar.ProgressBar(widgets = bar_widgets)

    def fit(self, X, y):
        n_samples, n_features = np.shape(X)
        _, n_outputs = np.shape(y)

        limit = 1 / math.sqrt(n_features)

        self.W = np.random.uniform(-limit, limit, (n_features, n_outputs))
        self.w0 = np.zeros((1, n_outputs))

        for i in self.progressbar(range(self.n_iterations)):

            linear_output = X.dot(self.W) + self.w0

            y_pred = self.activation_func(linear_output)

            if self.activation_func.__class__.__name__ == "Softmax":
                error_gradient = self.loss.gradient(y, y_pred)
            else:
                error_gradient = self.loss.gradient(y, y_pred) * self.activation_func.gradient(linear_output)

                # Calculate the gradient of the loss with respect to weights
            grad_wrt_w = X.T.dot(error_gradient)


            # X = [n, d]
            #  [n, m] = error gradient 

            # [d, m]

            
            

            grad_wrt_w0 = np.sum(error_gradient, axis=0, keepdims=True)

            # Update weights
            self.W  -= self.learning_rate * grad_wrt_w
            self.w0 -= self.learning_rate * grad_wrt_w0






