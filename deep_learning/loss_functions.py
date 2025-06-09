from __future__ import division
import numpy as np
from utils import accuracy_score
from deep_learning.activation_functions import Sigmoid


class Loss():
    def loss(self, y_true, y_pred):
        return NotImplementedError
    
    def gradient(self, y, y_pred):
        raise NotImplementedError
    
    def acc(self, y, y_pred):
        return 0

class SquareLoss(Loss):
    def __init__(self): pass

    def loss(self, y, y_pred):
        return 0.5 * np.power((y-y_pred), 2)
    
    def gradient(self, y, y_pred):
        return -(y-y_pred)
    import numpy as np
class CrossEntropy:
    def __init__(self):
        pass

    def loss(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)

        if y.shape[1] == 1:
            # Binary sigmoid → expand to two-class softmax form
            y = np.hstack([1 - y, y])
            p = np.hstack([1 - p, p])

        # Cross-entropy loss over all classes
        return -np.mean(np.sum(y * np.log(p), axis=1))

    def gradient(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)

        if y.shape[1] == 1:
            y = np.hstack([1 - y, y])
            p = np.hstack([1 - p, p])

        # Softmax + cross-entropy derivative: p - y
        return p - y

    def acc(self, y, p):
        # Convert both to class indices
        y_labels = np.argmax(y, axis=1)
        p_labels = np.argmax(p, axis=1)
        return np.mean(y_labels == p_labels)
