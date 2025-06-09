from __future__ import print_function, division
from terminaltables import AsciiTable
import numpy as np
import progressbar
from utils import batch_iterator
from utils.misc import bar_widgets



class NeuralNetwork():
    """Neural Network. Deep Learning base model.

    Parameters:
    -----------
    optimizer: class
        The weight optimizer that will be used to tune the weights in order of minimizing
        the loss.
    loss: class
        Loss function used to measure the model's performance. SquareLoss or CrossEntropy.
    validation: tuple
        A tuple containing validation data and labels (X, y)
    """

    def __init__(self, optimizer, loss, validation_data=None):
        self.optimizer = optimizer
        self.layers = []
        self.errors = {"training": [], "validation": []}
        self.loss_function = loss()
        self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)

        self.val_set = None
        if validation_data:
            X, y = validation_data
            self.val_set = {"X": X, "y": y}

    def set_trainable(self, trainable):
        for layer in self.layers:
            layer.trainable = trainable
    

    def add(self, layer):
        if self.layers:
            layer.set_input_shape(shape = self.layers[-1].output_shape())

        if hasattr(layer, 'initialize'):
            layer.initialize(optimizer = self.optimizer)

        self.layers.append(layer)

    def test_on_batch(self, X, y):
        y_pred = self._forward_pass(X)

        loss = np.mean(self.loss_function.loss(y,y_pred))

        acc = self.loss_function.acc(y, y_pred)

        loss_grad = self.loss_function.gradient(y, y_pred)

        self._backward_pass(loss_grad = loss_grad)

        return loss, acc   
    
    def fit(self, X, y, n_epochs, batch_size):

        for _ in self.progressbar(range(n_epochs)):
            batch_error = []
            for X_batch, y_batch in batch_iterator(X,y, batch_size = batch_size):
                loss, _ = self.train_on_batch(X_batch, y_batch)
                batch_error.append(loss)
            self.errors["training"].append(np.mean(batch_error))

            if self.val_set is not None:
                val_loss, _ = self.test_on_batch(self.val_set["X"], self.val_set["y"])
                self.errors["validation"].append(val_loss)
        
        return self.errors["training"], self.errors["validation"]
    
    def _forward_pass(self, X, training = True):

        layer_output = X
        for layer in self.layers: 
            layer_output = layer.forward_pass(layer_output, training)
        return layer_output
    
    def _backward_pass(self, loss_grad):
        for layer in reversed(self.layers):
            loss_grad = layer.backward_pass(loss_grad)

    def summary(self, name = "Model Summary"):
        print(AsciiTable([name]).table)

        print ("Input Shape: %s" % str(self.layers[0].input_shape))

        table_data = [["Layer Type", "Parameters", "Output Shape"]]
        tot_params = 0
        for layer in self.layers:
            layer_name = layer.layer_name()
            params = layer.parameters()
            out_shape = layer.output_shape()
            table_data.append([layer_name, str(params), str(out_shape)])
            tot_params += params
        # Print network configuration table
        print (AsciiTable(table_data).table)
        print ("Total Parameters: %d\n" % tot_params)
    

    def predict(self, X):
        """ Use the trained model to predict labels of X """
        return self._forward_pass(X, training=False)




