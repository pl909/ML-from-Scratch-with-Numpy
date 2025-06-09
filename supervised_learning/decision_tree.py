from __future__ import division, print_function
import numpy as np

from utils import divide_on_feature, train_test_split, standardize, mean_squared_error
from utils import calculate_entropy, accuracy_score, calculate_variance


class DecisionNode():
    """Class that represents a decision node or leaf in the decision tree

    Parameters:
    -----------
    feature_i: int
        Feature index which we want to use as the threshold measure.
    threshold: float
        The value that we will compare feature values at feature_i against to
        determine the prediction.
    value: float
        The class prediction if classification tree, or float value if regression tree.
    true_branch: DecisionNode
        Next decision node for samples where features value met the threshold.
    false_branch: DecisionNode
        Next decision node for samples where features value did not meet the threshold.
    """
    def __init__(self, feature_i=None, threshold=None,
                 value=None, true_branch=None, false_branch=None):
        self.feature_i = feature_i          # Index for the feature that is tested
        self.threshold = threshold          # Threshold value for feature
        self.value = value                  # Value if the node is a leaf in the tree
        self.true_branch = true_branch      # 'Left' subtree
        self.false_branch = false_branch    # 'Right' subtree
    

class DecisionTree():

    """ super of regressiontree and classification tree"""
    def __init__(self, min_samples_split=2, min_impurity=1e-7,
                 max_depth=float("inf"), loss=None):
        self.root = None
        # minimum of n samoples to justify split
        self.min_samples_split = min_samples_split
        # minimum impurity justify split 
        self.min_impurity = min_impurity
        self.max_depth = max_depth

        self._impurity_calculation = None

        self._leaf_value_calculation = None

        self.one_dim = None 

        self.loss = loss
    
    def fit(self, X, y, loss = None):
        self.one_dim = len(np.shape(y))==1
        self.root = self._build_tree(X,y)
        self.loss = None

    def _build_tree(self, X, y, current_depth = 0):
        largest_impurity = 0

        best_criteria = None # Feature index and threshold

        best_sets = None # subsets of the data

        if len(np.shape(y))==1:
            y = np.expand_dims(y,axis=1)
        

        Xy = np.concatenate((X,y), axis = 1)

        n_samples, n_features = np.shape(X)

        if n_samples >= self.min_samples_split and current_depth<=self.max_depth:

            for feature_i in range(n_features):
                # All values of feature_i
                feature_values = np.expanddims(X[:,feature_i], axis=1)
                unique_values = np.unique(feature_values)
                for threshold in unique_values:
                    Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)

                    if len(Xy1) > 0 and len(Xy2) > 0:
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]

                        impurity = self._impurity_calculation(y, y1, y2)

                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {"feature_i": feature_i, "threshold": threshold}
                            best_sets = {
                                "leftX": Xy1[:, :n_features],   # X of left subtree
                                "lefty": Xy1[:, n_features:],   # y of left subtree
                                "rightX": Xy2[:, :n_features],  # X of right subtree
                                "righty": Xy2[:, n_features:]   # y of right subtree
                            }
            if largest_impurity > self.min_impurity:
                # Build subtrees for the right and left branches
                true_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
                false_branch = self._build_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1)
                return DecisionNode(feature_i=best_criteria["feature_i"], threshold=best_criteria[
                                    "threshold"], true_branch=true_branch, false_branch=false_branch)
            
            leaf_value = self._leaf_value_calculation(y)

            return DecisionNode(value=leaf_value)
        
    def predict_value(self, x, tree=None):

        if tree is None:
            tree = self.root

        # If we have a value (i.e we're at a leaf) => return value as the prediction
        if tree.value is not None:
            return tree.value

        feature_value = x[tree.feature_i]

        branch = tree.false_branch

        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.true_branch
        elif feature_value == tree.threshold:
            branch = tree.true_branch

        return self.predict_value(x, branch)
                        


    def predict(self, X):
        """ Classify samples one by one and return the set of labels """
        y_pred = [self.predict_value(sample) for sample in X]
        return y_pred

    def print_tree(self, tree=None, indent=" "):
        """ Recursively print the decision tree """
        if not tree:
            tree = self.root

        # If we're at leaf => print the label
        if tree.value is not None:
            print (tree.value)
        # Go deeper down the tree
        else:
            # Print test
            print ("%s:%s? " % (tree.feature_i, tree.threshold))
            # Print the true scenario
            print ("%sT->" % (indent), end="")
            self.print_tree(tree.true_branch, indent + indent)
            # Print the false scenario
            print ("%sF->" % (indent), end="")
            self.print_tree(tree.false_branch, indent + indent)


class ClassificationTree(DecisionTree):
    def _calculate_information_gain(self, y, y1, y2):
        # Calculate information gain
        p = len(y1) / len(y)
        entropy = calculate_entropy(y)
        info_gain = entropy - p * \
            calculate_entropy(y1) - (1 - p) * \
            calculate_entropy(y2)

        return info_gain

    def _majority_vote(self, y):
        most_common = None
        max_count = 0
        for label in np.unique(y):
            # Count number of occurences of samples with label
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        return most_common

    def fit(self, X, y):
        self._impurity_calculation = self._calculate_information_gain
        self._leaf_value_calculation = self._majority_vote
        super(ClassificationTree, self).fit(X, y)