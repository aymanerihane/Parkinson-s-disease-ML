import numpy as np
import pandas as pd

# Function to calculate entropy
def entropy(y):
    # Ensure y is a pandas Series
    y = y.round().astype(int)
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

# Calculate information gain
def information_gain(y, y_left, y_right):
    parent_entropy = entropy(y)
    n = len(y)
    n_left, n_right = len(y_left), len(y_right)
    weighted_entropy = (n_left / n) * entropy(y_left) + (n_right / n) * entropy(y_right)
    return parent_entropy - weighted_entropy

# Node class for storing splits
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

# DecisionTree class
class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.root = None

    # Find the best split for the data
    def best_split(self, X, y):
        best_gain = -1
        split_feature, split_threshold = None, None
        left_split, right_split = None, None

        for feature in X.columns:
            thresholds = X[feature].unique()
            for threshold in thresholds:
                y_left = y[X[feature] <= threshold]
                y_right = y[X[feature] > threshold]
                if len(y_left) > 0 and len(y_right) > 0:
                    gain = information_gain(y, y_left, y_right)
                    if gain > best_gain:
                        best_gain = gain
                        split_feature, split_threshold = feature, threshold
                        left_split, right_split = (X[X[feature] <= threshold], y_left), (X[X[feature] > threshold], y_right)

        return split_feature, split_threshold, left_split, right_split

    # Recursively build the tree
    def build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or depth == self.max_depth:
            leaf_value = y.value_counts().idxmax()
            return Node(value=leaf_value)

        feature, threshold, (X_left, y_left), (X_right, y_right) = self.best_split(X, y)
        if feature is None:
            leaf_value = y.value_counts().idxmax()
            return Node(value=leaf_value)

        left_child = self.build_tree(X_left, y_left, depth + 1)
        right_child = self.build_tree(X_right, y_right, depth + 1)
        return Node(feature, threshold, left_child, right_child)

    # Fit the decision tree model
    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    # Predict class for a single example
    def predict_example(self, tree, example):
        if tree.is_leaf_node():
            return tree.value
        feature_value = example[tree.feature]
        if feature_value <= tree.threshold:
            return self.predict_example(tree.left, example)
        else:
            return self.predict_example(tree.right, example)

    # Predict class for multiple examples
    def predict(self, X):
        return X.apply(lambda x: self.predict_example(self.root, x), axis=1)