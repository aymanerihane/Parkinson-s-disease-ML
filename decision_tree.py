"""
Description: Implémentation de l'algorithme de l'arbre de décision
"""

import numpy as np
from collections import Counter
import graphviz

# Fonction pour calculer l'entropie
# y: vecteur de labels
def entropy(y):
    print("y type:", type(y))
    # y = np.asarray(y)  # Ensure y is a numpy array
    # y = y.astype(float)  # Convert to float first if it's not already numeric
    y = y.round().astype(int) 
    hist = np.bincount(y)
    ps = hist / len(y) # calcul des probabilités
    return -np.sum([p*np.log2(p) for p in ps if p > 0]) # calcul de l'entropie


# Classe de nœud de l’arbre 
class Node:
    def __init__(self, feature=None, threshold=None, right=None, left = None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.right = right
        self.left = left
        self.value = value
    
    def is_leaf_node(self):
        return self.value is not None
    
# Classe de l'arbre de décision
class DecisionTree:
    def __init__(self,min_samples_split=2,max_depth=100,n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self,X,y):
        print("Unique values in y:", np.unique(y))
        if not np.all(np.isin(y, [0, 1])):
            print("Warning: y contains values other than 0 and 1")

        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats,X.shape[1])
        self.root = self._grow_tree(X,y)

    def _grow_tree(self,X,y,depth=0):
        n_samples,n_features = X.shape
        n_labels = len(np.unique(y))

        #stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value) #leaf node
        
        feat_idxs = np.random.choice(n_features,self.n_feats,replace=False)
        

        #greedy search
        best_feat,best_thresh = self._best_criteria(X,y,feat_idxs)
        left_idxs,right_idxs = self._split(X[:,best_feat],best_thresh)
        left = self._grow_tree(X[left_idxs,:],y[left_idxs],depth+1)
        right = self._grow_tree(X[right_idxs,:],y[right_idxs],depth+1)
        return Node(best_feat,best_thresh,left,right)

    def _best_criteria(self, X, y, feat_idxs):
        # print("y^2:", y)
        best_gain = -1
        split_idx, split_thresh = None, None


        # Ensure that feat_idxs is iterable (it must be an array or list of feature indices)
        if isinstance(feat_idxs, int):
            feat_idxs = np.arange(X.shape[1])  # If it's an integer, consider all feature indices
        
        for feat_idx in feat_idxs:
        
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    
    def _information_gain(self,y,X_column,split_thresh):
        #parent entropy
        parent_entropy = entropy(y)

        #generate split
        left_idxs,right_idxs = self._split(X_column,split_thresh) #split the column
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        #compute the weighted average of the children entropies
        n = len(y)
        n_l,n_r = len(left_idxs),len(right_idxs)
        e_l,e_r = entropy(y[left_idxs]),entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        #return the information gain
        ig = parent_entropy - child_entropy
        return ig

    def _split(self,X_column,split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs,right_idxs

    def predict(self,X):
        
        return np.array([self._traverse_tree(x,self.root) for x in X])
    
    def _traverse_tree(self,x,node=None):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x,node.left)
        
        return self._traverse_tree(x,node.right)


    def _most_common_label(self,y):
        counter = Counter(y) # compter les occurences de chaque label
        most_common = counter.most_common(1)[0][0] # retourner le label le plus fréquent
        return most_common
    

    def fit_gradient_tree(self, X, gradients, hessians):
        """
        Fit a decision tree to the gradients and hessians for gradient boosting.
        """
        # Initialize the number of features to consider for the split
        n_samples, n_features = X.shape
        feat_idxs = np.arange(n_features) if isinstance(self.n_feats, int) else np.random.choice(n_features, self.n_feats, replace=False)
        self.root = self._grow_tree_with_gradient(X, gradients, hessians, feat_idxs)

    def _best_criteria_with_gradient(self, X, gradients, hessians, feat_idxs):
        """
        Find the best split based on gradients and hessians for gradient boosting.
        
        Args:
        X: The feature matrix.
        gradients: The gradients of the loss function with respect to predictions.
        hessians: The hessians (second derivatives) of the loss function.
        feat_idxs: The indices of features to consider for the split.
        
        Returns:
        best_feat: The feature index of the best split.
        best_thresh: The threshold value of the best split.
        """
        best_gain = -float('inf')
        best_feat, best_thresh = None, None
        
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]  # Extract the feature column for splitting
            thresholds = np.unique(X_column)  # Unique values in the feature column to check as potential splits
            
            for threshold in thresholds:
                # Calculate the sum of gradients and hessians for the left and right splits
                left_idxs, right_idxs = self._split(X_column, threshold)  # Split the data
                
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue  # Skip if the split is invalid (empty left or right)

                # Calculate gain for this split
                G_left = np.sum(gradients[left_idxs])  # Sum of gradients for left child
                H_left = np.sum(hessians[left_idxs])   # Sum of hessians for left child
                G_right = np.sum(gradients[right_idxs])  # Sum of gradients for right child
                H_right = np.sum(hessians[right_idxs])  # Sum of hessians for right child
                G_total = np.sum(gradients)  # Total sum of gradients
                H_total = np.sum(hessians)   # Total sum of hessians
                
                # Compute the gain from the split
                gain = 0.5 * ( (G_left**2 / H_left) + (G_right**2 / H_right) - (G_total**2 / H_total) )
                
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = threshold

        return best_feat, best_thresh


    def _grow_tree_with_gradient(self, X, gradients, hessians, depth=0):
        n_samples, n_features = X.shape

        # Stopping criteria
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            leaf_value = self._compute_leaf_value(gradients, hessians)
            return Node(value=leaf_value)

        # Find the best split based on gradients/hessians
        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)
        best_feat, best_thresh = self._best_criteria_with_gradient(X, gradients, hessians, feat_idxs)

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree_with_gradient(X[left_idxs], gradients[left_idxs], hessians[left_idxs], depth + 1)
        right = self._grow_tree_with_gradient(X[right_idxs], gradients[right_idxs], hessians[right_idxs], depth + 1)

        return Node(feature=best_feat, threshold=best_thresh, left=left, right=right)

    def _compute_leaf_value(self, gradients, hessians):
        return -np.sum(gradients) / (np.sum(hessians) + 1e-10)  # Prevent division by zero
import numpy as np
import pandas as pd

class LeafWiseTree:
    def __init__(self, max_depth=3, min_samples_split=2, min_gain_to_split=0.0, n_estimators=100, learning_rate=0.1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain_to_split = min_gain_to_split
        self.n_estimators = n_estimators  # Number of trees to build
        self.learning_rate = learning_rate  # Shrinkage factor for predictions
        self.trees = []  # List to store the trees

    # Log loss function (binary cross-entropy)
    def log_loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # Gradient and Hessian for log loss (binary classification)
    def compute_gradients_hessians(self, y_true, y_pred):
        gradients = y_pred - y_true  # First derivative (gradient)
        hessians = y_pred * (1 - y_pred)  # Second derivative (Hessian)
        return gradients, hessians

    # Calculate the weighted sum of gradients and Hessians for a split
    def calculate_info_gain(self, X, y, feature_index, threshold, gradients, hessians):
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) < self.min_samples_split or np.sum(right_mask) < self.min_samples_split:
            return 0  # Do not split if the resulting splits don't meet the minimum sample requirement

        left_grad = np.sum(gradients[left_mask])
        left_hess = np.sum(hessians[left_mask])
        right_grad = np.sum(gradients[right_mask])
        right_hess = np.sum(hessians[right_mask])

        gain = (left_grad ** 2) / (left_hess + 1e-6) + (right_grad ** 2) / (right_hess + 1e-6)
        return gain

    # Build the tree with gradients and Hessians
    def build_tree(self, X, y, grad, hess):
        self.tree = []
        nodes_to_split = [(X, y, grad, hess, 0)]  # (data, labels, gradients, hessians, current_depth)

        while nodes_to_split:
            node_data, node_labels, node_gradients, node_hessians, depth = nodes_to_split.pop(0)
            
            if depth >= self.max_depth:
                continue
            
            best_gain = 0
            best_split = None
            best_left_mask = None
            best_right_mask = None
            
            # Try splitting on every possible threshold in the feature
            for feature_index in range(X.shape[1]):
                unique_values = np.unique(node_data[:, feature_index])
                
                for threshold in unique_values:
                    gain = self.calculate_info_gain(node_data, node_labels, feature_index, threshold, node_gradients, node_hessians)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_split = (feature_index, threshold)
                        left_mask = node_data[:, feature_index] <= threshold
                        right_mask = ~left_mask
                        best_left_mask = left_mask
                        best_right_mask = right_mask
            
            if best_split:
                feature_index, threshold = best_split
                self.tree.append((feature_index, threshold))
                nodes_to_split.append((node_data[best_left_mask], node_labels[best_left_mask], node_gradients[best_left_mask], node_hessians[best_left_mask], depth + 1))
                nodes_to_split.append((node_data[best_right_mask], node_labels[best_right_mask], node_gradients[best_right_mask], node_hessians[best_right_mask], depth + 1))

    # Predict using the tree
    def predict(self, X):
        predictions = []
        for sample in X:
            node_data = sample
            for feature_index, threshold in self.tree:
                if node_data[feature_index] <= threshold:
                    continue  # Move to the left branch
                else:
                    continue  # Move to the right branch
            
            predictions.append(1 if np.random.random() > 0.5 else 0)  # Dummy prediction, just for structure
        return np.array(predictions)

    # Fit function to train the model (with precomputed gradients and Hessians)
    def fit(self, X,y, grad, hess):
        # Initialize predictions with the mean of y for classification (or average for regression)
        y_pred = np.full_like(grad, 0.5, dtype=float)  # Start with 0.5 for binary classification

        self.trees = []  # Initialize an empty list of trees

        for _ in range(self.n_estimators):
            # Build the tree using the precomputed gradients and Hessians
            self.build_tree(X,y, grad, hess)
            
            # Update predictions using the learned tree
            y_pred -= self.learning_rate * grad  # Adjust predictions by residuals (gradient descent step)
            
            # Store the tree in the list of trees
            self.trees.append(self.tree)
            
            # Optionally print progress
            if _ % 10 == 0:
                loss = self.log_loss(grad, y_pred)  # In practice, you would use the true labels here
                print(f"Iteration {_}, Loss: {loss}")

