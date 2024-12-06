import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Define the TreeNode class
class TreeNode:
    def __init__(self, value=None):
        self.value = value  # The feature index for the split
        self.threshold = None  # The threshold value for the split
        self.left = None      # Left child
        self.right = None     # Right child
        self.is_leaf = True   # Indicates if the node is a leaf
        self.prediction = None  # Prediction value for leaf nodes

# Define the LeafWiseTree class (LightGBM Tree)
class LeafWiseTree:
    def __init__(self):
        self.root = TreeNode()  # Initialize the root of the tree

    def grow_tree(self, X, gradients, hessians, max_depth=3, n_jobs=4):
        self._grow_node(self.root, X, gradients, hessians, depth=0, max_depth=max_depth, n_jobs=n_jobs)

    def _grow_node(self, node, X, gradients, hessians, depth, max_depth, n_jobs):
        if depth < max_depth and len(np.unique(gradients)) > 1:  # Check for stopping conditions
            best_feature, best_threshold = self._find_best_split(X, gradients, hessians, n_jobs)
            if best_feature is not None:
                node.value = best_feature
                node.threshold = best_threshold
                node.is_leaf = False
                # Split the data
                left_indices = X[:, best_feature] < best_threshold
                right_indices = X[:, best_feature] >= best_threshold
                node.left = TreeNode()
                node.right = TreeNode()
                # Parallelize the growing of child nodes
                with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                    futures = [
                        executor.submit(self._grow_node, node.left, X[left_indices], gradients[left_indices], hessians[left_indices], depth + 1, max_depth, n_jobs),
                        executor.submit(self._grow_node, node.right, X[right_indices], gradients[right_indices], hessians[right_indices], depth + 1, max_depth, n_jobs)
                    ]
                    for future in futures:
                        future.result()  # Wait for the child nodes to finish
        else:
            # Set the leaf prediction based on gradients and hessians
            mean_grad = np.sum(gradients)
            mean_hess = np.sum(hessians)
            node.prediction = mean_grad / (mean_hess + 1e-10)  # Add small value to avoid division by zero

    def _find_best_split(self, X, gradients, hessians, n_jobs):
        best_feature = None
        best_threshold = None
        best_gain = -float('inf')  # For regression, we maximize the gain
        
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = []
            for feature in range(X.shape[1]):
                futures.append(executor.submit(self._find_split_for_feature, feature, X, gradients, hessians))
            
            # Wait for all futures to complete and get the results
            results = [future.result() for future in futures]
            
            # Find the best split result from the computed results
            for res in results:
                if res['gain'] > best_gain:
                    best_gain = res['gain']
                    best_feature = res['feature']
                    best_threshold = res['threshold']

        return best_feature, best_threshold

    def _find_split_for_feature(self, feature, X, gradients, hessians):
        thresholds = np.unique(X[:, feature])
        best_gain = -float('inf')
        best_threshold = None
        
        for threshold in thresholds:
            left_indices = X[:, feature] < threshold
            right_indices = X[:, feature] >= threshold
            if len(gradients[left_indices]) == 0 or len(gradients[right_indices]) == 0:
                continue

            # Calculate the gain
            left_grad = gradients[left_indices]
            right_grad = gradients[right_indices]
            left_hess = hessians[left_indices]
            right_hess = hessians[right_indices]

            gain = self._calculate_gain(left_grad, left_hess, right_grad, right_hess)

            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold

        return {'feature': feature, 'threshold': best_threshold, 'gain': best_gain}

    def _calculate_gain(self, left_grad, left_hess, right_grad, right_hess):
        # Calculate the gain based on gradients and Hessians
        left_sum_grad = np.sum(left_grad)
        right_sum_grad = np.sum(right_grad)
        left_sum_hess = np.sum(left_hess)
        right_sum_hess = np.sum(right_hess)

        # Gain calculation based on the second derivative (Hessian)
        gain = 0.5 * (left_sum_grad ** 2 / (left_sum_hess + 1e-10) + 
                      right_sum_grad ** 2 / (right_sum_hess + 1e-10) - 
                      (left_sum_grad + right_sum_grad) ** 2 / (left_sum_hess + right_sum_hess + 1e-10))
        
        return gain
    def score(self,y,y_pred):
        return np.mean(y == y_pred)