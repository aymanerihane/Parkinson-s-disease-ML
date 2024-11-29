import numpy as np
# from decision_tree import LeafWiseTree
from leafWiseGrowth import LeafWiseTree
import matplotlib.pyplot as plt

# Utility: Calculate Binary Cross-Entropy Loss Gradient and Hessian
def binary_cross_entropy_grad_hess(preds, targets):
    preds = np.clip(preds, 1e-6, 1 - 1e-6)  # Avoid division by zero
    grad = preds - targets  # Gradient
    hess = preds * (1 - preds)  # Hessian
    return grad, hess

# Utility: Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# LightGBM model structure
class LightGBM:
    def __init__(self, n_estimators=10, learning_rate=0.1,min_samples_split=10):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        preds = np.full(n_samples, 0.5)  # Initial prediction (log odds = 0)

        for i in range(self.n_estimators):
            print(f"Fitting tree {i+1}...")
            grad, hess = binary_cross_entropy_grad_hess(preds, y)
            tree = LeafWiseTree(
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X,grad, hess)
            self.trees.append(tree)

            # Update predictions
            preds += self.learning_rate * tree.predict(X)
        
        # self.plot_tree(tree)

    def predict_proba(self, X):
        preds = np.full(X.shape[0], 0.5)
        for tree in self.trees:
            preds += self.learning_rate * tree.predict(X)
        return sigmoid(preds)

    def predict(self, X):
        probas = self.predict_proba(X)
        return (probas >= 0.5).astype(int)
    
    def evaluate(self, X, y):
        preds = self.predict(X)
        accuracy = np.mean(preds == y)
        return accuracy
    
    def visualize(self):
        self.trees[-1].plot_tree()  # Plot the tree
        plt.show()  # Display the plot
    
    def plot_tree(self, node, pos=(0.5, 1), level_width=0.4, vert_gap=0.1, ax=None, parent_pos=None):
        """Recursively plot the decision tree."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('off')

        # Check if the node is a leaf (has a 'value' attribute)
        if node.value is not None:  # Node is a leaf
            ax.text(pos[0], pos[1], f"Value: {node.value:.3f}", 
                    ha='center', va='center', bbox=dict(facecolor='lightgreen', edgecolor='black'))
        else:  # Internal node with a feature split
            ax.text(pos[0], pos[1], f"Feature {node.feature}\n<= {node.value:.2f}", 
                    ha='center', va='center', bbox=dict(facecolor='orange', edgecolor='black'))

        # Draw line to parent node
        if parent_pos is not None:
            ax.plot([parent_pos[0], pos[0]], [parent_pos[1], pos[1]], 'k-')

        # Recursive plot for left and right children
        if node.left is not None:
            self.plot_tree(node.left, (pos[0] - level_width, pos[1] - vert_gap), 
                        level_width * 0.5, vert_gap, ax, pos)
        if node.right is not None:
            self.plot_tree(node.right, (pos[0] + level_width, pos[1] - vert_gap), 
                        level_width * 0.5, vert_gap, ax, pos)
