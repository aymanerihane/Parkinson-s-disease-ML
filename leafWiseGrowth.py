import numpy as np
import heapq
import matplotlib.pyplot as plt

class Node:
    def __init__(self, feature=None, value=None, gain=None, left=None, right=None):
        self.feature = feature
        self.value = value
        self.gain = gain
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.left is None and self.right is None

    def __repr__(self):
        return f"Node(feature={self.feature}, value={self.value}, gain={self.gain})"

class LeafWiseTree:
    def __init__(self, min_samples_split=2, min_gain_to_split=0.1, l1_reg=0.1, l2_reg=0.1, max_leaves=10):
        self.min_samples_split = min_samples_split
        self.min_gain_to_split = min_gain_to_split
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.max_leaves = max_leaves
        self.tree = None
        self.leaf_count = 0  # Track current number of leaves

    def _calculate_gain(self, left_grad, left_hess, right_grad, right_hess):
        """Calculate the information gain from a split, considering L1 and L2 regularization."""
        def gain(grad, hess):
            return (grad**2) / (hess + self.l2_reg)
        
        gain_left = gain(np.sum(left_grad), np.sum(left_hess))
        gain_right = gain(np.sum(right_grad), np.sum(right_hess))
        total_gain = gain_left + gain_right
        
        regularization_penalty = self.l1_reg * (abs(np.sum(left_grad)) + abs(np.sum(right_grad)))
        
        return total_gain - regularization_penalty

    def _find_best_split(self, X, gradients, hessians):
        """Find the best split for the current node."""
        best_gain = -np.inf
        best_split = None
        n_features = X.shape[1]

        for feature in range(n_features):
            unique_values = np.unique(X[:, feature])

            for value in unique_values:
                left_mask = X[:, feature] <= value
                right_mask = ~left_mask

                if np.sum(left_mask) < self.min_samples_split or np.sum(right_mask) < self.min_samples_split:
                    continue

                left_grad, right_grad = gradients[left_mask], gradients[right_mask]
                left_hess, right_hess = hessians[left_mask], hessians[right_mask]

                gain = self._calculate_gain(left_grad, left_hess, right_grad, right_hess)
                if gain > best_gain and gain > self.min_gain_to_split:
                    best_gain = gain
                    best_split = {
                        'feature': feature,
                        'value': value,
                        'gain': gain,
                        'left_mask': left_mask,
                        'right_mask': right_mask
                    }
        return best_split

    def _build_tree(self, X, gradients, hessians):
        """Build the tree using a priority queue for leaf-wise growth."""
        priority_queue = []  # Max-heap for potential splits

        # Find initial split (root node)
        initial_split = self._find_best_split(X, gradients, hessians)
        if initial_split is None:
            return None  # No valid split found

        # Create the root node and add it to the tree
        root = Node(
            feature=initial_split['feature'],
            value=initial_split['value'],
            gain=initial_split['gain']
        )

        # Push initial split into the queue
        heapq.heappush(priority_queue, (-initial_split['gain'], 0, root, X, gradients, hessians))
        self.leaf_count = 1

        while priority_queue and self.leaf_count < self.max_leaves:
            _, depth, current_node, X_node, grad_node, hess_node = heapq.heappop(priority_queue)

            # Perform the split for the current node
            left_mask = X_node[:, current_node.feature] <= current_node.value
            right_mask = ~left_mask

            left_X, right_X = X_node[left_mask], X_node[right_mask]
            left_grad, right_grad = grad_node[left_mask], grad_node[right_mask]
            left_hess, right_hess = hess_node[left_mask], hess_node[right_mask]

            # Find best splits for left and right child nodes
            left_split = self._find_best_split(left_X, left_grad, left_hess)
            right_split = self._find_best_split(right_X, right_grad, right_hess)

            # Create child nodes and attach to current node
            if left_split:
                left_child = Node(
                    feature=left_split['feature'],
                    value=left_split['value'],
                    gain=left_split['gain']
                )
                current_node.left = left_child  # Attach left child
                heapq.heappush(priority_queue, (-left_split['gain'], depth + 1, left_child, left_X, left_grad, left_hess))
                self.leaf_count += 1

            if right_split:
                right_child = Node(
                    feature=right_split['feature'],
                    value=right_split['value'],
                    gain=right_split['gain']
                )
                current_node.right = right_child  # Attach right child
                heapq.heappush(priority_queue, (-right_split['gain'], depth + 1, right_child, right_X, right_grad, right_hess))
                self.leaf_count += 1

        return root  # Return the root of the constructed tree

    def fit(self, X, gradients, hessians):
        """Fit the tree using the given data, gradients, and hessians."""
        self.tree = self._build_tree(X, gradients, hessians)
        print(f"Constructed tree with {self.tree} leaves.")
        return self.tree
    
    def predict(self, X):
        """Predict using the built tree."""
        def traverse_tree(x, node):
            if node is None:
                # Return a default value if node is None (unlikely but failsafe)
                return 0

            if node.is_leaf():
                # Leaf node reached
                return node.value

            # Traverse the tree based on the feature split
            if x[node.feature] <= node.value:
                return traverse_tree(x, node.left)
            else:
                return traverse_tree(x, node.right)

        return np.array([traverse_tree(row, self.tree) for row in X])

    def plot_tree(self, node=None, pos=None, level=0, vert_gap=0.4, ax=None, parent_pos=None, max_depth=10):
        """Plot the tree structure using Matplotlib."""
        if node is None:
            node = self.tree

        if pos is None:
            pos = (0.5, 1.0)  # Start plotting from the top center

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('off')

        if parent_pos is not None:
            ax.plot([parent_pos[0], pos[0]], [parent_pos[1], pos[1]], color='k', lw=2)

        if node.is_leaf():
            ax.text(pos[0], pos[1], f"Leaf\nValue: {node.value:.2f}", ha='center', va='center', fontsize=10,
                    bbox=dict(facecolor='lightgreen', edgecolor='black', boxstyle='round,pad=0.3'))
        else:
            ax.text(pos[0], pos[1], f"X{node.feature} <= {node.value:.2f}\nGain: {node.gain:.2f}", ha='center',
                    va='center', fontsize=10, bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=0.3'))

            # Recursively plot left and right children, but limit depth
            if level < max_depth:
                dx = 1.0 / (2 ** (min(level + 1, 10)))  # Limit `dx` to prevent overflow

                # Only recurse if the tree has children at the current level
                if node.left:
                    self.plot_tree(node.left, pos=(pos[0] - dx, pos[1] - vert_gap), level=level + 1, ax=ax, parent_pos=pos, max_depth=max_depth)
                if node.right:
                    self.plot_tree(node.right, pos=(pos[0] + dx, pos[1] - vert_gap), level=level + 1, ax=ax, parent_pos=pos, max_depth=max_depth)

        return ax

