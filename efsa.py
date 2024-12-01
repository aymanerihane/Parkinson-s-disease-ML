import numpy as np

class EFSA:
    def __init__(self, correlation_threshold=0.3, lasso_penalty=0.01, n_features=None):
        """
        Parameters:
        - correlation_threshold: Threshold for feature correlation with the target variable.
        - lasso_penalty: Regularization parameter for Lasso-like feature selection.
        - n_features: Number of features to select in the ensemble step.
        """
        self.correlation_threshold = correlation_threshold
        self.lasso_penalty = lasso_penalty
        self.n_features = n_features
        self.selected_features_ = None

    def correlation2(self, X, y):
        mean_x = np.mean(X)
        mean_y = np.mean(y)
        n = len(X)

        # cov
        numerator = np.sum((X - mean_x) * (y - mean_y))

        # std
        denominator = np.sqrt(np.sum((X - mean_x) ** 2) * np.sum((y - mean_y) ** 2))

        return numerator / denominator
    
    def correlation_selection(self, X, y):
        """
        Select features based on correlation with the target.
        Parameters:
        - X: Feature matrix.
        - y: Target vector.
        
        Returns:
        - selected: List of selected feature indices.
        - correlations: List of tuples (feature_index, correlation_score).
        """
        correlations = []
        scores = []
        for i in range(X.shape[1]):
            feature = X[:, i]
            corr = self.correlation2(feature, y)
            if abs(corr) > self.correlation_threshold:
                correlations.append((i, abs(corr)))
                scores.append(i)
            

        return correlations, scores
        

    # def correlation_selection(self, X, y):
    #     """
    #     Select features based on correlation with the target.
    #     Parameters:
    #     - X: Feature matrix.
    #     - y: Target vector.
        
    #     Returns:
    #     - selected: List of selected feature indices.
    #     - correlations: List of tuples (feature_index, correlation_score).
    #     """
    #     correlations = []
    #     for i in range(X.shape[1]):
    #         feature = X[:, i]
    #         corr = np.corrcoef(feature, y)[0, 1]  # Pearson correlation
    #         correlations.append((i, abs(corr)))

    #     # Select features above the correlation threshold
    #     selected = [i for i, corr in correlations if corr >= self.correlation_threshold]
    #     return selected, correlations

    def lasso_selection(self, X, y):
        """
        Simplified Lasso-like feature selection using gradient descent.
        Parameters:
        - X: Feature matrix.
        - y: Target vector.
        
        Returns:
        - selected: List of selected feature indices.
        - weights: Feature weights from Lasso.
        """
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)  # Initialize weights to zero
        lr = 0.01  # Learning rate
        n_iterations = 1000

        for _ in range(n_iterations):
            y_pred = np.dot(X, weights)
            residual = y - y_pred

            # Compute gradients for weights
            gradients = -2 * np.dot(X.T, residual) / n_samples + self.lasso_penalty * np.sign(weights)

            # Update weights
            weights -= lr * gradients

        # Select features with non-zero weights
        selected = [i for i in range(n_features) if abs(weights[i]) > 1e-4]
        return selected, weights

    def mutual_info_selection(self, X, y):
        """
        Approximation of mutual information using entropy-based computation.
        Parameters:
        - X: Feature matrix.
        - y: Target vector.
        
        Returns:
        - selected: List of sorted feature indices based on mutual information.
        - mutual_infos: List of tuples (feature_index, mutual_information_score).
        """
        def entropy(arr):
            # Offset negative values by adding the absolute value of the minimum element + 1
            offset = abs(np.min(arr)) + 1 if np.min(arr) < 0 else 0
            arr = arr + offset  # Make sure all values are non-negative

            # Calculate probabilities and entropy
            probs = np.bincount(arr) / len(arr)
            return -np.sum(p * np.log2(p) for p in probs if p > 0)

        mutual_infos = []
        for i in range(X.shape[1]):
            feature = X[:, i].astype(int)
            joint_entropy = entropy(np.vstack((feature, y)).T.flatten())
            mi = entropy(feature) + entropy(y) - joint_entropy
            mutual_infos.append((i, mi))

        # Sort by mutual information
        sorted_features = sorted(mutual_infos, key=lambda x: x[1], reverse=True)
        return [i for i, _ in sorted_features], mutual_infos


    def fit(self, X, y):
        """
        Runs the ensemble feature selection process.
        Parameters:
        - X: Feature matrix.
        - y: Target vector.
        """
        # Validate inputs
        if X.shape[0] != len(y):
            raise ValueError("Number of samples in X and y must match.")
        
        # Correlation-based selection
        corr_selected, corr_scores = self.correlation_selection(X, y)
        print(f"Correlation-selected features: {corr_selected}")

        # Lasso-based selection
        lasso_selected, lasso_weights = self.lasso_selection(X, y)
        print(f"Lasso-selected features: {lasso_selected}")

        # Mutual information-based selection
        mi_selected, mi_scores = self.mutual_info_selection(X, y)
        print(f"Mutual information-selected features: {mi_selected[:5]} (top 5)")

        # Combine selected features and score them
        combined_scores = {}
        
        # Add correlation scores
        for feature, corr in corr_scores:
            combined_scores[feature] = corr
        
        # Add Lasso scores (using weights)
        for i, weight in enumerate(lasso_weights):
            if abs(weight) > 1e-4:  # Non-zero weight
                if i not in combined_scores:
                    combined_scores[i] = 0
                combined_scores[i] += abs(weight)
        
        # Add Mutual Information scores
        for feature, mi in mi_scores:
            if feature not in combined_scores:
                combined_scores[feature] = 0
            combined_scores[feature] += mi

        # Sort features by combined score and select top n_features
        sorted_features = sorted(combined_scores, key=combined_scores.get, reverse=True)
        self.selected_features_ = sorted_features[:self.n_features] if self.n_features else sorted_features

    def transform(self, X):
        """
        Transforms the dataset to include only the selected features.
        Parameters:
        - X: Feature matrix.
        
        Returns:
        - Transformed feature matrix.
        """
        if self.selected_features_ is None:
            raise ValueError("The EFSA model has not been fitted yet.")
        return X[:, self.selected_features_]

    def fit_transform(self, X, y):
        """
        Fits the model and transforms the dataset in a single step.
        Parameters:
        - X: Feature matrix.
        - y: Target vector.
        
        Returns:
        - Transformed feature matrix.
        """
        self.fit(X, y)
        return self.transform(X)
