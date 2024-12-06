# pca_utils.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        # Mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # Covariance matrix (function needs samples as columns)
        cov = np.cov(X.T)

        # Eigenvalues, eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Sort eigenvectors by eigenvalues in descending order
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # Explained variance ratio
        self.explained_variance_ratio_ = eigenvalues / np.sum(eigenvalues)

        # Store the first n eigenvectors (components)
        if self.n_components is not None:
            self.components = eigenvectors[0: self.n_components]
        else:
            self.components = eigenvectors

    def transform(self, X):
        # Project data onto the principal components
        X = X - self.mean
        return np.dot(X, self.components.T)


def load_data(file_path):
    """Load the data from a CSV file."""
    data = pd.read_csv(file_path)
    # Assuming the target variable is the last column and the rest are features
    X = data.iloc[:, :-1].values  # Features (all columns except last)
    y = data.iloc[:, -1].values   # Target (last column)
    return X, y


def plot_cumulative_variance(explained_variance_ratio_):
    """Plot the cumulative explained variance against the number of components."""
    cumulative_variance = np.cumsum(explained_variance_ratio_)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Explained Variance by Number of Principal Components")
    plt.grid()
    plt.show()


def find_optimal_components(cumulative_variance, threshold=0.9998):
    """Find the optimal number of components to explain a certain variance threshold."""
    optimal_components = np.argmax(cumulative_variance >= threshold) + 1  # Adding 1 for index correction
    return optimal_components
