import numpy as np
from sklearn.model_selection import train_test_split

class Logistic:

    def __init__(self,learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights_list = []
        self.losses = []

    # Step 1: Sigmoid Function
    def _sigmoid(self, z):
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


    # Step 2: Prediction Function
    def _predict(self, X, weights):
        z = np.dot(X, weights)
        return self._sigmoid(z)

    # Step 3: Loss Function (Binary Cross-Entropy)
    def _compute_loss(self, y, y_pred):
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    # Step 4: Gradient Descent
    def _gradient_descent(self, X, y, weights):
        N = len(y)
        y_pred = self._predict(X, weights)
        gradient = np.dot(X.T, (y_pred - y)) / N
        weights -= self.learning_rate * gradient
        return weights

    # Step 5: Training Function
    def train_logistic_regression(self, X, y_binary):
        # Initialize weights
        weights = np.zeros(X.shape[1])
        losses = []
        # Gradient Descent
        for epoch in range(self.epochs):
            y_pred = self._predict(X, weights)
            loss = self._compute_loss(y_binary, y_pred)
            weights = self._gradient_descent(X, y_binary, weights)

            # Print the loss every 100 epochs for tracking
            if epoch % 100 == 0:
                losses.append(loss)
                print(f"Epoch {epoch}: Loss = {loss}")

        self.weights_list.append(weights)
        self.losses.append(losses)

        return self.weights_list, self.losses
    
    # One-vs-All Logistic Regression
    def one_vs_all(self,X, y):
        # One-vs-Rest Strategy for multiclass classification
        num_classes = len(np.unique(self.y))
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # Train a logistic regression model for each class
        for i in range(num_classes):
            # Create binary target variable: 1 if current class, 0 otherwise
            y_binary = np.where(y_train == i, 1, 0)
            weights, losses = self.train_logistic_regression(X_train, y_binary)
            self.weights_list.append(weights)
            self.losses.append(losses)
        
        return self.weights_list, self.losses

    def split_data(self, X, y):
        # Split data into training and testing sets
        return train_test_split(X, y, test_size=0.2, random_state=42)

    # Function to calculate error rate
    @staticmethod
    def error_rate(y_true, y_pred):
        incorrect = np.sum(y_true != y_pred)  # Count incorrect predictions
        total = len(y_true)  # Total predictions
        return incorrect / total  # Calculate error rate
    
    # Function to predict class for new samples
    def predict_multiclass(self, X):
        if not self.weights_list:
            raise ValueError("Model has not been trained yet. Call 'one_vs_all' first.")
        
        probabilities = np.array([self._predict(X, weights) for weights in self.weights_list]).T
        return np.argmax(probabilities, axis=1)
