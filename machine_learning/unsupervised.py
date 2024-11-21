import numpy as np
from typing import NoReturn


class Perceptron:
    """
    A simple single-layer Perceptron neural network.
    """

    def __init__(self, learning_rate=0.01, epochs=1000) -> NoReturn:
        """
        Initializes a Perceptron model with the given learning rate and number of epochs.

        Parameters:
            learning_rate (float): The learning rate of the model.
            epochs (int): The number of training epochs.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        """
        Trains the model on the given data.

        Parameters:
            X (numpy.array): The feature matrix of the training data.
            y (numpy.array): The target vector of the training data.
        """
        # Initialize weights and bias
        num_features = X.shape[1]
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Training loop
        for _ in range(self.epochs):
            for i, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = np.where(linear_output >= 0, 1, -1)

                # Update weights if the prediction is incorrect
                if y_pred != y[i]:
                    update = self.learning_rate * y[i]
                    self.weights += update * x_i
                    self.bias += update
    
    def predict(self, X):
        """
        Predicts the labels for the given feature matrix.

        Parameters:
            X (numpy.array): The feature matrix of the data to be predicted.

        Returns:
            numpy.array: The predicted labels.
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, -1)



if __name__ == "__main__":
    print("Example usage:")
    # Sample binary classification data
    X1 = np.array([[2, 3], [1, 1], [2, 1], [3, 1]])
    y1 = np.array([1, -1, -1, 1])

    # Initialize and train the Perceptron
    perceptron = Perceptron(learning_rate=0.1, epochs=10)
    perceptron.fit(X1, y1)

    # Make predictions
    predictions = perceptron.predict(X1)
    print(predictions)  # Expected output: array of -1 and 1 for binary classes

