import numpy as np

from .activators import relu, sigmoid, tanh, leaky_relu, softmax, linear, swift
from .optimizers import adam_optimizer, rmsprop, sgd

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

class NeuralNetwork:
    """
    A multi-layer neural network with backpropagation training.
    Supports custom activation functions and optimizers.
    """
    def __init__(self, layers, activators, learning_rate=0.01, optimizer="sgd"):
        self.layers = layers
        self.activators = activators
        self.learning_rate = learning_rate
        self.weights = [np.random.randn(next_layer, input_size) for input_size, next_layer in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(size, 1) for size in layers[1:]]
        
        self.optimizer = optimizer
        self.velocity_w = [np.zeros_like(w) for w in self.weights]
        self.velocity_b = [np.zeros_like(b) for b in self.biases]

    def forward_pass(self, input_matrix):
        activations = [input_matrix]
        layer_input = input_matrix

        for w, b, activation in zip(self.weights, self.biases, self.activators):
            z = np.dot(w, layer_input) + b
            layer_input = activation(z)
            activations.append(layer_input)

        return activations

    def backward_pass(self, activations, y_true):
        loss_grad = mse_loss_derivative(y_true, activations[-1])
        delta = loss_grad

        grad_weights = []
        grad_biases = []

        for i in reversed(range(len(self.weights))):
            layer_output = activations[i + 1]
            activation_grad = self._activation_derivative(layer_output, self.activators[i])
            delta *= activation_grad

            dW = np.dot(delta, activations[i].T) / y_true.shape[1]
            dB = np.mean(delta, axis=1, keepdims=True)

            grad_weights.insert(0, dW)
            grad_biases.insert(0, dB)

            delta = np.dot(self.weights[i].T, delta)

        self._update_weights_biases(grad_weights, grad_biases)

    def _update_weights_biases(self, grad_weights, grad_biases):
        if self.optimizer == "sgd":
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * grad_weights[i]
                self.biases[i] -= self.learning_rate * grad_biases[i]

        elif self.optimizer == "adam":
            for i in range(len(self.weights)):
                self.weights[i] = adam_optimizer(
                    lambda _: grad_weights[i],
                    self.weights[i],
                    learning_rate=self.learning_rate,
                    steps=1
                )
                self.biases[i] = adam_optimizer(
                    lambda _: grad_biases[i],
                    self.biases[i],
                    learning_rate=self.learning_rate,
                    steps=1
                )

        elif self.optimizer == "rmsprop":
            for i in range(len(self.weights)):
                self.weights[i] = rmsprop(
                    self.weights[i],
                    lambda _: grad_weights[i],
                    learning_rate=self.learning_rate,
                    steps=1
                )
                self.biases[i] = rmsprop(
                    self.biases[i],
                    lambda _: grad_biases[i],
                    learning_rate=self.learning_rate,
                    steps=1
                )

    def _activation_derivative(self, layer_output, activation):
        if activation == relu:
            return np.where(layer_output > 0, 1, 0)
        elif activation == sigmoid:
            sigmoid_output = sigmoid(layer_output)
            return sigmoid_output * (1 - sigmoid_output)
        elif activation == tanh:
            return 1 - np.tanh(layer_output) ** 2
        elif activation == softmax:
            return softmax(layer_output) * (1 - softmax(layer_output))
        elif activation == linear:
            return 1
        elif activation == swift:
            sigmoid_x = sigmoid(layer_output)
            return sigmoid_x + layer_output * sigmoid_x * (1 - sigmoid_x)
        elif activation == leaky_relu:
            alpha = 0.01
            return np.where(layer_output > 0, 1, alpha)

    def train(self, X_train, y_train, epochs=100, batch_size=32):
        for epoch in range(epochs):
            permutation = np.random.permutation(X_train.shape[1])
            X_shuffled = X_train[:, permutation]
            y_shuffled = y_train[:, permutation]

            for start in range(0, X_train.shape[1], batch_size):
                end = start + batch_size
                X_batch = X_shuffled[:, start:end]
                y_batch = y_shuffled[:, start:end]

                activations = self.forward_pass(X_batch)
                self.backward_pass(activations, y_batch)

            if epoch % 10 == 0 or epoch == epochs - 1:
                predictions = self.forward_pass(X_train)[-1]
                loss = mse_loss(y_train, predictions)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

    def predict(self, X):
        output_activations = self.forward_pass(X)[-1]
        if self.activators[-1] == softmax:
            return np.argmax(output_activations, axis=0)
        else:
            return output_activations
