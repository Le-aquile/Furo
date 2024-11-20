import numpy as np

from .activators import relu, sigmoid, tanh, leaky_relu, softmax, linear, swift


def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

class NeuralNetwork:
    """
    A multi-layer neural network with backpropagation training.
    
    Parameters:
        layers (list): A list of integers representing the number of neurons in each layer.
        activators (list): A list of activation functions (callables) for each layer.
        learning_rate (float, optional): The learning rate of the network. Defaults to 0.01.
    """
    def __init__(self, layers, activators, learning_rate=0.01):
        self.layers = layers
        self.activators = activators
        self.learning_rate = learning_rate
        self.weights = [np.random.randn(next_layer, input_size) for input_size, next_layer in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(size, 1) for size in layers[1:]]

    def forward_pass(self, input_matrix):
        """
        Perform a forward pass through the network on input data.
        
        Parameters:
            input_matrix (numpy.array): Input data of shape (input_size, batch_size)
        
        Returns:
            activations (list): A list of output activations for each layer.
        """
        activations = [input_matrix]
        layer_input = input_matrix

        for w, b, activation in zip(self.weights, self.biases, self.activators):
            z = np.dot(w, layer_input) + b
            layer_input = activation(z)
            activations.append(layer_input)

        return activations

    def backward_pass(self, activations, y_true):
        """
        Perform a backward pass through the network to compute gradients.
        
        Parameters:
            activations (list): Output activations from the forward pass.
            y_true (numpy.array): True labels of shape (output_size, batch_size)
        """
        loss_grad = mse_loss_derivative(y_true, activations[-1])
        delta = loss_grad

        for i in reversed(range(len(self.weights))):
            layer_output = activations[i + 1]
            if self.activators[i] == relu:
                activation_grad = relu(layer_output)
            elif self.activators[i] == sigmoid:
                activation_grad = sigmoid(layer_output)
            elif self.activators[i] == tanh:
                activation_grad = tanh(layer_output)
            elif self.activators[i] == softmax:
                activation_grad = softmax(layer_output)
            elif self.activators[i] == linear:
                activation_grad = 1
            elif self.activators[i] == swift:
                activation_grad = swift(layer_output)
            elif self.activators[i] == leaky_relu:
                alpha = 0.01
                activation_grad = np.where(layer_output > 0, 1, alpha)

            delta *= activation_grad
            dW = np.dot(delta, activations[i].T) / y_true.shape[1]
            dB = np.mean(delta, axis=1, keepdims=True)

            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * dB

            delta = np.dot(self.weights[i].T, delta)

    def train(self, X_train, y_train, epochs=100, batch_size=32):
        """
        Train the network on input data.
        
        Parameters:
            X_train (numpy.array): Input data of shape (input_size, batch_size)
            y_train (numpy.array): True labels of shape (output_size, batch_size)
            epochs (int, optional): Number of epochs to train the network. Defaults to 100.
            batch_size (int, optional): Number of samples per mini-batch. Defaults to 32.
        """
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
        """
        Generate predictions for input data X.
        
        Parameters:
            X (numpy.array): Input data of shape (input_size, batch_size)
        
        Returns:
            predictions (numpy.array): Predictions of shape (output_size, batch_size)
        """
        output_activations = self.forward_pass(X)[-1]

        if self.activators[-1] == softmax:
            # Classification: Return index of the highest probability
            return np.argmax(output_activations, axis=0)  # Use argmax to return the index of the highest probability
        else:
            # Regression: Return raw output
            return output_activations



if __name__ == "__main__":
    print("Example usage:")
    input_data = np.random.rand(3, 100)  
    target_output = np.random.rand(3, 100)


    layers_configuration = [3, 6, 6, 6,3]
    activators = [swift, tanh, sigmoid, linear,softmax]
    learning_rate = 0.01

    network = NeuralNetwork(layers_configuration, activators, learning_rate)

    # Train the network with 100 epochs and mini-batch size of 16
    network.train(input_data, target_output, epochs=100, batch_size=16)

    # Test predictions after training
    test_data = np.random.rand(3, 5)
    predictions = network.predict(test_data)
    print("Predictions on new data classes:", predictions)

