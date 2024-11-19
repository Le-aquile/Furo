import numpy as np

def linear(x) -> np.ndarray:
    return x

def relu(x) -> np.ndarray:
    
    """
    Applies the Rectified Linear Unit (ReLU) activation function to the input.

    Parameters
    ----------
    x : array-like
        The input values.

    Returns
    -------
    np.ndarray
        The ReLU of the input values, i.e. the maximum of 0 and the input values.
    """
    return np.maximum(0, x)

def sigmoid(x) -> np.ndarray:
    """
    Applies the sigmoid activation function to the input.

    Parameters
    ----------
    x : array-like
        The input values.

    Returns
    -------
    np.ndarray
        The sigmoid of the input values, i.e. 1 / (1 + exp(-x)).
    """
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def swift(x) -> np.ndarray:
    """
    Applies the Swift activation function to the input.

    Parameters
    ----------
    x : array-like
        The input values.

    Returns
    -------
    np.ndarray
        The Swift of the input values, i.e. x * 1 / (1 + exp(-x)).
    """
    x = np.clip(x, -500, 500)
    return x * sigmoid(x)
def softmax(x) -> np.ndarray:
    """
    Applies the softmax activation function to the input.

    Parameters
    ----------
    x : array-like
        The input values.

    Returns
    -------
    np.ndarray
        The softmax of the input values, which is a probability distribution over the input.
    """
    x_max = np.max(x, axis=0, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def tanh(x) -> np.ndarray:
    """
    Applies the hyperbolic tangent activation function to the input.

    Parameters
    ----------
    x : array-like
        The input values.

    Returns
    -------
    np.ndarray
        The hyperbolic tangent of the input values, i.e. tanh(x).
    """
    return np.tanh(x)

def leaky_relu(x, alpha=0.01) -> np.ndarray:
    
    """
    Applies the leaky ReLU activation function to the input.

    Parameters
    ----------
    x : array-like
        The input values.
    alpha : float, optional
        The slope of the negative part of the leaky ReLU, by default 0.01.

    Returns
    -------
    np.ndarray
        The leaky ReLU of the input values, i.e. f(x) = max(alpha * x, x).
    """
    return np.maximum(alpha * x, x)

