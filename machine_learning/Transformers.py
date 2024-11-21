import numpy as np
from numpy.linalg import norm
from typing import NoReturn
try:
    from .activators import softmax
except ImportError:
    def softmax(x: np.ndarray) -> np.ndarray:
        """Compute the softmax function along the last axis."""
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class MultiHeadAttention:
    def __init__(self, length=4, dim_model=16, num_heads=2, tomask=False) -> NoReturn:
        assert dim_model % num_heads == 0, "dim_model must be divisible by num_heads"
        self.length = length
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_k = dim_model // num_heads
        self.dim_v = dim_model // num_heads
        self.tomask = tomask

        # Weight matrices initialized using Glorot uniform
        limit = np.sqrt(6 / (dim_model + dim_model))
        self.w_q = np.random.uniform(-limit, limit, (dim_model, dim_model))
        self.w_k = np.random.uniform(-limit, limit, (dim_model, dim_model))
        self.w_v = np.random.uniform(-limit, limit, (dim_model, dim_model))
        self.w_out = np.random.uniform(-limit, limit, (dim_model, dim_model))

    def _split_heads(self, x):
        """Split input for multiple heads."""
        return x.reshape(self.length, self.num_heads, self.dim_k).transpose(1, 0, 2)

    def _combine_heads(self, x):
        """Combine multiple heads into original dimensions."""
        return x.transpose(1, 0, 2).reshape(self.length, self.dim_model)

    def activate(self, q, k, v):
        """Perform multi-head attention."""
        # Linear projections
        q = np.dot(q, self.w_q)
        k = np.dot(k, self.w_k)
        v = np.dot(v, self.w_v)

        # Split into heads
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Scaled dot-product attention
        scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(self.dim_k)
        if self.tomask:
            mask = np.tril(np.ones((self.length, self.length)))
            mask[mask == 0] = -np.inf
            scores += mask
        weights = softmax(scores)

        # Combine outputs
        head_outputs = np.matmul(weights, v)
        combined = self._combine_heads(head_outputs)
        output = np.dot(combined, self.w_out)
        return output


class FeedForwardNetwork:
    def __init__(self, dim_model, dim_ff, activation="relu") -> NoReturn:
        """Initialize feed-forward network."""
        limit1 = np.sqrt(6 / (dim_model + dim_ff))
        limit2 = np.sqrt(6 / (dim_ff + dim_model))
        self.w1 = np.random.uniform(-limit1, limit1, (dim_model, dim_ff))
        self.w2 = np.random.uniform(-limit2, limit2, (dim_ff, dim_model))

        # Set activation function
        if activation == "relu":
            self.activation = lambda x: np.maximum(0, x)
        elif activation == "tanh":
            self.activation = np.tanh
        elif activation == "sigmoid":
            self.activation = lambda x: 1 / (1 + np.exp(-x))
        else:
            raise ValueError("Unsupported activation function")

    def activate(self, x):
        """Apply FFN: activation(W1x)W2."""
        return np.dot(self.activation(np.dot(x, self.w1)), self.w2)


class TransformerEncoder:
    def __init__(self, length, dim_model, num_heads, dim_ff, tomask=False, activation="relu") -> NoReturn:
        """Initialize Transformer Encoder."""
        self.length = length
        self.dim_model = dim_model
        self.attention = MultiHeadAttention(length, dim_model, num_heads, tomask)
        self.ffn = FeedForwardNetwork(dim_model, dim_ff, activation)

        # Layer normalization parameters
        self.layer_norm1 = np.ones(dim_model)
        self.layer_norm2 = np.ones(dim_model)

    def _layer_norm(self, x):
        """Apply simple layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return (x - mean) / (std + 1e-6)

    def activate(self, x):
        """Perform the encoder layer computation."""
        # Multi-head attention with residual and normalization
        attention_output = self.attention.activate(x, x, x)
        x = x + attention_output  # Residual connection
        x = self._layer_norm(x) * self.layer_norm1  # Layer norm

        # Feed-forward network with residual and normalization
        ffn_output = self.ffn.activate(x)
        x = x + ffn_output  # Residual connection
        x = self._layer_norm(x) * self.layer_norm2  # Layer norm

        return x


def positional_encoding(length, dim_model) -> np.ndarray:
    """
    Generate positional encoding for a sequence.

    Args:
    - length (int): Length of the sequence.
    - dim_model (int): Dimensionality of the model.

    Returns:
    - np.ndarray: Positional encoding of shape (length, dim_model).
    """
    position = np.arange(length)[:, np.newaxis]  # Shape: (length, 1)
    i = np.arange(dim_model)[np.newaxis, :]  # Shape: (1, dim_model)
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(dim_model))

    pos_encoding = position * angle_rates
    pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])  # Apply sine to even indices
    pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])  # Apply cosine to odd indices

    return pos_encoding


