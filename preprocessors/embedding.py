import numpy as np
from typing import List, NoReturn

class TokenEmbedding:
    def __init__(self, vocab_size: int, embedding_dim: int) -> NoReturn:
        """
        Initialize the embedding matrix.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimensionality of the embedding vectors.
        """
        self.vocab_size: int = vocab_size
        self.embedding_dim: int = embedding_dim
        # Initialize the embedding matrix with random values
        self.embeddings: np.ndarray = np.random.randn(vocab_size, embedding_dim) * 0.01

    def forward(self, token_ids: List[int]) -> np.ndarray:
        """
        Retrieve embeddings for a batch of token IDs.

        Args:
            token_ids (List[int]): List of token IDs.

        Returns:
            np.ndarray: Corresponding embeddings of shape (len(token_ids), embedding_dim).
        """
        return self.embeddings[token_ids]

    def update_embeddings(self, token_ids: List[int], gradients: np.ndarray, learning_rate: float) -> NoReturn:
        """
        Update the embedding vectors for specific token IDs.

        Args:
            token_ids (List[int]): List of token IDs.
            gradients (np.ndarray): Gradients for the embeddings of shape (len(token_ids), embedding_dim).
            learning_rate (float): Learning rate for the update.
        """
        for i, token_id in enumerate(token_ids):
            self.embeddings[token_id] -= learning_rate * gradients[i]

    def get_embedding(self, token_id: int) -> np.ndarray:
        """
        Retrieve the embedding for a single token ID.

        Args:
            token_id (int): The token ID.

        Returns:
            np.ndarray: The embedding vector of the token.
        """
        return self.embeddings[token_id]

    def get_vocab_size(self) -> int:
        """
        Returns the size of the vocabulary.

        Returns:
            int: The size of the vocabulary.
        """
        return self.vocab_size

    def get_embedding_dim(self) -> int:
        """
        Returns the dimensionality of the embedding vectors.

        Returns:
            int: The dimensionality of the embedding vectors.
        """
        return self.embedding_dim

# Example usage:
if __name__ == "__main__":
    vocab_size: int = 10000
    embedding_dim: int = 128
    embedding: TokenEmbedding = TokenEmbedding(vocab_size, embedding_dim)

    # Example token IDs
    token_ids: List[int] = [1, 5, 7]
    embeddings: np.ndarray = embedding.forward(token_ids)


    # Example gradient update
    gradients: np.ndarray = np.random.randn(len(token_ids), embedding_dim)
    learning_rate: float = 0.01
    embedding.update_embeddings(token_ids, gradients, learning_rate)

    # Get single token embedding
    single_embedding: np.ndarray = embedding.get_embedding(1)
    print("Single token embedding shape:", single_embedding.shape)
