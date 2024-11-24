from collections import defaultdict
from typing import List, Tuple


class MiniBPE:
    def _init_(self, vocab_size: int) -> None:
        
        self.vocab_size = vocab_size
        self.tokenized_texts = []
        self.vocab = set()

    def train(self, texts: List[str]) -> Tuple[List[str], List[List[str]]]:
        self.tokenized_texts = self.tokenize(texts)
        self.vocab = set(char for tokens in self.tokenized_texts for char in tokens)

        while len(self.vocab) < self.vocab_size:
            pairs = self.count_pairs(self.tokenized_texts)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self.tokenized_texts = self.merge_pair(best_pair, self.tokenized_texts)
            self.vocab.add("".join(best_pair))

        return list(self.vocab), self.tokenized_texts

    def count_pairs(self, tokenized_texts: List[List[str]]) -> dict:
        # Create a mapping of tokens to indices
        token_to_idx = {token: idx for idx, token in enumerate(self.vocab)}

        # Create pairs of consecutive tokens
        pairs = defaultdict(int)
        for tokens in tokenized_texts:
            for i in range(len(tokens) - 1):
                pairs[(tokens[i], tokens[i + 1])] += 1

        return pairs

    def merge_pair(self, pair: Tuple[str, str], tokenized_texts: List[List[str]]) -> List[List[str]]:
        new_tokenized_texts = []
        for tokens in tokenized_texts:
            merged_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    merged_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    merged_tokens.append(tokens[i])
                    i += 1
            new_tokenized_texts.append(merged_tokens)
        return new_tokenized_texts

    @staticmethod
    def tokenize(texts: List[str]) -> List[List[str]]:
        return [list(word) + ["</w>"] for word in texts]  # Add </w> to signify word boundaries


    def final(self):
        {token: idx for idx, token in enumerate(self.vocab)}


    def decode(self, tokens):
        """
        Decodes a list of token indices back into their corresponding words using the vocabulary.

        Args:
            tokens (list): A list of token indices.

        Returns:
            list: A list of decoded words corresponding to the token indices.
        """
        vocab_dict = {v: k for k, v in enumerate(self.vocab)}
        return [vocab_dict[token] for token in tokens]
    

    def dump(self):
        """Saves the vocabulary to a file named vocab.vocab.
        Each token is saved on a separate line."""
        with open("vocab.vocab", "w") as f:
            for token in self.vocab:
                f.write(token + "\n")

    def load(self):
        """Loads the vocabulary from a file named vocab.vocab.
        Each token is expected to be on a separate line."""
        with open("vocab.txt", "r") as f:
            self.vocab = set(f.read().splitlines())
    

    def digest(self, filename):
        """
        Reads a file line-by-line, tokenizes each line, appends it to tokenized_texts, and trains the model on the new tokens.
        """
        with open(filename, "r") as f:
            for line in f:
                tokens = self.tokenize(line)
                self.tokenized_texts.append(tokens)
                self.train(tokens)
                
    def _add_(self, other):
        """
        Overloads the '+' operator to combine the vocabularies of two MiniBPE objects.
        The resulting MiniBPE object will have a vocabulary that is the union of the two operand vocabularies.
        """
        
        if isinstance(other, MiniBPE):
            combined_vocab = self.vocab.union(other.vocab)
            return MiniBPE(len(combined_vocab))._replace_vocab(combined_vocab)
        else:
            raise TypeError("Operand must be an instance of MiniBPE")
    
    def _replace_vocab(self, new_vocab):
        """
        Replaces the vocabulary of this MiniBPE object with the given new vocabulary.

        Args:
            new_vocab (set): The new vocabulary to replace the existing one.

        Returns:
            MiniBPE: The modified MiniBPE object.
        """

        self.vocab = new_vocab
        return self
    
    def _len_(self) -> int:
        """
        Returns the size of the vocabulary.

        Returns:
            int: The number of elements in the vocabulary.
        """
        return len(self.vocab)

    def _eq_(self, value: object) -> bool:
        """Checks if the given MiniBPE object has the same vocabulary as this object.

        Args:
            value (object): The object to compare.

        Returns:
            bool: True if the vocabularies are equal, False otherwise.
        """
        if isinstance(value, MiniBPE):
            return self.vocab == value.vocab
        return False
