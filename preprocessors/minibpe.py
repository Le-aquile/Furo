from collections import defaultdict
from typing import List, Tuple


class MiniBPE:
    def __init__(self, vocab_size: int) -> None:
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

if __name__ == "__main__":
    # Example texts for training
    texts = [
        "low",
        "lower",
        "newer",
        "widow"
    ]

    # Specify the desired vocabulary size
    vocab_size = 12

    # Create an instance of MiniBPE
    bpe = MiniBPE(vocab_size=vocab_size)

    # Train the BPE model on the texts
    vocab, tokenized_texts = bpe.train(texts)

    # Print the resulting vocabulary
    print("Vocabulary:")
    print(vocab)

    # Print the tokenized texts after training
    print("\nTokenized Texts:")
    for text in tokenized_texts:
        print(text)

