import re
import collections
from typing import List, Tuple, Dict

class FastBPE:
    """
    A class for performing Byte Pair Encoding (BPE) tokenization and training.

    This class implements the BPE algorithm for subword tokenization. It allows users to:
    - Train a BPE tokenizer on a corpus of text.
    - Encode new text into subword tokens based on the learned BPE merges.

    BPE is useful for tasks such as machine translation, text generation, and other NLP tasks where handling out-of-vocabulary words is essential.

    Example Usage:

    1. **Initialize** the FastBPE tokenizer with an optional vocabulary and list of merges:
       ```python
       bpe_tokenizer = FastBPE(vocab={}, merges=[])
       ```

    2. **Read text from a file**:
       ```python
       text = bpe_tokenizer.read_file("example_text.txt")
       ```

    3. **Train the tokenizer** using the text and specify the number of merges:
       ```python
       bpe_tokenizer.train_bpe(text, num_merges=10)
       ```

    4. **Encode new text** using the trained BPE model:
       ```python
       encoded_tokens = bpe_tokenizer.encode("low lowery")
       ```

    Attributes:
        vocab (Dict[str, int]): A dictionary representing the vocabulary, where keys are words and values are frequencies.
        merges (List[Tuple[str, str]]): A list of token pair merges learned during BPE training.
        bpe_ranks (Dict[Tuple[str, str], int]): A mapping from token pairs to their merge rank.
        merged_vocab (Dict[str, List[str]]): A cached vocabulary where words are split into individual characters for encoding.
    """


    def __init__(self, vocab: Dict[str, int] = None, merges: List[Tuple[str, str]] = None):
        self.vocab = vocab if vocab else {}
        self.merges = merges if merges else []
        # Directly store merge ranks for fast lookup during encoding
        self.bpe_ranks = {pair: i for i, pair in enumerate(self.merges)}
        
        # Cache merged vocab to avoid recomputing during encoding
        self.merged_vocab = {}

    def get_vocab(self, text: str) -> Dict[str, int]:
        """Tokenizes text into word-level tokens and counts frequency."""
        words = re.findall(r'\S+', text)
        return collections.Counter(words)

    def encode(self, text: str) -> List[str]:
        """Encodes the input text using BPE with precomputed merges."""
        vocab = self.get_vocab(text)
        bpe_tokens = []

        # Cache merged vocab for fast access
        self._prepare_merged_vocab(vocab)

        # Apply BPE to each word
        for word in vocab:
            word_tokens = self._apply_bpe_to_word(word)
            bpe_tokens.extend(word_tokens)
        
        return bpe_tokens

    def _prepare_merged_vocab(self, vocab: Dict[str, int]):
        """Precomputes merged tokens for fast encoding lookup."""
        self.merged_vocab = {word: list(word) for word in vocab}

    def _apply_bpe_to_word(self, word: str) -> List[str]:
        """Applies the BPE encoding process to a single word."""
        word_tokens = self.merged_vocab[word]
        while len(word_tokens) > 1:
            pairs = [(word_tokens[i], word_tokens[i + 1]) for i in range(len(word_tokens) - 1)]
            # Find the most frequent pair to merge
            pair = min(pairs, key=lambda p: self.bpe_ranks.get(p, float('inf')))
            if pair in self.bpe_ranks:
                word_tokens = self.merge_pair(word_tokens, pair)
            else:
                break
        return word_tokens

    def merge_pair(self, word_tokens: List[str], pair: Tuple[str, str]) -> List[str]:
        """Merges a pair of tokens."""
        new_token = ''.join(pair)
        new_tokens = []
        i = 0
        while i < len(word_tokens):
            if i < len(word_tokens) - 1 and (word_tokens[i], word_tokens[i + 1]) == pair:
                new_tokens.append(new_token)
                i += 2  # Skip the next token as it has been merged
            else:
                new_tokens.append(word_tokens[i])
                i += 1
        return new_tokens
    
    def train_bpe(self, text: str, num_merges: int) -> None:
        """Trains the BPE tokenizer on a given text."""
        vocab = self.get_vocab(text)
        merges = []
        
        for _ in range(num_merges):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            best_pair = min(pairs, key=pairs.get)
            merges.append(best_pair)
            vocab = self.merge_vocab(vocab, best_pair)

        self.merges = merges
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        self.vocab = vocab
        self._prepare_merged_vocab(vocab)

    def get_stats(self, vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """Calculates the frequency of pairs of tokens in the vocabulary."""
        pairs = collections.defaultdict(int)
        for word, freq in vocab.items():
            symbols = list(word)
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def merge_vocab(self, vocab: Dict[str, int], pair: Tuple[str, str]) -> Dict[str, int]:
        """Merges a pair in the vocabulary."""
        bigram = ''.join(pair)
        new_vocab = {}
        for word, freq in vocab.items():
            new_word = word.replace(bigram, ''.join(pair))
            new_vocab[new_word] = freq
        return new_vocab

    def read_file(self, file_path: str) -> str:
        """Reads text data from a file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()



