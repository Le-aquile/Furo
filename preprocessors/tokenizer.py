import functools
import collections
import re
from concurrent.futures import ThreadPoolExecutor
from typing import AbstractSet, Collection, Literal, NoReturn, Sequence, Dict, List, Tuple


class TokenBPEEncoder:
    """
    A class that combines both Byte Pair Encoding (BPE) and regular tokenization, with special token handling.

    This class provides functionalities for:
    - Encoding and decoding text using both regular encoding or BPE-based tokenization.
    - Training a BPE model on text data.
    - Handling special tokens during encoding and decoding.
    """

    def __init__(self, name: str, vocab: Dict[str, int] = None, merges: List[Tuple[str, str]] = None, 
                 special_tokens: Dict[str, int] = None, pattern: str = r'\S+', explicit_vocab_size: int | None = None):
        """
        Initializes the TokenBPEEncoder with BPE and special token capabilities.
        
        :param name: The name of the encoder.
        :param vocab: The vocabulary for BPE training (word-frequency dictionary).
        :param merges: A list of BPE merges.
        :param special_tokens: Dictionary of special tokens to handle during encoding/decoding.
        :param pattern: The pattern to use for regular tokenization.
        :param explicit_vocab_size: If specified, checks that vocab size matches the given size.
        """
        self.name = name
        self._pattern = pattern
        self._special_tokens = special_tokens or {}
        self.max_token_value = max(max(vocab.values(), default=0), max(self._special_tokens.values(), default=0))

        # Initialize the BPE model
        self.bpe = FastBPE(vocab, merges)
        
        # Regular token encoding logic
        self._core_bpe = self._init_core_bpe(vocab, self._special_tokens, self._pattern)

        if explicit_vocab_size:
            assert len(vocab) + len(special_tokens) == explicit_vocab_size
            assert self.max_token_value == explicit_vocab_size - 1

    def __repr__(self) -> str:
        return f"<TokenBPEEncoder {self.name!r}>"

    # ========================
    # BPE Encoding and Decoding
    # ========================

    def encode(self, text: str, use_bpe: bool = True) -> List[str]:
        """
        Encodes the input text using either regular encoding or BPE.
        
        :param text: Text to be encoded.
        :param use_bpe: Whether to use BPE tokenization or not.
        :return: A list of encoded tokens.
        """
        if use_bpe:
            return self.bpe.encode(text)
        else:
            return self._encode_regular(text)

    def encode_batch(self, texts: List[str], use_bpe: bool = True, num_threads: int = 8) -> List[List[str]]:
        """
        Encodes a batch of texts using either regular encoding or BPE.
        
        :param texts: List of texts to be encoded.
        :param use_bpe: Whether to use BPE tokenization or not.
        :param num_threads: Number of threads to use for parallel processing.
        :return: A list of lists containing encoded tokens for each input text.
        """
        encoder = functools.partial(self.encode, use_bpe=use_bpe)
        with ThreadPoolExecutor(num_threads) as executor:
            return list(executor.map(encoder, texts))

    def decode(self, tokens: List[str], errors: str = "replace") -> str:
        """
        Decodes a list of tokens back into a string.
        
        :param tokens: List of tokens to decode.
        :param errors: Handling strategy for errors in decoding.
        :return: Decoded string.
        """
        return self._decode_regular(tokens)

    def decode_batch(self, token_batches: List[List[str]], errors: str = "replace", num_threads: int = 8) -> List[str]:
        """
        Decodes a batch of token batches into strings.
        
        :param token_batches: List of token batches to decode.
        :param errors: Handling strategy for errors in decoding.
        :param num_threads: Number of threads to use for parallel decoding.
        :return: A list of decoded strings.
        """
        decoder = functools.partial(self.decode, errors=errors)
        with ThreadPoolExecutor(num_threads) as executor:
            return list(executor.map(decoder, token_batches))

    # ========================
    # Special Token Handling
    # ========================

    def encode_with_special(
        self,
        text: str,
        allowed_special: Literal["all"] | AbstractSet[str] = set(),
        disallowed_special: Literal["all"] | Collection[str] = "all",
    ) -> List[str]:
        """
        Encodes the text with special token handling.
        
        :param text: The text to encode.
        :param allowed_special: The set of allowed special tokens.
        :param disallowed_special: The set of disallowed special tokens.
        :return: List of tokens including special tokens where allowed.
        """
        if allowed_special == "all":
            allowed_special = self.special_tokens_set
        if disallowed_special == "all":
            disallowed_special = self.special_tokens_set - allowed_special
        
        # Check for disallowed special tokens
        if disallowed_special:
            if not isinstance(disallowed_special, frozenset):
                disallowed_special = frozenset(disallowed_special)
            if match := _special_token_regex(disallowed_special).search(text):
                raise_disallowed_special_token(match.group())

        return self._encode_regular(text)

    @property
    def special_tokens_set(self) -> set[str]:
        """Returns a set of all special token strings."""
        return set(self._special_tokens.keys())

    @property
    def vocab_size(self) -> int:
        """Returns the total vocabulary size (including special tokens)."""
        return self.max_token_value + 1

    # ========================
    # BPE Training
    # ========================

    def train_bpe(self, text: str, num_merges: int) -> None:
        """
        Trains the BPE tokenizer on a given text.
        
        :param text: Text data used to train the BPE model.
        :param num_merges: The number of merges to perform during training.
        """
        self.bpe.train_bpe(text, num_merges)

    # ========================
    # Private Helper Methods
    # ========================

    def _encode_regular(self, text: str) -> List[str]:
        """Regular encoding (non-BPE) based on the provided pattern."""
        return re.findall(self._pattern, text)

    def _decode_regular(self, tokens: List[str]) -> str:
        """Regular decoding logic (converts token list to string)."""
        return " ".join(tokens)

    def _init_core_bpe(self, vocab: Dict[str, int], special_tokens: Dict[str, int], pattern: str):
        """Initializes regular BPE + special token encoder."""
        return {
            "vocab": vocab,
            "special_tokens": special_tokens,
            "pattern": pattern
        }


class FastBPE:
    """Class for performing Byte Pair Encoding (BPE) tokenization"""

    def __init__(self, vocab: Dict[str, int] = None, merges: List[Tuple[str, str]] = None):
        """Initializes the FastBPE model with given vocabulary and merges."""
        self.vocab = vocab or {}
        self.merges = merges or []
        self.bpe_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.merged_vocab = {}

    def get_vocab(self, text: str) -> Dict[str, int]:
        """Tokenizes text into word-level tokens and counts frequency."""
        words = re.findall(r'\S+', text)
        return collections.Counter(words)

    def encode(self, text: str) -> List[str]:
        """Encodes the input text using BPE with precomputed merges."""
        vocab = self.get_vocab(text)
        bpe_tokens = []
        self._prepare_merged_vocab(vocab)

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
                i += 2
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

# Helper functions

def _special_token_regex(tokens: frozenset[str]) -> re.Pattern:
    """Returns a regex pattern for detecting special tokens."""
    return re.compile(r'|'.join(re.escape(token) for token in tokens))

def raise_disallowed_special_token(token: str) -> NoReturn:
    """Raises an error for disallowed special tokens."""
    raise ValueError(f"Disallowed special token found: {token}")