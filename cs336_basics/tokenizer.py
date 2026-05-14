from typing import Iterable, Iterator
import regex as re

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        """
        Constructs a tokenizer from a given vocab, list of merges, and optionally a list of special tokens.
        Args:
            vocab: dict[int, bytes]
            merges: list[tuple[bytes, bytes]]
            special_toens: list[str] | None = None
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []

        # regex pattern to split on special tokens during pre-tokenization
        # match longer special tokens first to avoid overlapping matches (e.g. <|endoftext|><|endoftext|> should be matched before <|endoftext|>)
        self.escaped_special_tokens_re = '|'.join([re.escape(token) \
             for token in sorted(self.special_tokens, key=lambda s: -len(s))])
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # reverse mapping for decoding
        self.token_to_id = { token_bytes: token_id for token_id, token_bytes in vocab.items() }

        # quick access to merge ranks for encoding
        self.merge_ranks = {}
        for i, merge in enumerate(merges):
            self.merge_ranks[merge] = i


    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        Constructs and returns a `Tokenizer` from a serialized vocab and list of merges
        (in the same format as bpe training code output) and optionally a list of special tokens.
        Args:
            vocab_filepath: str
            merges_filepath: str
            special_tokens: list[str] | None = None
        """
        pass

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs
        """

        encoded = []

        # Step 1: pre-tokenize
        if self.special_tokens:
            splits = re.split(f"({self.escaped_special_tokens_re})", text)
        else:
            splits = [text]

        # Step 2: apply merges to pretokens
        for split in splits:
            if split in self.special_tokens:
                # if the split is a special token, add its ID and continue
                encoded.append(self.token_to_id[split.encode("utf-8")])
                continue
            for pretoken in re.finditer(self.PAT, split):
                # apply merges to the pretoken
                bytestring = pretoken.group().encode("utf-8")
                token_bytes = [bytestring[i:i+1] for i in range(len(bytestring))]


                while True:
                    pairs = [(token_bytes[i], token_bytes[i+1]) for i in range(len(token_bytes)-1)]
                    possible_merge_pairs = [pair for pair in pairs if pair in self.merge_ranks]

                    if not possible_merge_pairs:
                        break

                    best_pair = min(possible_merge_pairs, key=lambda pair: self.merge_ranks[pair])
                    merged_bytes = best_pair[0] + best_pair[1]

                    new_token_bytes = []
                    i = 0
                    while i < len(token_bytes):
                        if i < len(token_bytes) - 1 and (token_bytes[i], token_bytes[i+1]) == best_pair:
                            new_token_bytes.append(merged_bytes)
                            i += 2  # skip the next byte since it's merged
                        else:
                            new_token_bytes.append(token_bytes[i])
                            i += 1

                    token_bytes = new_token_bytes

                # done with merging, now convert to token IDs and add to output
                for token_byte in token_bytes:
                    token_id = self.token_to_id[token_byte]
                    encoded.append(token_id)

        return encoded


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g. a Python file handle), return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        decoded_bytes = []

        for token_id in ids:
            if token_id not in self.vocab:
                raise ValueError(f"Token ID {token_id} is not in the vocabulary.")

            token_bytes = self.vocab[token_id]
            decoded_bytes.append(token_bytes)

        return b"".join(decoded_bytes).decode("utf-8", errors="replace")