import os
from typing import BinaryIO

import multiprocessing as mp
import regex as re
import itertools

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    Each chunk should start at the beginning of the split_special_token.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    # bi for boundary index
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pre_tokenize(input_path: str, start: int, end: int, special_tokens: list[str]):
    '''
    Pre-tokenize the chunk using the same regex pattern, and count the occurrences of each pre-token (except for special tokens).
    '''
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        # split the chunk by special tokens
        # need to escape special tokens for regex splitting
        escaped_special_tokens = [re.escape(special_token) for special_token in special_tokens]
        splits = re.split('|'.join(escaped_special_tokens), chunk)

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # store counts for each pre-token
        counts = {}
        for split in splits:
            for pre_token in re.finditer(PAT, split):
                token_bytes = tuple(pre_token.group().encode("utf-8"))
                counts[token_bytes] = counts.get(token_bytes, 0) + 1

        return counts

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    '''
    Input:
        input_path: str. Path to a text file with BPE tokenizer training data.
        vocab_size: int. A positive integer that defines the maximum final vocab size (including the initial byte vocab, vocab items produced from merging and any special tokens).
        special_tokens: list[str]. A list of str to add to the vocab. During training, treat them as hard boundaries that prevent merges across their spans, but do not include them when computing merge statistics.

    Output:
        vocab: dict[int, bytes]. The tokenizer vocab, a mapping from int (token ID in the vocab) to bytes (token bytes).
        merges: list[tuple[bytes, bytes]]. A list of BPE merges produced from training. Each list item is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with <token2>. The merges should be ordered by order of creation.
    '''

    pre_token_count = {} # pre-token bytes tuple -> count
    vocab = {}
    merges = []

    # Step 1: Parallel pre-tokenization
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # parallel pre-tokenize each chunk
        jobs = zip(itertools.repeat(input_path),
                   boundaries[:-1],
                   boundaries[1:],
                   itertools.repeat(special_tokens))

        with mp.Pool(processes=num_processes) as pool:
            results = pool.starmap(pre_tokenize, jobs)

            # aggregate counts from each process
            for counts in results:
                for token_bytes, count in counts.items():
                    pre_token_count[token_bytes] = pre_token_count.get(token_bytes, 0) + count
            
            print(pre_token_count)
 
    return vocab, merges