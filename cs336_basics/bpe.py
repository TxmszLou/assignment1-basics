import os
from typing import BinaryIO

import multiprocessing as mp
import regex as re
import itertools
from collections import defaultdict, Counter
import heapq

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    Each chunk should start at the beginning of the split_special_token.
    """
    # assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"
    assert isinstance(split_special_tokens, list) and all(isinstance(token, bytes) for token in split_special_tokens), "Must represent special tokens as a list of bytestrings"

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
            special_token_found = False
            for split_special_token in split_special_tokens:
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    special_token_found = True
                    break

            if special_token_found:
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
                bytestring = pre_token.group().encode("utf-8")
                token_bytes = tuple(bytestring[i:i+1] for i in range(len(bytestring)))
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

    pretoken_count = {} # pre-token bytes tuple -> count
    vocab = {}

    # initialize vocab dict for 0-255 byte and special tokens
    for i in range(256):
        vocab[i] = bytes([i])

    for i, special_token in enumerate(special_tokens):
        vocab[256 + i] = special_token.encode("utf-8")

    merges = []

    # Step 1: Parallel pre-tokenization
    with open(input_path, "rb") as f:
        num_processes = max(1, os.cpu_count() - 1)
        # boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        boundaries = find_chunk_boundaries(f, num_processes, [token.encode("utf-8") for token in special_tokens])

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
                    pretoken_count[token_bytes] = pretoken_count.get(token_bytes, 0) + count
            
        # Step 2: Train BPE merges
        pair_counts = {} # (token_bytes1, token_bytes2) -> current freq
        pair_to_pretoken_bytes = defaultdict(set) # (token_bytes1, token_bytes2) -> set of pre-token bytes that contain this pair
        for token_bytes, count in pretoken_count.items():
            for pair in zip(token_bytes[:-1], token_bytes[1:]):
                pair_to_pretoken_bytes[pair].add(token_bytes)
                pair_counts[pair] = pair_counts.get(pair, 0) + count
        
        # max heap of pairs by count, not exact, the count gives an upper bound for the true count
        pair_counts_heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(pair_counts_heap)

        while len(vocab) < vocab_size:
            if not pair_counts:
                break

            most_common_pair_candidates = []
            while pair_counts_heap and (pair_counts_heap[0][1] not in pair_counts or pair_counts[pair_counts_heap[0][1]] != -pair_counts_heap[0][0]):
                heapq.heappop(pair_counts_heap)

            if not pair_counts_heap:
                break

            # resolve ties with largest lexicographically pair
            max_count = -pair_counts_heap[0][0]
            while pair_counts_heap and -pair_counts_heap[0][0] == max_count:
                (neg_count, pair) = heapq.heappop(pair_counts_heap)
                if pair in pair_counts and pair_counts[pair] == -neg_count:
                    most_common_pair_candidates.append(pair)
            most_common_pair = max(most_common_pair_candidates)

            # push back the other candidates that weren't chosen
            for candidate in most_common_pair_candidates:
                if candidate != most_common_pair:
                    heapq.heappush(pair_counts_heap, (-max_count, candidate))

            merges.append(most_common_pair)

            most_common_pair_bytes = most_common_pair[0] + most_common_pair[1]
            assert most_common_pair_bytes not in vocab.values(), "Most common pair already in vocab, cannot merge further"

            vocab[len(vocab)] = most_common_pair_bytes

            affected_pretoken_bytes = list(pair_to_pretoken_bytes[most_common_pair])

            for pretoken_bytes in affected_pretoken_bytes:
                # compute adjacent pair count of the old sequence
                old_pairs = list(zip(pretoken_bytes[:-1], pretoken_bytes[1:]))
                old_pair_counts = Counter(old_pairs)

                # apply the merge to get the new pretoken bytes
                new_pretoken_bytes = []
                i = 0
                while i < len(pretoken_bytes):
                    if i < len(pretoken_bytes) - 1 and (pretoken_bytes[i], pretoken_bytes[i+1]) == most_common_pair:
                        new_pretoken_bytes.append(most_common_pair_bytes)
                        i += 2
                    else:
                        new_pretoken_bytes.append(pretoken_bytes[i])
                        i += 1

                new_pretoken_bytes = tuple(new_pretoken_bytes)
                
                new_pairs = list(zip(new_pretoken_bytes[:-1], new_pretoken_bytes[1:]))
                new_pairs_counts = Counter(new_pairs)

                # update pair_counts using old and new pair counts
                for pair in set(old_pairs + new_pairs):
                    if pair in old_pair_counts and pair in new_pairs_counts and old_pair_counts[pair] == new_pairs_counts[pair]:
                        # don't need to update these counts
                        continue
                        
                    pair_counts[pair] = pair_counts.get(pair, 0) + (new_pairs_counts.get(pair, 0) - old_pair_counts.get(pair, 0)) * pretoken_count[pretoken_bytes]
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    else:
                        heapq.heappush(pair_counts_heap, (-pair_counts[pair], pair))

                # update pair_to_pretoken_bytes for old and new pairs
                for pair in set(old_pairs + new_pairs):
                    if pair in pair_to_pretoken_bytes:
                        pair_to_pretoken_bytes[pair].discard(pretoken_bytes)
                    if pair in new_pairs_counts:
                        pair_to_pretoken_bytes[pair].add(new_pretoken_bytes)

                # update pretoken_count
                pretoken_count[new_pretoken_bytes] = pretoken_count.get(new_pretoken_bytes, 0) + pretoken_count[pretoken_bytes]
                pretoken_count.pop(pretoken_bytes)
 
    return vocab, merges