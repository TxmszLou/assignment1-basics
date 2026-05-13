import time
import json
import pathlib
import pickle

from cs336_basics.bpe import train_bpe

DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data"
train_file = "TinyStoriesV2-GPT4-train.txt"
vocab_size = 32000

if __name__ == "__main__":
    input_path = DATA_PATH / train_file
    start_time = time.time()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
        verbose=True
    )
    end_time = time.time()
    print(f"Training BPE on {train_file} took {end_time - start_time:.2f} seconds")

    print(f"Learned vocab size: {len(vocab)}")
    print(f"Learned merges size: {len(merges)}")

    print(f"Top 5 longest tokens in the learned vocab:")
    top_5_longest = sorted(vocab.items(), key=lambda x: len(x[1]), reverse=True)[:5]

    for idx, (token_id, token_bytes) in enumerate(top_5_longest):
        print(f"{idx+1}. ID {token_id}: {repr(token_bytes)} ({len(token_bytes)} bytes)")

    with open(f"{train_file}_{vocab_size}.pkl", "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)