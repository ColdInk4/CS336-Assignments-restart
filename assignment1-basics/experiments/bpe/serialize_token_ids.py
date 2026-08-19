import numpy as np
from cs336_basics.bpe import Tokenizer
import time
from cs336_basics.bpe.pretokenization import find_chunk_boundaries
import os

TinyStories_Vocab_Filepath = "results/TinyStoriesV2-GPT4-vocab.txt"
TinyStories_Merges_Filepath = "results/TinyStoriesV2-GPT4-merges.txt"
TinyStories_Tokenizer = Tokenizer.from_files(
    TinyStories_Vocab_Filepath,
    TinyStories_Merges_Filepath,
    special_tokens=["<|endoftext|>"],
)

OWT_Vocab_Filepath = "results/owt-vocab.txt"
OWT_Merges_Filepath = "results/owt-merges.txt"
OWT_Tokenizer = Tokenizer.from_files(
    OWT_Vocab_Filepath, OWT_Merges_Filepath, special_tokens=["<|endoftext|>"]
)

TinyStories_Valid_Filepath = "data/TinyStoriesV2-GPT4-valid.txt"
TinyStories_Train_Filepath = "data/TinyStoriesV2-GPT4-train.txt"
OWT_Valid_Filepath = "data/owt_valid.txt"
OWT_Train_Filepath = "data/owt_train.txt"

Result_Filepath_TinyStories_Valid = "results/tokenids/ts-valid-tokenids.npy"
Result_Filepath_TinyStories_Train = "results/tokenids/ts-train-tokenids.npy"
Result_Filepath_OWT_Valid = "results/tokenids/owt-valid-tokenids.npy"
Result_Filepath_OWT_Train = "results/tokenids/owt-train-tokenids.npy"

CHUNK_SIZE = 1024 * 1024


def iter_chunk(file_path: str, special_tokens):
    with open(file_path, "rb") as f:
        file_size_bytes = os.path.getsize(f.name)
        num_chunks = file_size_bytes // CHUNK_SIZE + 1
        special_tokens_bytes = [
            special_token.encode("utf-8") for special_token in special_tokens
        ]
        boundaries = find_chunk_boundaries(f, num_chunks, special_tokens_bytes)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            yield chunk


def file_to_nparr(input_path: str, output_path: str, tokenizer: Tokenizer):
    print(f"Start processing: {input_path}")
    start_time = time.time()

    token_ids = np.fromiter(
        tokenizer.encode_iterable(iter_chunk(input_path, tokenizer.special_tokens)),
        dtype=np.uint16,
    )

    elapsed = time.time() - start_time
    print(f"num of tokens: {len(token_ids)} | time: {elapsed:.2f}s")
    np.save(output_path, token_ids)


file_to_nparr(
    TinyStories_Valid_Filepath, Result_Filepath_TinyStories_Valid, TinyStories_Tokenizer
)
file_to_nparr(
    TinyStories_Train_Filepath, Result_Filepath_TinyStories_Train, TinyStories_Tokenizer
)
file_to_nparr(OWT_Valid_Filepath, Result_Filepath_OWT_Valid, OWT_Tokenizer)
file_to_nparr(OWT_Train_Filepath, Result_Filepath_OWT_Train, OWT_Tokenizer)
