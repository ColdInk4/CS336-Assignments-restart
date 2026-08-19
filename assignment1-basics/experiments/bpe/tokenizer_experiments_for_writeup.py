from cs336_basics.bpe import Tokenizer
import random
import time

# 2.7 Experiments

TinyStories_Vocab_Filepath = "results/TinyStoriesV2-GPT4-vocab.txt"
TinyStories_Merges_Filepath = "results/TinyStoriesV2-GPT4-merges.txt"
TinyStories_Tokenizer = Tokenizer.from_files(
    TinyStories_Vocab_Filepath, TinyStories_Merges_Filepath
)

OWT_Vocab_Filepath = "results/owt-vocab.txt"
OWT_Merges_Filepath = "results/owt-merges.txt"
OWT_Tokenizer = Tokenizer.from_files(OWT_Vocab_Filepath, OWT_Merges_Filepath)

TinyStories_Valid_Filepath = "data/TinyStoriesV2-GPT4-valid.txt"
OWT_Valid_Filepath = "data/owt_valid.txt"

SPECIAL_STRING = "<|endoftext|>"

## (a)
with open(TinyStories_Valid_Filepath, "r") as f:
    content = f.read()
    documents = content.split(SPECIAL_STRING)
    sample_documents = random.sample(documents, 10)

    total_bytes = 0
    total_token_ids_byTinyStoriesTokenizer = 0
    total_token_ids_byOWTTokenizer = 0
    for document in sample_documents:
        total_bytes += len(document.encode("utf-8"))
        total_token_ids_byTinyStoriesTokenizer += len(
            TinyStories_Tokenizer.encode(document)
        )
        total_token_ids_byOWTTokenizer += len(OWT_Tokenizer.encode(document))
    print(
        f"TinyStories Tokenizer on TinyStories's compression ratio is {total_bytes/total_token_ids_byTinyStoriesTokenizer}"
    )
    print(
        f"OWT Tokenizer on TinyStories's compression ratio is {total_bytes/total_token_ids_byOWTTokenizer}"
    )
    # (c)
    total_file_bytes = len(content.encode("utf-8"))
    start_time = time.perf_counter()
    TinyStories_Tokenizer.encode(content)
    end_time = time.perf_counter()

    throughput = total_file_bytes / (end_time - start_time)
    print(
        f"throughput of my ts-tokenizer is {throughput} bytes/second, which is about {throughput/1024/1024} MB/s"
    )
    print(
        f"Time to tokenize the Pile dataset is about {825*1024*1024*1024/throughput} s, which is about {825*1024*1024*1024/throughput/60/60} hours"
    )

with open(OWT_Valid_Filepath, "r") as f:
    # (a)
    content = f.read()
    documents = content.split(SPECIAL_STRING)
    sample_documents = random.sample(documents, 10)

    total_bytes = 0
    total_token_ids_byTinyStoriesTokenizer = 0
    total_token_ids_byOWTTokenizer = 0
    for document in sample_documents:
        total_bytes += len(document.encode("utf-8"))
        total_token_ids_byTinyStoriesTokenizer += len(
            TinyStories_Tokenizer.encode(document)
        )
        total_token_ids_byOWTTokenizer += len(OWT_Tokenizer.encode(document))

    print(
        f"TinyStories Tokenizer on OWT's compression ratio is {total_bytes/total_token_ids_byTinyStoriesTokenizer}"
    )
    print(
        f"OWT Tokenizer on OWT's compression ratio is {total_bytes/total_token_ids_byOWTTokenizer}"
    )

    # (c)
    total_file_bytes = len(content.encode("utf-8"))
    start_time = time.perf_counter()
    OWT_Tokenizer.encode(content)
    end_time = time.perf_counter()

    throughput = total_file_bytes / (end_time - start_time)
    print(
        f"throughput of my owt-tokenizer is {throughput} bytes/second, which is about {throughput/1024/1024} MB/s"
    )
    print(
        f"Time to tokenize the Pile dataset is about {825*1024*1024*1024/throughput} s, which is about {825*1024*1024*1024/throughput/60/60} hours"
    )
