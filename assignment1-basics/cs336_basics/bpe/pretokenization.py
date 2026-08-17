import os, cProfile, pstats
from typing import BinaryIO
import regex as re
from collections import Counter
from multiprocessing import Pool
from functools import partial

PROFILE_PRETOKENIZATION: bool = False


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """

    if split_special_tokens:
        assert isinstance(
            split_special_tokens[0], bytes
        ), "Must represent special token as a bytestring"

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

            if split_special_tokens:
                found_at_ids = [
                    pos
                    for split_special_token in split_special_tokens
                    if (pos := mini_chunk.find(split_special_token)) >= 0
                ]
                found_at = min(found_at_ids) if found_at_ids else -1
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def worker(
    input_path: str | os.PathLike,
    boundaries: tuple[int, int],
    special_tokens: list[str],
) -> Counter[tuple[bytes, ...]]:
    # 正则表达式
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    special_tokens_PAT = "|".join(
        [re.escape(special_token) for special_token in special_tokens]
    )
    # 获取当前chunk
    with open(input_path, "rb") as f:
        start, end = boundaries
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    # 确保 special_tokens 为空时不报错
    if special_tokens_PAT:
        texts = re.split(special_tokens_PAT, chunk)

    freq = Counter()
    for text in texts:
        for token in re.finditer(PAT, text):
            token_bytes = tuple(bytes([i]) for i in token.group().encode("utf-8"))
            freq[token_bytes] += 1
    return freq


def pretokenizer(
    input_path: str | os.PathLike, num_processes: int, special_tokens: list[str]
) -> Counter[tuple[bytes, ...]]:

    print("=====Start Pretokenizer=====")
    if PROFILE_PRETOKENIZATION:
        profiler = cProfile.Profile()
        profiler.enable()

    frequency_table = Counter()

    # 找到各个边界
    with open(input_path, "rb") as f:
        special_tokens_bytes = [
            special_token.encode("utf-8") for special_token in special_tokens
        ]
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens_bytes)

    # 锁定 worker 的两个参数
    cur_worker = partial(worker, input_path, special_tokens=special_tokens)

    # 并行做worker
    with Pool(processes=num_processes) as pool:
        counters = pool.map(cur_worker, zip(boundaries[:-1], boundaries[1:]))

    # 合并各个计数器
    for counter in counters:
        frequency_table.update(counter)

    print("=====Finish Pretokenizer=====")
    if PROFILE_PRETOKENIZATION:
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumtime").print_stats(20)
    return frequency_table


if __name__ == "__main__":
    print(
        pretokenizer(
            "/home/jiepengjin/Study/CS336-new/assignment1-basics/data/demo.txt",
            4,
            ["<|endoftext|>"],
        )
    )
    a = Counter(
        {
            (b"\n",): 3,
            (b" ", b"h", b"a", b"p", b"p", b"y"): 3,
            (b"h", b"a", b"p", b"p", b"y"): 1,
            (b" ", b"h", b"a", b"p", b"p"): 1,
            (b",",): 1,
            (b" ", b"h", b"a", b"u", b"g", b"n", b"t"): 1,
        }
    )
    print(a[(b" ", b"h", b"a", b"p", b"p", b"y")])
