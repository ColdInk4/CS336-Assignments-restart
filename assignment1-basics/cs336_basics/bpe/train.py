from cs336_basics.bpe.pretokenization import pretokenizer
from collections import Counter, defaultdict
import os, cProfile, pstats
import heapq
from dataclasses import dataclass

PROFILE_BPE_TRAINING: bool = False


@dataclass
class PairCount:
    pair: tuple[bytes, bytes]
    count: int

    # 确保用堆的时候能按这个顺序把需要的最大值放到顶部
    def __lt__(self, other):
        if self.count != other.count:
            return self.count > other.count
        return self.pair > other.pair


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    print("====Start Training BPE====")
    if PROFILE_BPE_TRAINING:
        profiler = cProfile.Profile()
        profiler.enable()

    # 1. Vocabulary initialization
    sorted_special_tokens = sorted(
        special_tokens, key=len, reverse=True
    )  # 确保长的在前面，之后正则匹配会贪婪
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i in range(len(sorted_special_tokens)):
        vocab[256 + i] = sorted_special_tokens[i].encode("utf-8")

    # 2. Pre-tokenization
    # to get frequency table like {low: 5, lower: 2}
    initial_pretoken_frequencies: Counter[tuple[bytes, ...]] = pretokenizer(
        input_path, num_processes, sorted_special_tokens
    )

    # 3. Merges
    # 3.1 init
    # 3.1.1 Initialize merge state
    # merges: [(b'a', b'b'), ]
    merges: list[tuple[bytes, bytes]] = list()

    # 3.1.2 Build initial pair indexes
    # tokens_by_pretoken_id: id -> pretoken (changable)
    # frequency_by_pretoken_id: id -> pretoken_count
    # pair_counts: {(b'a', b'b'): 2, (b'b', b'c'): 1, }
    # pretoken_ids_by_pair: {(b'a', b'b'): (1,)}, 1 is pretoken_id
    tokens_by_pretoken_id: dict[int, tuple[bytes, ...]] = dict()
    frequency_by_pretoken_id: dict[int, int] = dict()
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pretoken_ids_by_pair: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    for pretoken_id, (current_tokens, pretoken_frequency) in enumerate(
        initial_pretoken_frequencies.items()
    ):
        tokens_by_pretoken_id[pretoken_id] = current_tokens
        frequency_by_pretoken_id[pretoken_id] = pretoken_frequency

        for i, (left_token, right_token) in enumerate(
            zip(current_tokens[:-1], current_tokens[1:])
        ):
            pair_counts[(left_token, right_token)] += pretoken_frequency
            pretoken_ids_by_pair[(left_token, right_token)].add(pretoken_id)

    # 建堆，来获取最大值
    heap = [PairCount(pair, count) for pair, count in pair_counts.items()]
    heapq.heapify(heap)
    # 3.2 找一下出现次数最多，字典序最大的那对，合并并计入词表
    while len(vocab) < vocab_size and pair_counts:
        while True:
            max_pair_count = heapq.heappop(heap)
            if (
                max_pair_count.pair in pair_counts
                and pair_counts[max_pair_count.pair] == max_pair_count.count
            ):
                selected_pair = max_pair_count.pair
                break
        merges.append(selected_pair)

        # 更新一下新的词表
        merged_token: bytes = selected_pair[0] + selected_pair[1]
        vocab[len(vocab)] = merged_token

        for pretoken_id in pretoken_ids_by_pair[selected_pair]:
            idx = 0
            # 当 idx, idx + 1 存在时：
            while idx < len(tokens_by_pretoken_id[pretoken_id]) - 1:
                current_tokens: tuple[bytes, ...] = tokens_by_pretoken_id[pretoken_id]
                # 如果匹配
                if (current_tokens[idx], current_tokens[idx + 1]) == selected_pair:
                    pretoken_frequency: int = frequency_by_pretoken_id[pretoken_id]
                    merged_tokens: tuple[bytes, ...] = (
                        current_tokens[:idx]
                        + (merged_token,)
                        + current_tokens[idx + 2 :]
                    )

                    # 中间的 token 变了，前后 pair 对应的内容发生改变
                    if idx > 0:
                        # 原始 [idx - 1], [idx] 项的内容移除，换成 [idx - 1], new_token
                        # 对 pair counts 的处理
                        pair_counts[
                            (current_tokens[idx - 1], current_tokens[idx])
                        ] -= pretoken_frequency
                        if (
                            pair_counts[(current_tokens[idx - 1], current_tokens[idx])]
                            == 0
                        ):
                            del pair_counts[
                                (current_tokens[idx - 1], current_tokens[idx])
                            ]
                        else:
                            heapq.heappush(
                                heap,
                                PairCount(
                                    (current_tokens[idx - 1], current_tokens[idx]),
                                    pair_counts[
                                        (current_tokens[idx - 1], current_tokens[idx])
                                    ],
                                ),
                            )

                        pair_counts[
                            (current_tokens[idx - 1], merged_token)
                        ] += pretoken_frequency
                        heapq.heappush(
                            heap,
                            PairCount(
                                (current_tokens[idx - 1], merged_token),
                                pair_counts[(current_tokens[idx - 1], merged_token)],
                            ),
                        )

                        # 对 pretoken_ids_by_pair 的处理（目前没有处理原先相邻的部分，因为不好判断组合是否还存在）
                        pretoken_ids_by_pair[
                            (current_tokens[idx - 1], merged_token)
                        ].add(pretoken_id)

                    if idx < len(current_tokens) - 2:
                        # 原始 [idx + 1], [idx + 2] 项的内容移除，换成 new_token, [idx + 2]
                        # 对 pair counts 的处理
                        pair_counts[
                            (current_tokens[idx + 1], current_tokens[idx + 2])
                        ] -= pretoken_frequency
                        if (
                            pair_counts[
                                (current_tokens[idx + 1], current_tokens[idx + 2])
                            ]
                            == 0
                        ):
                            del pair_counts[
                                (current_tokens[idx + 1], current_tokens[idx + 2])
                            ]
                        else:
                            heapq.heappush(
                                heap,
                                PairCount(
                                    (current_tokens[idx + 1], current_tokens[idx + 2]),
                                    pair_counts[
                                        (
                                            current_tokens[idx + 1],
                                            current_tokens[idx + 2],
                                        )
                                    ],
                                ),
                            )
                        pair_counts[
                            (merged_token, current_tokens[idx + 2])
                        ] += pretoken_frequency
                        heapq.heappush(
                            heap,
                            PairCount(
                                (merged_token, current_tokens[idx + 2]),
                                pair_counts[(merged_token, current_tokens[idx + 2])],
                            ),
                        )
                        # 对 pretoken_ids_by_pair 的处理（目前没有处理原先相邻的部分，因为不好判断组合是否还存在）
                        pretoken_ids_by_pair[
                            (merged_token, current_tokens[idx + 2])
                        ].add(pretoken_id)

                    tokens_by_pretoken_id[pretoken_id] = merged_tokens

                idx += 1

        del pretoken_ids_by_pair[selected_pair]
        del pair_counts[selected_pair]

    if PROFILE_BPE_TRAINING:
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumtime").print_stats(20)

    return (vocab, merges)


if __name__ == "__main__":
    vocab, merges = train_bpe(
        "data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"], 64
    )
