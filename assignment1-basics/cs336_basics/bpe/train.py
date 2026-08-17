from cs336_basics.bpe.pretokenization import pretokenizer
from collections import Counter, defaultdict
import os


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # 1. Vocabulary initialization
    special_tokens.sort(key=len, reverse=True)  # 确保长的在前面，之后正则匹配会贪婪
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i in range(len(special_tokens)):
        vocab[256 + i] = special_tokens[i].encode("utf-8")

    # 2. Pre-tokenization
    # to get frequency table like {low: 5, lower: 2}
    initial_pretoken_frequencies: Counter[tuple[bytes, ...]] = pretokenizer(
        input_path, num_processes, special_tokens
    )

    # 3. Merges
    # 3.1 init
    # merges: [(b'a', b'b'), ]
    # tokens_by_pretoken_id: id -> pretoken (changable)
    # frequency_by_pretoken_id: id -> pretoken_count
    # pair_counts: {(b'a', b'b'): 2, (b'b', b'c'): 1, }
    # pretoken_ids_by_pair: {(b'a', b'b'): (1,)}, 1 is pretoken_id
    merges: list[tuple[bytes, bytes]] = list()
    tokens_by_pretoken_id: dict[int, tuple[bytes, ...]] = dict()
    frequency_by_pretoken_id: dict[int, int] = dict()
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pretoken_ids_by_pair: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    # 初始化四张表
    for pretoken_id, (current_tokens, pretoken_count) in enumerate(
        initial_pretoken_frequencies.items()
    ):
        tokens_by_pretoken_id[pretoken_id] = current_tokens
        frequency_by_pretoken_id[pretoken_id] = pretoken_count

        for i, (left_token, right_token) in enumerate(
            zip(current_tokens[:-1], current_tokens[1:])
        ):
            pair_counts[(left_token, right_token)] += pretoken_count
            pretoken_ids_by_pair[(left_token, right_token)].add(pretoken_id)
    # 找一下数值最大，然后字典序最大的那对，合并并计入词表
    while len(vocab) < vocab_size and pair_counts:
        selected_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
        merges.append(selected_pair)

        # 更新一下新的词表
        merged_token: bytes = selected_pair[0] + selected_pair[1]
        vocab[len(vocab)] = merged_token

        for pretoken_id in pretoken_ids_by_pair[selected_pair]:
            idx = 0
            while idx < len(tokens_by_pretoken_id[pretoken_id]) - 1:
                current_tokens: tuple[bytes, ...] = tokens_by_pretoken_id[pretoken_id]
                if (current_tokens[idx], current_tokens[idx + 1]) != selected_pair:
                    idx += 1
                    continue
                pretoken_count: int = frequency_by_pretoken_id[pretoken_id]

                merged_tokens: tuple[bytes, ...] = (
                    current_tokens[:idx] + (merged_token,) + current_tokens[idx + 2 :]
                )

                # 中间的token变了，前后 pair 对应的内容发生改变
                if idx > 0:
                    pair_counts[
                        (current_tokens[idx - 1], current_tokens[idx])
                    ] -= pretoken_count
                    if pair_counts[(current_tokens[idx - 1], current_tokens[idx])] == 0:
                        del pair_counts[(current_tokens[idx - 1], current_tokens[idx])]
                    pair_counts[
                        (current_tokens[idx - 1], merged_token)
                    ] += pretoken_count
                    pretoken_ids_by_pair[(current_tokens[idx - 1], merged_token)].add(
                        pretoken_id
                    )

                if idx < len(current_tokens) - 2:
                    pair_counts[
                        (current_tokens[idx + 1], current_tokens[idx + 2])
                    ] -= pretoken_count
                    if (
                        pair_counts[(current_tokens[idx + 1], current_tokens[idx + 2])]
                        == 0
                    ):
                        del pair_counts[
                            (current_tokens[idx + 1], current_tokens[idx + 2])
                        ]
                    pair_counts[
                        (merged_token, current_tokens[idx + 2])
                    ] += pretoken_count
                    pretoken_ids_by_pair[(merged_token, current_tokens[idx + 2])].add(
                        pretoken_id
                    )

                idx += 1
                initial_pretoken_frequencies[merged_tokens] = (
                    initial_pretoken_frequencies[current_tokens]
                )
                tokens_by_pretoken_id[pretoken_id] = merged_tokens
                del initial_pretoken_frequencies[current_tokens]
        del pretoken_ids_by_pair[selected_pair]
        del pair_counts[selected_pair]

    return (vocab, merges)


if __name__ == "__main__":
    a, b = train_bpe("data/demo.txt", 290, ["<|endoftext|>"], 4)
    print(b)
