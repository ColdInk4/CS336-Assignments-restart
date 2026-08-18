import regex as re
from collections.abc import Iterable, Iterator

PRETOKEN_PATTERN_STR = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class Tokenizer:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str] | None = None
    vocab_inverse: dict[bytes, int]
    merges_order: dict[tuple[bytes, bytes], int]
    pretoken_pattern: re.Pattern
    special_tokens_pattern: re.Pattern | None

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
            for special_token in special_tokens:
                if special_token.encode("utf-8") not in self.vocab.values():
                    self.vocab[len(self.vocab)] = special_token.encode("utf-8")
            special_tokens_pattern_str = "|".join(
                [re.escape(special_token) for special_token in self.special_tokens]
            )
            self.special_tokens_pattern = re.compile(special_tokens_pattern_str)
        else:
            self.special_tokens = None
            self.special_tokens_pattern = None
        self.vocab_inverse = {
            token_bytes: token_id for token_id, token_bytes in self.vocab.items()
        }
        self.merges_order = {merge: i for i, merge in enumerate(merges)}
        self.pretoken_pattern = re.compile(PRETOKEN_PATTERN_STR)

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        vocab = dict()
        merges = []
        with open(vocab_filepath, "r") as f:
            for line in f:
                parts = line.split()
                token_id, token_bytes = int(parts[0]), bytes.fromhex(parts[1])
                vocab[token_id] = token_bytes

        with open(merges_filepath, "r") as f:

            for line in f:
                parts = line.split()
                left_token, right_token = bytes.fromhex(parts[0]), bytes.fromhex(
                    parts[1]
                )
                merges.append((left_token, right_token))

        return Tokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        # 1. Pre-tokenize
        if self.special_tokens_pattern:
            passages = self.special_tokens_pattern.split(text)
            special_tokens_iterator = self.special_tokens_pattern.finditer(text)
        else:
            passages = [text]

        result: list[int] = []

        for passage in passages:
            result.extend(self._encode_passage(passage))

            if self.special_tokens:
                cur_special_tokens = next(special_tokens_iterator, None)
                if cur_special_tokens:
                    result.append(
                        self.vocab_inverse[cur_special_tokens.group().encode("utf-8")]
                    )
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            for token in self.encode(chunk):
                yield token

    def decode(self, ids: list[int]) -> str:

        return (b"".join(self.vocab[id] for id in ids)).decode(
            "utf-8", errors="replace"
        )

    def _encode_passage(self, passage: str) -> list[int]:
        result = []

        for pretoken in self.pretoken_pattern.finditer(passage):
            result.extend(self._encode_pretoken(pretoken.group()))

        return result

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        result = []
        # represent each pre-token as a sequence of UTF-8 bytes
        pretoken_bytes = [bytes([i]) for i in pretoken.encode("utf-8")]

        merge_flag = True

        while merge_flag:
            merge_flag = False
            merge_pair: tuple[bytes, bytes] | None = None
            merge_position = -1
            for idx, (left_token, right_token) in enumerate(
                zip(pretoken_bytes[:-1], pretoken_bytes[1:])
            ):
                cur_pair = (left_token, right_token)
                if cur_pair in self.merges_order:
                    if (not merge_pair) or (
                        merge_pair
                        and self.merges_order[merge_pair] > self.merges_order[cur_pair]
                    ):
                        merge_pair = cur_pair
                        merge_position = idx
            if merge_pair:
                merge_flag = True
                pretoken_bytes = (
                    pretoken_bytes[:merge_position]
                    + [merge_pair[0] + merge_pair[1]]
                    + pretoken_bytes[merge_position + 2 :]
                )

        for cur_bytes in pretoken_bytes:
            result.append(self.vocab_inverse[cur_bytes])
        return result
