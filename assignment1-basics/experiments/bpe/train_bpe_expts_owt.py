from cs336_basics.bpe import train_bpe

vocab, merges = train_bpe("data/owt_train.txt", 32000, ["<|endoftext|>"], 64)

with open("results/owt-vocab.txt", "w") as f:
    for token_id, token_bytes in vocab.items():
        f.write(f"{token_id}\t{token_bytes.hex()}\n")

with open("results/readable/owt-vocab-readable.txt", "w") as f:
    for token_id, token_bytes in vocab.items():
        f.write(f"{token_id}: {token_bytes}\n")

with open("results/owt-merges.txt", "w") as f:
    for left_bytes, right_bytes in merges:
        f.write(f"{left_bytes.hex()}\t{right_bytes.hex()}\n")

with open("results/readable/owt-merges-readable.txt", "w") as f:
    for left_bytes, right_bytes in merges:
        f.write(f"({left_bytes}, {right_bytes})\n")

longest_token = max(vocab.values(), key=len)
print("longest_token =", longest_token)
