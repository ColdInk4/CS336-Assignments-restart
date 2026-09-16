from cs336_basics.transformer import TransformerLM
from cs336_basics.transformer.functions import softmax
from jaxtyping import Int
import torch
from torch import Tensor
import tyro
from dataclasses import dataclass, field
from typing import Literal
from cs336_basics.bpe import Tokenizer


@dataclass
class ModelConfig:
    device: Literal["cpu", "cuda"] = "cuda"
    dtype: Literal["float32"] = "float32"


@dataclass
class CkptConfig:
    path: str


@dataclass
class TokenizerConfig:
    vocab_path: str
    merges_path: str
    special: list[str] = field(default_factory=lambda: ["<|endoftext|>"])


@dataclass
class GenerateConfig:
    temperature: float = 1.0
    max_new_tokens: int = 500
    top_p: float = 0.2


def main(
    ckpt_cfg: CkptConfig,
    model_cfg: ModelConfig,
    tokenizer_cfg: TokenizerConfig,
    generate_cfg: GenerateConfig,
    prompt: str,
):
    device = torch.device(model_cfg.device)
    dtype = getattr(torch, model_cfg.dtype)
    model = TransformerLM.from_checkpoint(ckpt_cfg.path, device=device, dtype=dtype)
    tokenizer = Tokenizer.from_files(
        tokenizer_cfg.vocab_path, tokenizer_cfg.merges_path, tokenizer_cfg.special
    )
    prompt_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)
    result_ids = model.generate_top_p(
        prompt_ids,
        max_new_tokens=generate_cfg.max_new_tokens,
        temperature=generate_cfg.temperature,
        top_p=generate_cfg.top_p,
        special_token_ids=tokenizer.special_token_ids,
    )
    result = tokenizer.decode(result_ids.tolist())
    print(result)


if __name__ == "__main__":
    tyro.cli(main)
