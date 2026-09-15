from cs336_basics.transformer import TransformerLM
from cs336_basics.training import (
    AdamW,
    SGD,
    get_batch,
    load_checkpoint,
    save_checkpoint,
    lr_cosine_schedule,
    cross_entropy,
    gradient_clipping,
)
from dataclasses import dataclass
import torch
import tyro
from typing import Literal, Union
import numpy as np
from loguru import logger
from pathlib import Path
import random


@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float
    device: Literal["cpu", "cuda"] = "cuda"
    dtype: Literal["float32"] = "float32"


@dataclass
class AdamWConfig:
    lr: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float


@dataclass
class SGDConfig:
    lr: float


@dataclass
class OptimizerConfig:
    optim: Union[AdamWConfig, SGDConfig]


@dataclass
class TrainingConfig:
    train_data_path: str
    val_data_path: str
    max_iters: int
    batch_size: int
    max_l2_norm: float
    seed: int = 42


@dataclass
class LogConfig:
    log_interval: int


@dataclass
class EvalConfig:
    eval_interval: int
    eval_batches: int


@dataclass
class CheckpointConfig:
    interval: int
    out_dir: str
    resume_from: str | None = None


@dataclass
class ScheduleConfig:
    max_learning_rate: float
    min_learning_rate: float
    warmup_iters: int
    cosine_cycle_iters: int


def main(
    model_cfg: ModelConfig,
    optimizer_cfg: OptimizerConfig,
    train_cfg: TrainingConfig,
    ckpt_cfg: CheckpointConfig,
    schedule_cfg: ScheduleConfig,
    log_cfg: LogConfig,
    eval_cfg: EvalConfig,
):
    torch.manual_seed(train_cfg.seed)
    torch.cuda.manual_seed_all(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    random.seed(train_cfg.seed)

    out_dir = Path(ckpt_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(model_cfg.device)
    dtype = getattr(torch, model_cfg.dtype)
    model = TransformerLM(
        model_cfg.vocab_size,
        model_cfg.context_length,
        model_cfg.d_model,
        model_cfg.num_layers,
        model_cfg.num_heads,
        model_cfg.d_ff,
        model_cfg.rope_theta,
        device,
        dtype,
    )
    if isinstance(optimizer_cfg.optim, AdamWConfig):
        opt = AdamW(
            model.parameters(),
            optimizer_cfg.optim.lr,
            (optimizer_cfg.optim.beta1, optimizer_cfg.optim.beta2),
            optimizer_cfg.optim.eps,
            optimizer_cfg.optim.weight_decay,
        )
    elif isinstance(optimizer_cfg.optim, SGDConfig):
        opt = SGD(model.parameters(), optimizer_cfg.optim.lr)

    train_data = np.memmap(train_cfg.train_data_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(train_cfg.val_data_path, dtype=np.uint16, mode="r")

    step = 0
    if ckpt_cfg.resume_from is not None:
        step = load_checkpoint(ckpt_cfg.resume_from, model, opt)

    while step < train_cfg.max_iters:
        lr_t = lr_cosine_schedule(
            step,
            schedule_cfg.max_learning_rate,
            schedule_cfg.min_learning_rate,
            schedule_cfg.warmup_iters,
            schedule_cfg.cosine_cycle_iters,
        )

        for group in opt.param_groups:
            group["lr"] = lr_t

        opt.zero_grad()

        inputs, targets = get_batch(
            train_data, train_cfg.batch_size, model_cfg.context_length, device
        )

        logits = model(inputs)

        train_loss = cross_entropy(logits, targets)

        train_loss.backward()

        gradient_clipping(model.parameters(), train_cfg.max_l2_norm)

        opt.step()

        step += 1
        if step % log_cfg.log_interval == 0:
            logger.info(f"iter {step:07d} | loss: {train_loss.item():.4f}")

        if step % eval_cfg.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                total_loss = 0.0
                for _ in range(eval_cfg.eval_batches):
                    val_inputs, val_targets = get_batch(
                        val_data,
                        train_cfg.batch_size,
                        model_cfg.context_length,
                        device,
                    )
                    total_loss += cross_entropy(model(val_inputs), val_targets).item()
                val_loss = total_loss / eval_cfg.eval_batches
            logger.info(f"iter {step:07d} | eval loss: {val_loss:.4f}")
            model.train()

        if step % ckpt_cfg.interval == 0:
            save_checkpoint(model, opt, step, f"{ckpt_cfg.out_dir}/ckpt_{step:07d}.pt")
    save_checkpoint(model, opt, step, f"{ckpt_cfg.out_dir}/ckpt_{step:07d}_final.pt")


if __name__ == "__main__":

    tyro.cli(main)
