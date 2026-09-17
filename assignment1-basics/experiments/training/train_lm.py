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
from dataclasses import dataclass, asdict
import torch
import tyro
from typing import Literal, Union
import numpy as np
from loguru import logger
from pathlib import Path
import random
import wandb


@dataclass
class ModelConfig:
    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 512
    d_ff: int = 1344
    rope_theta: float = 10000
    num_layers: int = 4
    num_heads: int = 16
    device: Literal["cpu", "cuda"] = "cuda"
    dtype: Literal["float32"] = "float32"


@dataclass
class AdamWConfig:
    lr: float
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01


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
    max_steps: int
    batch_size: int
    max_l2_norm: float
    seed: int = 42


@dataclass
class LogConfig:
    log_interval: int = 10


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


@dataclass
class WandbConfig:
    project: str = "CS336-new-assignment1"
    entity: str | None = None  # None 时让 wandb 用环境变量
    run_name: str | None = None


def train(
    model_cfg: ModelConfig,
    optimizer_cfg: OptimizerConfig,
    train_cfg: TrainingConfig,
    ckpt_cfg: CheckpointConfig,
    schedule_cfg: ScheduleConfig,
    log_cfg: LogConfig,
    eval_cfg: EvalConfig,
    wandb_cfg: WandbConfig,
):
    # 设置 seed
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    random.seed(train_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_cfg.seed)

    # 初始化
    out_dir = Path(ckpt_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    optim_dict = asdict(optimizer_cfg.optim)
    optim_dict["kind"] = (
        type(optimizer_cfg.optim).__name__.replace("Config", "").lower()
    )

    run = wandb.init(  # Set the wandb entity where your project will be logged (generally your team name).
        entity=wandb_cfg.entity,
        # Set the wandb project where this run will be logged.
        project=wandb_cfg.project,
        name=wandb_cfg.run_name,
        # Track hyperparameters and run metadata.
        config={
            "model": asdict(model_cfg),
            "optimizer": optim_dict,
            "training": asdict(train_cfg),
            "schedule": asdict(schedule_cfg),
            "checkpoint": asdict(ckpt_cfg),
            "eval": asdict(eval_cfg),
            "log": asdict(log_cfg),
        },
    )

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

    try:
        while step < train_cfg.max_steps:
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
                run.log({"train/loss": train_loss.item(), "lr": lr_t}, step=step)

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
                        total_loss += cross_entropy(
                            model(val_inputs), val_targets
                        ).item()
                    val_loss = total_loss / eval_cfg.eval_batches
                logger.info(f"iter {step:07d} | eval loss: {val_loss:.4f}")
                run.log({"val/loss": val_loss}, step=step)
                model.train()

            if step % ckpt_cfg.interval == 0:
                save_checkpoint(model, opt, step, out_dir / f"ckpt_{step:07d}.pt")
    finally:
        save_checkpoint(model, opt, step, out_dir / f"ckpt_{step:07d}_final.pt")
        run.finish()


if __name__ == "__main__":

    tyro.cli(train)
