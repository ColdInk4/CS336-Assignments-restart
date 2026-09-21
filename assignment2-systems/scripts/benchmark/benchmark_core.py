# benchmark_core.py

import torch
import timeit
import torch.cuda.nvtx as nvtx
from utils.configs import (
    ModelConfig,
    AdamWConfig,
    TrainingConfig,
    BenchmarkConfig,
    MODEL_PRESETS,
    set_seed,
)
from utils.results import BenchmarkResult
import numpy as np

from cs336_basics.transformer import TransformerLM
from cs336_basics.training import AdamW, cross_entropy


def _build_model(size_name: str, model_cfg: ModelConfig):
    device = torch.device(model_cfg.device)
    preset = MODEL_PRESETS[size_name]
    dtype = getattr(torch, model_cfg.dtype)
    model = TransformerLM(
        model_cfg.vocab_size,
        model_cfg.context_length,
        preset.d_model,
        preset.num_layers,
        preset.num_heads,
        preset.d_ff,
        model_cfg.rope_theta,
        device,
        dtype,
    )
    return model


def _build_opt(model: TransformerLM, opt_cfg: AdamWConfig):
    return AdamW(
        model.parameters(),
        opt_cfg.lr,
        (opt_cfg.beta1, opt_cfg.beta2),
        opt_cfg.eps,
        opt_cfg.weight_decay,
    )


def _make_batch(model_cfg, train_cfg):
    device = torch.device(model_cfg.device)
    shape = (train_cfg.batch_size, model_cfg.context_length)
    inputs = torch.randint(
        0,
        model_cfg.vocab_size,
        shape,
        device=device,
        dtype=torch.long,
    )
    targets = torch.randint(
        0,
        model_cfg.vocab_size,
        shape,
        device=device,
        dtype=torch.long,
    )
    return inputs, targets


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_step(model, opt, mode, inputs, targets, device) -> None:
    with nvtx.range("forward"):
        logits = model(inputs)
        _sync(device)

    if mode in ("fwdbwd", "fwdbwdopt"):
        with nvtx.range("backward"):
            loss = cross_entropy(logits, targets)
            loss.backward()
            _sync(device)
    if mode == "fwdbwdopt":
        with nvtx.range("optmizer"):
            opt.step()
            opt.zero_grad()
            _sync(device)


def measure_size(
    size_name: str,
    model_cfg: ModelConfig,
    opt_cfg: AdamWConfig,
    train_cfg: TrainingConfig,
    bench_cfg: BenchmarkConfig,
) -> BenchmarkResult:
    set_seed(train_cfg.seed)
    device = torch.device(model_cfg.device)
    model = _build_model(size_name, model_cfg)
    opt = _build_opt(model, opt_cfg)

    inputs, targets = _make_batch(model_cfg, train_cfg)

    with nvtx.range("warm_up"):
        for _ in range(bench_cfg.warm_up_steps):
            _run_step(model, opt, bench_cfg.mode, inputs, targets, device)
    model.zero_grad(set_to_none=True)
    _sync(device)

    times = []
    for _ in range(bench_cfg.execution_steps):
        _sync(device)
        start = timeit.default_timer()
        with nvtx.range("one_step"):
            _run_step(model, opt, bench_cfg.mode, inputs, targets, device)
        _sync(device)
        end = timeit.default_timer()
        times.append(end - start)

    times = np.asarray(times)

    return BenchmarkResult(
        size=size_name,
        mode=bench_cfg.mode,
        mean_ms=times.mean() * 1e3,
        std_ms=times.std() * 1e3,
        n=len(times),
    )
