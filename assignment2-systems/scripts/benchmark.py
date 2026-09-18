from dataclasses import dataclass
import torch
from typing import Literal, Tuple
import numpy as np
import random
import timeit
import pandas as pd

from cs336_basics.transformer import TransformerLM
from cs336_basics.training import AdamW, cross_entropy

MODEL_PRESETS = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B": dict(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}


@dataclass
class ModelConfig:
    vocab_size: int = 10000
    context_length: int = 512
    rope_theta: float = 10000
    device: Literal["cpu", "cuda"] = "cuda"
    dtype: Literal["float32"] = "float32"


@dataclass
class AdamWConfig:
    lr: float = 3e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01


@dataclass
class TrainingConfig:
    batch_size: int = 4
    seed: int = 42


@dataclass
class BenchmarkConfig:
    warm_up_steps: int = 5
    execution_steps: int = 20
    mode: Literal["fwd", "fwdbwd", "fwdbwdopt"] = "fwdbwdopt"
    sizes: Tuple[str, ...] = ("small", "medium", "large", "xl", "10B")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def benchmark_one(
    size_name: str,
    model_cfg: ModelConfig,
    opt_cfg: AdamWConfig,
    train_cfg: TrainingConfig,
    bench_cfg: BenchmarkConfig,
):
    set_seed(train_cfg.seed)

    device = torch.device(model_cfg.device)
    dtype = getattr(torch, model_cfg.dtype)
    preset = MODEL_PRESETS[size_name]

    model = TransformerLM(
        model_cfg.vocab_size,
        model_cfg.context_length,
        preset["d_model"],
        preset["num_layers"],
        preset["num_heads"],
        preset["d_ff"],
        model_cfg.rope_theta,
        device,
        dtype,
    )
    opt = AdamW(
        model.parameters(),
        opt_cfg.lr,
        (opt_cfg.beta1, opt_cfg.beta2),
        opt_cfg.eps,
        opt_cfg.weight_decay,
    )

    inputs = torch.randint(
        0,
        model_cfg.vocab_size,
        (train_cfg.batch_size, model_cfg.context_length),
        device=device,
        dtype=torch.long,
    )
    targets = torch.randint(
        0,
        model_cfg.vocab_size,
        (train_cfg.batch_size, model_cfg.context_length),
        device=device,
        dtype=torch.long,
    )

    def one_step():
        logits = model(inputs)
        if bench_cfg.mode in ("fwdbwd", "fwdbwdopt"):
            loss = cross_entropy(logits, targets)
            loss.backward()
        if bench_cfg.mode == "fwdbwdopt":
            opt.step()
            opt.zero_grad()

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(bench_cfg.warm_up_steps):
        one_step()
    sync()

    times = []
    for _ in range(bench_cfg.execution_steps):
        sync()
        start = timeit.default_timer()
        one_step()
        sync()
        end = timeit.default_timer()
        times.append(end - start)

    # 返回平均值 + 标准差，以及样本数
    return float(np.average(times)), float(np.std(times)), len(times)


def benchmark(
    model_cfg: ModelConfig,
    opt_cfg: AdamWConfig,
    train_cfg: TrainingConfig,
    bench_cfg: BenchmarkConfig,
):
    rows = []
    for size_name in bench_cfg.sizes:
        print(
            f"\n=== {size_name} | mode={bench_cfg.mode} | dtype={model_cfg.dtype} ==="
        )
        status = "ok"
        mean_t = std_t = float("nan")
        n = 0
        err = ""

        try:
            mean_t, std_t, n = benchmark_one(
                size_name, model_cfg, opt_cfg, train_cfg, bench_cfg
            )
            print(f"  mean = {mean_t*1e3:8.2f} ms | std = {std_t*1e3:7.2f} ms | n={n}")
        except torch.cuda.OutOfMemoryError as e:
            status = "OOM"
            err = str(e).splitlines()[0][:80]
            print(f"  OOM: {err}")
        except RuntimeError as e:
            # dtype / 算子不支持、cudnn 报错之类
            status = "RuntimeError"
            err = str(e).splitlines()[0][:80]
            print(f"  RuntimeError: {err}")
        except Exception as e:
            status = "Failed"
            err = f"{type(e).__name__}: {str(e).splitlines()[0][:80]}"
            print(f"  Failed: {err}")
        finally:
            # 关键：无论成功失败都清一下显存，避免影响后续 size
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

        rows.append(
            {
                "size": size_name,
                "status": status,
                "ave_time_ms": mean_t * 1e3,
                "std_ms": std_t * 1e3,
                "n": n,
                "error": err,
            }
        )

    df = pd.DataFrame(rows).set_index("size")
    pd.set_option("display.width", 160)
    print("\n================ Summary ================")
    print(df.to_string(float_format=lambda x: f"{x:.2f}"))
    return df


def collect_long(
    model_cfg, opt_cfg, train_cfg, sizes, modes, warm_up_steps, execution_steps
):
    """跑所有 (mode, size) 组合，返回长表。"""
    rows = []
    for mode in modes:
        print(f"\n########## mode = {mode} ##########")
        bench_cfg = BenchmarkConfig(
            warm_up_steps=warm_up_steps,
            execution_steps=execution_steps,
            mode=mode,
            sizes=sizes,
        )
        df = benchmark(model_cfg, opt_cfg, train_cfg, bench_cfg)
        for size_name, r in df.iterrows():
            rows.append(
                {
                    "size": size_name,
                    "mode": mode,
                    "status": r["status"],
                    "mean_ms": r["ave_time_ms"],
                    "std_ms": r["std_ms"],
                    "n": r["n"],
                }
            )
    return pd.DataFrame(rows)


def to_wide(long_df: pd.DataFrame, modes, sizes) -> pd.DataFrame:
    """pivot 成 size × mode，每个格子是 'mean ± std' 或失败状态。"""
    df = long_df.copy()
    df["time"] = df.apply(
        lambda r: (
            f"{r['mean_ms']:.2f} ± {r['std_ms']:.2f}"
            if r["status"] == "ok"
            else r["status"]
        ),
        axis=1,
    )
    wide = df.pivot(index="size", columns="mode", values="time")
    wide = wide.reindex(columns=[m for m in modes if m in wide.columns])
    wide = wide.reindex(index=[s for s in sizes if s in wide.index])
    return wide


if __name__ == "__main__":
    model_cfg = ModelConfig(
        vocab_size=10000, context_length=512, device="cuda", dtype="float32"
    )
    opt_cfg = AdamWConfig()
    train_cfg = TrainingConfig(batch_size=4, seed=42)

    sizes = ("small", "medium", "large", "xl", "10B")
    modes = ("fwd", "fwdbwd", "fwdbwdopt")

    long_df = collect_long(
        model_cfg,
        opt_cfg,
        train_cfg,
        sizes=sizes,
        modes=modes,
        warm_up_steps=5,
        execution_steps=10,
    )

    wide = to_wide(long_df, modes, sizes)
    print("\n================ Summary ================")
    print(wide.to_string())

    print("\n================ Typst ================")
    print(wide.style.to_typst())
