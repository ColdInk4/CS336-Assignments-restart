# run_benchmark.py

import argparse
import torch
import pandas as pd
import traceback

from benchmark_core import measure_size
from utils.configs import (
    ModelConfig,
    AdamWConfig,
    TrainingConfig,
    BenchmarkConfig,
)
from utils.results import BenchmarkResult
from utils.report import results_to_wide


def measure_all_sizes(model_cfg, opt_cfg, train_cfg, bench_cfg):
    results = []
    for size_name in bench_cfg.sizes:
        print(
            f"\n=== {size_name} | ctx={model_cfg.context_length} "
            f"| mode={bench_cfg.mode} | dtype={model_cfg.dtype} ==="
        )
        try:
            r = measure_size(size_name, model_cfg, opt_cfg, train_cfg, bench_cfg)
        except torch.cuda.OutOfMemoryError as e:
            r = BenchmarkResult.failed(size_name, bench_cfg.mode, "OOM", e)
        except RuntimeError as e:
            r = BenchmarkResult.failed(size_name, bench_cfg.mode, "RuntimeError", e)
        except Exception as e:
            traceback.print_exc()  # 完整打印堆栈
            r = BenchmarkResult.failed(size_name, bench_cfg.mode, type(e).__name__, e)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

        if r.status == "ok":
            print(f"  mean={r.mean_ms:8.2f} ms | std={r.std_ms:7.2f} ms | n={r.n}")
        else:
            print(f"  {r.status}: {r.error}")
        results.append(r)
    return results


def run_sweep(
    opt_cfg,
    train_cfg,
    vocab_size,
    ctx_lens,
    sizes,
    modes,
    warmup,
    iters,
    device,
    dtype,
):
    """扫 (ctx_len, size, mode) 所有组合，返回长表。"""
    rows = []
    for ctx_len in ctx_lens:
        model_cfg = ModelConfig(
            vocab_size=vocab_size,
            context_length=ctx_len,
            device=device,
            dtype=dtype,
        )
        for mode in modes:
            print(f"\n########## ctx={ctx_len} | mode={mode} ##########")
            cfg = BenchmarkConfig(
                warm_up_steps=warmup,
                execution_steps=iters,
                mode=mode,
                sizes=sizes,
            )
            for r in measure_all_sizes(model_cfg, opt_cfg, train_cfg, cfg):
                row = vars(r)
                row["ctx_len"] = ctx_len
                rows.append(row)
    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser(description="Transformer benchmark sweep")
    p.add_argument(
        "--sizes",
        nargs="+",
        default=["small"],
        help="model size presets, e.g. small medium large xl 10B",
    )
    p.add_argument(
        "--ctx-lens",
        nargs="+",
        type=int,
        default=[512],
        help="context lengths, e.g. 512 1024 2048",
    )
    p.add_argument(
        "--modes",
        nargs="+",
        choices=["fwd", "fwdbwd", "fwdbwdopt"],
        default=["fwd", "fwdbwd", "fwdbwdopt"],
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--dtype", default="float32")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    sizes = tuple(args.sizes)
    modes = tuple(args.modes)
    ctx_lens = tuple(args.ctx_lens)

    opt_cfg = AdamWConfig()
    train_cfg = TrainingConfig(batch_size=args.batch_size, seed=42)

    long_df = run_sweep(
        opt_cfg=opt_cfg,
        train_cfg=train_cfg,
        vocab_size=args.vocab_size,
        ctx_lens=ctx_lens,
        sizes=sizes,
        modes=modes,
        warmup=args.warmup,
        iters=args.iters,
        device=args.device,
        dtype=args.dtype,
    )

    for ctx_len in ctx_lens:
        sub = long_df[long_df["ctx_len"] == ctx_len]
        if sub.empty:
            continue
        wide = results_to_wide(sub, modes, sizes)
        print(f"\n================ ctx_len = {ctx_len} ================")
        print(wide.to_string())
        print("\n-- Typst --")
        print(wide.style.to_typst())
