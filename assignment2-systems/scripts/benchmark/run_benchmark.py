# run_benchmark.py

import torch
import pandas as pd
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
            f"\n=== {size_name} | mode={bench_cfg.mode} | dtype={model_cfg.dtype} ==="
        )
        try:
            r = measure_size(size_name, model_cfg, opt_cfg, train_cfg, bench_cfg)
        except torch.cuda.OutOfMemoryError as e:
            r = BenchmarkResult.failed(size_name, bench_cfg.mode, "OOM", e)
        except RuntimeError as e:
            r = BenchmarkResult.failed(size_name, bench_cfg.mode, "RuntimeError", e)
        except Exception as e:
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


def run_sweep(model_cfg, opt_cfg, train_cfg, sizes, modes, warmup, iters):
    rows = []
    for mode in modes:
        print(f"\n########## mode = {mode} ##########")
        cfg = BenchmarkConfig(
            warm_up_steps=warmup, execution_steps=iters, mode=mode, sizes=sizes
        )
        rows.extend(
            vars(r) for r in measure_all_sizes(model_cfg, opt_cfg, train_cfg, cfg)
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    model_cfg = ModelConfig(
        vocab_size=10000, context_length=512, device="cuda", dtype="float32"
    )
    opt_cfg = AdamWConfig()
    train_cfg = TrainingConfig(batch_size=4, seed=42)

    sizes = ("small",)
    # sizes = ("small", "medium", "large", "xl", "10B")
    modes = ("fwd", "fwdbwd", "fwdbwdopt")

    long_df = run_sweep(
        model_cfg,
        opt_cfg,
        train_cfg,
        sizes,
        modes,
        warmup=5,
        iters=10,
    )

    wide = results_to_wide(long_df, modes, sizes)
    print("\n================ Summary ================")
    print(wide.to_string())

    print("\n================ Typst ================")
    print(wide.style.to_typst())
