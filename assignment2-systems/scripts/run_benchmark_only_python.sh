CUDA_VISIBLE_DEVICES=5 uv run \
python scripts/benchmark/run_benchmark.py \
--sizes large \
--iters 1 \
--ctx-lens 128 1024 \
--modes fwd fwdbwdopt \
--mem-profile \
--mixed-dtype bfloat16
