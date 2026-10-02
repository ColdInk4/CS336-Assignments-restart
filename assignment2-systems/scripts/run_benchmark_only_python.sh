CUDA_VISIBLE_DEVICES=5 uv run \
python scripts/benchmark/run_benchmark.py \
--sizes small medium large xl 10B \
--iters 10 \
--modes fwd fwdbwd \
--dtype float32