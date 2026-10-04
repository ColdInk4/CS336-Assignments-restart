CUDA_VISIBLE_DEVICES=2 uv run \
python scripts/benchmark/run_benchmark.py \
--sizes small medium large xl 10B \
--iters 10 \
--ctx-lens 512 \
--modes fwd fwdbwdopt
