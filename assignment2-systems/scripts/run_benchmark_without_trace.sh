CUDA_VISIBLE_DEVICES=5 uv run nsys profile \
-o "reports/small-ctx(256_512_2048)_fwd" \
--trace=cuda,cudnn,cublas,osrt,nvtx,cublas-verbose \
--nvtx-capture="one_step" \
-- python scripts/benchmark/run_benchmark.py \
--sizes small \
--iters 3 \
--ctx-lens 256 512 2048 \
--modes fwd \