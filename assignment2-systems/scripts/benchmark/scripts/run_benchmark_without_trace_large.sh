CUDA_VISIBLE_DEVICES=5 uv run nsys profile \
-o "reports/large-ctx(256_512_1024)_fwd_softmax" \
--trace=cuda,cudnn,cublas,osrt,nvtx,cublas-verbose \
--nvtx-capture="one_step" \
-- python scripts/benchmark/run_benchmark.py \
--sizes large \
--iters 3 \
--ctx-lens 256 512 1024 \
--modes fwd