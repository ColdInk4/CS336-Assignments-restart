CUDA_VISIBLE_DEVICES=2 uv run nsys profile \
-o reports/test_%n \
--trace=cuda,cudnn,cublas,osrt,nvtx,cublas-verbose \
--pytorch=functions-trace,autograd-shapes-nvtx \
--nvtx-capture="benchmark" \
-- python scripts/benchmark/run_benchmark.py \
--sizes small \
--ctx-lens 4096 \