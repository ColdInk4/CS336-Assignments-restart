CUDA_VISIBLE_DEVICES=5 uv run nsys profile \
-o reports/test_%n \
--trace=cuda,cudnn,cublas,osrt,nvtx,cublas-verbose \
--pytorch=functions-trace,autograd-shapes-nvtx \
--nvtx-capture="benchmark" \
--cudabacktrace=all \
--python-backtrace=cuda \
-- python scripts/benchmark/run_benchmark.py