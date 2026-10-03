CUDA_VISIBLE_DEVICES=2 \
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
uv run nsys profile \
-o "reports/large-ctx(1024)_fwdbwd_nocache" \
-c nvtx \
--trace=cuda,cudnn,cublas,osrt,nvtx,cublas-verbose \
--pytorch=functions-trace,autograd-shapes-nvtx \
--cuda-memory-usage=true \
--nvtx-capture="one_step" \
-- python scripts/benchmark/run_benchmark.py \
--sizes large \
--iters 1 \
--ctx-lens 1024 \
--modes fwdbwd