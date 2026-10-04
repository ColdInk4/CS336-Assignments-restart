from layers.scaled_dot_product_attention import scaled_dot_product_attention
import torch
import timeit
import statistics

batch_size = 8
d_models = [16, 32, 64, 128]
context_lengths = [256, 1024, 4096, 8192, 16384, 32768]

device = torch.device("cuda")


def forward(Q, K, V):
    return scaled_dot_product_attention(Q, K, V)


for d_model in d_models:
    for context_length in context_lengths:
        Q = K = V = gradient = y = None
        try:
            Q = torch.randn(
                (batch_size, context_length, d_model), requires_grad=True, device=device
            )
            K = torch.randn(
                (batch_size, context_length, d_model), requires_grad=True, device=device
            )
            V = torch.randn(
                (batch_size, context_length, d_model), requires_grad=True, device=device
            )
            gradient = torch.randn(
                (batch_size, context_length, d_model),
                requires_grad=False,
                device=device,
            )
            for _ in range(5):
                forward(Q, K, V).backward(gradient=gradient)
            torch.cuda.synchronize()
            forward_time = []
            backward_time = []
            memory_used = []
            for _ in range(100):

                start = timeit.default_timer()
                y = forward(Q, K, V)
                torch.cuda.synchronize()
                forward_time.append(timeit.default_timer() - start)

                memory_used.append(torch.cuda.memory_allocated())

                start = timeit.default_timer()
                y.backward(gradient=gradient)
                torch.cuda.synchronize()
                backward_time.append(timeit.default_timer() - start)
        except torch.OutOfMemoryError:
            print(
                f"batch size({batch_size}), context length({context_length:>5}), d_model({d_model:>3}): OOM"
            )
        else:
            print(
                f"batch size({batch_size}), context length({context_length:>5}), d_model({d_model:>3}), forward time: {statistics.mean(forward_time)*1000:>7.4f}ms"
            )
            print(
                f"batch size({batch_size}), context length({context_length:>5}), d_model({d_model:>3}), backward time: {statistics.mean(backward_time)*1000:>7.4f}ms"
            )
            print(
                f"Total memory in used: {statistics.mean(memory_used)/ (1024**2):.2f} MiB"
            )
        finally:
            del Q
            del K
            del V
            del gradient
            del y
            torch.cuda.empty_cache()
