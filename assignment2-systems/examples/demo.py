import triton
import triton.language as tl
import torch
from einops import rearrange


@triton.jit
def weighted_sum_fwd(
    x_ptr,
    weight_ptr,  # 输入指针
    output_ptr,  # 输出指针
    x_stride_row,
    x_stride_dim,  # stride 告诉我们：沿张量的每个轴移动一个元素时，地址要跳多远
    weight_stride_dim,  # 大概是 1
    output_stride_row,  # 大概是 1
    NUM_ROWS,
    D,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,  # tile 形状必须在编译期就确定
):
    # 每个 program instance 负责算 x 里【一块(tile)行】的加权和。
    # `tl.program_id` 能让我们知道当前跑的是哪一个 thread block
    row_tile_idx = tl.program_id(0)
    # block ptr 让你能从多维内存里【切出一块区域】，并把这块选择在内存里移动。
    # 一个 block ptr 必须知道：
    # - 张量首元素的指针（基地址）
    # - 张量整体的 shape，用来判断有没有越界
    # - 每个维度的 stride，用来正确地按内存布局寻址
    # - 这一块的起始坐标，也就是 "offsets"
    # - 每次 load/store 的块大小 block_shape
    # - 各维在内存中从主到次的顺序 order
    #    （= np.argsort(strides)）。这用于优化，
    #    在 Hopper 及以后的 GPU 上是 TMA 的必要条件
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(
            NUM_ROWS,
            D,
        ),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )
    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(NUM_ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )
    # 初始化一个缓冲区，用来累加、最后写出
    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)
    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # 按照当前 block ptr 的描述，把这一块数据取到寄存器里
        # 因为 ROWS_TILE_SIZE 未必整除 NUM_ROWS，且 D_TILE_SIZE 未必整除 D，
        # 所以两个维度都要做越界检查
        row = tl.load(
            x_block_ptr, boundary_check=(0, 1), padding_option="zero"
        )  # (ROWS_TILE_SIZE, D_TILE_SIZE)
        weight = tl.load(
            weight_block_ptr, boundary_check=(0,), padding_option="zero"
        )  # (D_TILE_SIZE,)
        # 计算这一块的加权和
        output += tl.sum(row * weight[None, :], axis=1)
        # 把指针推进到下一块 tile。
        # 参数是 (行, 列) 两个方向上的坐标增量
        x_block_ptr = x_block_ptr.advance(
            (0, D_TILE_SIZE)
        )  # 在最后一个维度（列方向）上前进 D_TILE_SIZE
        weight_block_ptr = weight_block_ptr.advance(
            (D_TILE_SIZE,)
        )  # 前进 D_TILE_SIZE
    # 把结果写回 output block ptr（每行一个标量）
    # 因为 ROWS_TILE_SIZE 未必整除 NUM_ROWS，需要做越界检查
    tl.store(output_block_ptr, output, boundary_check=(0,))


@triton.jit
def weighted_sum_backward(
    x_ptr,
    weight_ptr,  # Input
    grad_output_ptr,  # 梯度输入
    grad_x_ptr,
    partial_grad_weight_ptr,  # 梯度输出
    stride_xr,
    stride_xd,
    stride_wd,
    stride_gr,
    stride_gxr,
    stride_gxd,
    stride_gwb,
    stride_gwd,
    NUM_ROWS,
    D,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)
    # 输入的 block ptr
    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(NUM_ROWS,),
        strides=(stride_gr,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(stride_wd,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )
    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(n_row_tiles, D),
        strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0),
    )
    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        grad_output = tl.load(
            grad_output_block_ptr, boundary_check=(0,), padding_option="zero"
        )  # (ROWS_TILE_SIZE,)
        # grad_x 是【外积】：每一行都等于 grad_output 乘以整个 weight
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")  # (D_TILE_SIZE,)
        grad_x_row = grad_output[:, None] * weight[None, :]
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))
        # 把行方向能归约的先归约掉，得到这一块对 grad_weight 的贡献
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (ROWS_TILE_SIZE, D_TILE_SIZE)
        grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)
        tl.store(
            partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1,)
        )  # 第 0 维永远不会越界：0
        # 沿 D 方向，把指针推进到下一块 tile
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))


class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # 把 x 和 weight 缓存起来，留给 backward 用。
        # 因为 backward 时我们只会收到"对输出张量的梯度"，
        # 还需要自己算出"对 x 和 weight 的梯度"。
        D, output_dims = x.shape[-1], x.shape[:-1]
        # 把输入张量 reshape 成 2D（kernel 只认 2D）
        input_shape = x.shape
        x = rearrange(x, "... d -> (...) d")
        ctx.save_for_backward(x, weight)
        assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"
        ctx.D_TILE_SIZE = (
            triton.next_power_of_2(D) // 16
        )  # 大致让 embedding 维循环 16 轮
        ctx.ROWS_TILE_SIZE = 16  # 每个 thread block 一次处理 16 个 batch 元素
        ctx.input_shape = input_shape
        # 注意这里必须是 empty 而不是 zeros：这些元素会被 kernel 完全写满，
        # 它们的初值不一定是 0（初始化成 0 只是白花钱）
        y = torch.empty(output_dims, device=x.device)
        # 用一个 1D grid 启动 kernel，grid 的大小 = 要跑多少个 instance
        n_rows = y.numel()
        weighted_sum_fwd[(triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x,
            weight,
            y,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            y.stride(0),
            NUM_ROWS=n_rows,
            D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE,
            D_TILE_SIZE=ctx.D_TILE_SIZE,
        )
        return y.view(input_shape[:-1])

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE  # 这里用的 tile size可以和 forward 里不一样
        n_rows, D = x.shape
        # 我们的策略：让每个 thread block 先把结果写进一个"局部(partial)"缓冲区，
        # 然后在 kernel 外面再对这张表求和归约，得到最终梯度。
        partial_grad_weight = torch.empty(
            (triton.cdiv(n_rows, ROWS_TILE_SIZE), D), device=x.device, dtype=x.dtype
        )
        grad_x = torch.empty_like(x)
        weighted_sum_backward[(triton.cdiv(n_rows, ROWS_TILE_SIZE),)](
            x,
            weight,
            grad_out,
            grad_x,
            partial_grad_weight,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            grad_out.stride(0),
            grad_x.stride(0),
            grad_x.stride(1),
            partial_grad_weight.stride(0),
            partial_grad_weight.stride(1),
            NUM_ROWS=n_rows,
            D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE,
            D_TILE_SIZE=D_TILE_SIZE,
        )
        grad_weight = partial_grad_weight.sum(axis=0)
        return grad_x, grad_weight


# 用户是这样调用它的，用法和 torch.nn.functional 里的算子一样
f_weightedsum = WeightedSumFunc.apply
