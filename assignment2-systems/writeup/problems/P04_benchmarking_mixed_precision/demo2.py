# test_F_layer_norm.py —— 用 F.layer_norm 实测 (b) 的每个论断
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
N = 4096
EPS = 1e-5
device = "cuda" if torch.cuda.is_available() else "cpu"
z = torch.randn(N, device=device)

# ---- T1 主证据：输入表示塌了，kernel 内部再稳也救不回来 ----
print("[T1] 输入表示是主因")


def t1(mu, sigma, dtype):
    x32 = mu + sigma * z
    xd = x32.to(dtype)
    y_d = F.layer_norm(xd, (N,)).float()  # 低精度输入直接跑 F.layer_norm
    y_32 = F.layer_norm(x32, (N,))
    print(
        f"   mu={mu:<5g} sigma={sigma:<6g} {str(dtype):>15} | "
        f"levels(x)={torch.unique(xd).numel():<3d} "
        f"levels(y)={torch.unique(y_d).numel():<3d} "
        f"std(x_d)/sigma={xd.float().std().item()/sigma:6.3f} | "
        f"std(y)={y_d.std().item():.4f} (fp32 ref {y_32.std().item():.4f})"
    )


for dt in (torch.float16, torch.bfloat16):
    for mu, sigma in [(50.0, 0.01), (50.0, 0.2), (50.0, 2.0), (0.0, 0.01)]:
        t1(mu, sigma, dt)

# ---- T2a kernel 内部是不是低精度累加 ----
print("\n[T2a] 同一低精度输入：kernel vs 量化后再用 fp32 算 vs naive 一阶式")
mu, sigma = 2.0, 0.02  # fp16 ulp(2.0)=1.95e-3，sigma/ulp≈10，输入表示没问题
x32 = mu + sigma * z
x16 = x32.to(torch.float16)
y_kernel = F.layer_norm(x16, (N,)).float()
y_qref = F.layer_norm(x16.float(), (N,))  # 同样的输入，纯 fp32 计算
y_ref = F.layer_norm(x32, (N,))
var_naive_f16 = (x16 * x16).mean() - x16.mean() ** 2  # 全 fp16
var_naive_f32 = (x16.float() ** 2).mean() - x16.float().mean() ** 2
var_true = x16.float().var(unbiased=False)
print(
    f"   var: true={var_true.item():.6f}  naive(fp16)={var_naive_f16.float().item():.6f}  "
    f"naive(fp32)={var_naive_f32.item():.6f}"
)
print(
    f"   max|kernel-量化后fp32|={(y_kernel - y_qref).abs().max().item():.3e}  "
    f"max|kernel-fp32ref|={(y_kernel - y_ref).abs().max().item():.3e}"
)

# ---- T2b 决定性实验：sigma/mu=2e-4，fp32 下 naive 公式也塌，看 kernel 跟谁一致 ----
print("\n[T2b] fp32 输入（表示无问题）：kernel 是两遍式还是 E[x^2]-mu^2？")
mu, sigma = 50.0, 0.01
x = mu + sigma * z  # fp32，ulp(50)=3.8e-6 ≪ sigma
naive_var = (x * x).mean() - x.mean() ** 2
tp_var = ((x - x.mean()) ** 2).mean()
y_kernel = F.layer_norm(x, (N,))
y_tp = (x - x.mean()) * torch.rsqrt(tp_var + EPS)
y_naive = (x - x.mean()) * torch.rsqrt(naive_var.clamp_min(0.0) + EPS)
print(
    f"   naive_var={naive_var.item():.6e}  two_pass_var={tp_var.item():.6e}  "
    f"naive/true={naive_var.item() / tp_var.item():.2f}"
)
print(
    f"   max|kernel-two_pass|={(y_kernel - y_tp).abs().max().item():.3e}  "
    f"max|kernel-naive|={(y_kernel - y_naive).abs().max().item():.3e}"
)

# ---- T3 eps 与 var 的表示边缘（顺带证明 4.7% 与精度无关）----
print("\n[T3] eps / var 表示边缘")


def smallest_subnormal(dt):
    # torch.finfo 在部分版本没有 smallest_subnormal / nmant 属性。
    # tiny = 最小正规数 (2^(1-emax))，eps = 1.0 处的 ulp (2^(-尾数位数))。
    # 最小次正规数 = tiny * eps，例如 fp16: 2^-14 * 2^-10 = 2^-24 ≈ 5.96e-8。
    f = torch.finfo(dt)
    return getattr(f, "smallest_subnormal", f.tiny * f.eps)


for n, dt in [
    ("fp16", torch.float16),
    ("bf16", torch.bfloat16),
    ("fp32", torch.float32),
]:
    f = torch.finfo(dt)
    print(
        f"   {n}: smallest_normal={f.smallest_normal:.3e}  "
        f"smallest_subnormal={smallest_subnormal(dt):.3e}"
    )
print(
    f"   1e-4 / smallest_normal(fp16) = {1e-4 / torch.finfo(torch.float16).smallest_normal:.2f}   <- 贴着表示边缘"
)
# 注意：torch.tensor 的 dtype 必须用关键字传（部分版本不接受第二个位置参数）
print(
    f"   fp16(1e-5)={torch.tensor(1e-5, dtype=torch.float16).item():.3e}   "
    f"fp16(1e-8)={torch.tensor(1e-8, dtype=torch.float16).item():.3e}"
)
print(
    f"   var=1e-8 若被舍成 0：rstd 钳到 1/sqrt(eps)={1 / (1e-5 ** 0.5):.1f}，真值 1/(1e-8)^0.5={1e4:.0f}"
)
v = torch.tensor(1e-4)
print(
    f"   4.7% 偏移在 fp32 下同样存在：带 eps rstd={torch.rsqrt(v + 1e-5).item():.2f} vs "
    f"无 eps={torch.rsqrt(v).item():.2f} -> {torch.rsqrt(v).item() / torch.rsqrt(v + 1e-5).item() - 1:.1%}"
)

# ---- T4 autocast dtype（(a) + layer_norm 输出 fp32 的实测）----
print("\n[T4] autocast dtypes on", device)


class ToyModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.ln(self.relu(self.fc1(x))))


model = ToyModel(64, 10).to(device)
x = torch.randn(8, 64, device=device)
tgt = torch.randint(0, 10, (8,), device=device)
amps = (torch.float16, torch.bfloat16) if device == "cuda" else (torch.bfloat16,)
for amp in amps:
    with torch.autocast(device, dtype=amp):
        h = model.relu(model.fc1(x))
        l = model.ln(h)
        logits = model.fc2(l)
        loss = F.cross_entropy(logits, tgt)
        y_raw = F.layer_norm(x, (64,))
    model.zero_grad(set_to_none=True)
    loss.backward()
    print(
        f"   autocast {str(amp):>15}: param={model.fc1.weight.dtype} fc1.out={h.dtype} "
        f"ln.out={l.dtype} logits={logits.dtype} loss={loss.dtype} "
        f"grad={model.fc1.weight.grad.dtype} F.layer_norm.out={y_raw.dtype}"
    )
print(
    "   对照（无 autocast）：F.layer_norm(x.bfloat16()) 输出 =",
    F.layer_norm(x.to(torch.bfloat16), (64,)).dtype,
)
