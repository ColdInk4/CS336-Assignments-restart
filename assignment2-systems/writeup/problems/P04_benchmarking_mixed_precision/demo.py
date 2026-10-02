import torch.nn as nn
import torch
import torch.nn.functional as F


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


device = "cuda"
x = torch.rand(20, 100, dtype=torch.float32, device=device)
model = ToyModel(100, 200).to(device)
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    for name, p in model.named_parameters():
        print(name, p.dtype)
    output_fc1 = model.fc1(x)
    print("fc1:", output_fc1.dtype)

    output_relu = model.relu(output_fc1)
    print("relu:", output_relu.dtype)

    output_ln = model.ln(output_relu)
    print("ln:", output_ln.dtype)

    logits = model.fc2(output_ln)
    print("fc2:", logits.dtype)

    loss = F.cross_entropy(logits, torch.zeros(20, dtype=torch.long, device="cuda"))
    print("loss:", loss.dtype)

    loss.backward()

    for name, p in model.named_parameters():
        if p.grad is not None:
            print(name, p.dtype, p.grad.dtype)
