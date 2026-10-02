#import "../../template.typ": solution, deliverable

#solution(
  "mixed_precision_accumulation",
  "Mixed-Precision Accumulation",
  "1 point",
  [
    Run the following code and comment on the accuracy of the results.

    #raw("s = torch.tensor(0,dtype=torch.float32)\nfor i in range(1000):\n    s += torch.tensor(0.01,dtype=torch.float32)\nprint(s)\n\ns = torch.tensor(0,dtype=torch.float16)\nfor i in range(1000):\n    s += torch.tensor(0.01,dtype=torch.float16)\nprint(s)\n\ns = torch.tensor(0,dtype=torch.float32)\nfor i in range(1000):\n    s += torch.tensor(0.01,dtype=torch.float16)\nprint(s)\n\ns = torch.tensor(0,dtype=torch.float32)\nfor i in range(1000):\n    x = torch.tensor(0.01,dtype=torch.float16)\n    s += x.type(torch.float32)\nprint(s)", block: true, lang: "python")

    #deliverable[A 2-3 sentence response.]
  ],
  answer: [
    运行结果为
    ```
tensor(10.0001)
tensor(9.9531, dtype=torch.float16)
tensor(10.0021)
tensor(10.0021)
```
  四个结果分别为 10.0001、9.9531、10.0021、10.0021，其中累加器为 fp16 的第二种情况误差最大，因为随着运行和增大，fp16 的表示粒度变粗（10 附近 ulp 约 0.008），每次 0.01 的增量几乎被舍入吞掉。而累加器为 fp32 的三种情况都很准确，剩余的微小偏差来自被加数自身的表示误差（fp16 存的 0.01 实际是 0.0100021，故偏向 10.0021），与累加过程无关。案例 3 与 4 结果完全相同，说明 PyTorch 会自动把 fp16 加数转成 fp32 后再加，因此即使张量被降精度存储，也应始终用更高精度累加——这几乎没有成本，却能保留输入所能提供的全部精度。
  ],
)