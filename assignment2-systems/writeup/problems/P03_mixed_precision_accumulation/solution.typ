#import "../../template.typ": solution, deliverable

#solution(
  "mixed_precision_accumulation",
  "Mixed-Precision Accumulation",
  "1 point",
  [
    Run the following code and comment on the accuracy of the results.

    #raw("s = torch.tensor(0,dtype=torch.float32)\nfor i in range(1000):\n    s += torch.tensor(0.01,dtype=torch.float32)\nprint(s)\n\ns = torch.tensor(0,dtype=torch.float16)\nfor i in range(1000):\n    s += torch.tensor(0.01,dtype=torch.float16)\nprint(s)\n\ns = torch.tensor(0,dtype=torch.float32)\nfor i in range(1000):\n    x = torch.tensor(0.01,dtype=torch.float16)\n    s += x.type(torch.float32)\nprint(s)", block: true, lang: "python")

    #deliverable[A 2-3 sentence response.]
  ],
  answer: [
  ],
)