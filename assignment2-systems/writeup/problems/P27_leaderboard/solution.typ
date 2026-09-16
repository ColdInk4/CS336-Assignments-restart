#import "../../template.typ": solution, deliverable

#solution(
  "leaderboard",
  "Leaderboard: fastest training step",
  "10 points",
  [
    The benchmark will be run at batch size 2 on two B200 GPUs. Your submission will be evaluated on wall-clock time for a complete training step: forward pass, loss, backward pass, and AdamW update.

    From an empty PyTorch/Triton cache, your benchmarking run must complete within 10 minutes, so be careful with overly aggressive #raw("torch.compile") and Triton autotuning.

    #deliverable[Your best wall-clock time for a full forward-and-backward training step with AdamW.

    We expect leaderboard submissions to beat the naïve baseline of 10 seconds.

    Submit your result to the leaderboard here: #raw("github.com/stanford-cs336/assignment2-systems-leaderboard").]
  ],
  answer: [
  ],
)