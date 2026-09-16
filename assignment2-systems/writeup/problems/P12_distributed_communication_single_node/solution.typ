#import "../../template.typ": solution, deliverable

#solution(
  "distributed_communication_single_node",
  "Distributed Communication (Single Node)",
  "5 points",
  [
    Write a script to benchmark the runtime of the all-reduce operation in the single-node multi-process setup. The example code above may provide a reasonable starting point. Experiment with varying the following settings:

    *all-reduce data size* float32 data tensors ranging over 1MB, 10MB, 100MB, 1GB.

    *Number of GPUs/processes* 2, 4, or 6.

    *Resource requirements*: Up to 6 GPUs. Each benchmarking run should take less than 5 minutes.

    #deliverable[Plot(s) and/or table(s) comparing the various settings, with 2-3 sentences of commentary about your results and thoughts about how the various factors interact.]
  ],
  answer: [
  ],
)