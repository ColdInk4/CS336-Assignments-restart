#set document(
  title: "CS336 Assignment 2 (systems): Systems and Parallelism",
  author: "",
)
#set page(paper: "us-letter", margin: (x: 0.85in, y: 0.8in))
#set text(font: ("New Computer Modern", "Noto Serif CJK SC", "Noto Sans CJK SC"), size: 10.5pt)
#show raw: set text(font: ("DejaVu Sans Mono", "Noto Sans Mono CJK SC"), size: 9pt)
#set par(justify: true, leading: 0.58em)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 18pt, weight: "bold")[CS336 Assignment 2 (systems): Systems and Parallelism]
  #v(0.3em)
  #text(fill: rgb("#4b5563"))[Writeup]
]

#v(1.2em)
#outline(title: [Contents])
#pagebreak()

#include "problems/P01_benchmarking_script/solution.typ"
#include "problems/P02_nsys_profile/solution.typ"
#include "problems/P03_mixed_precision_accumulation/solution.typ"
#include "problems/P04_benchmarking_mixed_precision/solution.typ"
#include "problems/P05_memory_profiling/solution.typ"
#include "problems/P06_gradient_checkpointing/solution.typ"
#include "problems/P07_pytorch_attention/solution.typ"
#include "problems/P08_torch_compile/solution.typ"
#include "problems/P09_flash_forward/solution.typ"
#include "problems/P10_flash_backward/solution.typ"
#include "problems/P11_flash_benchmarking/solution.typ"
#include "problems/P12_distributed_communication_single_node/solution.typ"
#include "problems/P13_naive_ddp/solution.typ"
#include "problems/P14_naive_ddp_benchmarking/solution.typ"
#include "problems/P15_minimal_ddp_flat_benchmarking/solution.typ"
#include "problems/P16_ddp_overlap_individual_parameters/solution.typ"
#include "problems/P17_ddp_overlap_individual_parameters_benchmarking/solution.typ"
#include "problems/P18_optimizer_state_sharding/solution.typ"
#include "problems/P19_optimizer_state_sharding_accounting/solution.typ"
#include "problems/P20_fsdp/solution.typ"
#include "problems/P21_fsdp_accounting/solution.typ"
#include "problems/P22_alternate_ring_all_reduce/solution.typ"
#include "problems/P23_data_parallel_calcs/solution.typ"
#include "problems/P24_fsdp_calcs/solution.typ"
#include "problems/P25_tp_calcs/solution.typ"
#include "problems/P26_fsdp_tp_calcs/solution.typ"
#include "problems/P27_leaderboard/solution.typ"