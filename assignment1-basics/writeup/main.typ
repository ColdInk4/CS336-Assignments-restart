#set document(
  title: "CS336 Assignment 1: Basics",
  author: "",
)
#set page(paper: "us-letter", margin: (x: 0.85in, y: 0.8in))
#set text(font: ("New Computer Modern", "Noto Serif CJK SC", "Noto Sans CJK SC"), size: 10.5pt)
#show raw: set text(font: ("DejaVu Sans Mono", "Noto Sans Mono CJK SC"), size: 9pt)
#set par(justify: true, leading: 0.58em)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 18pt, weight: "bold")[CS336 Assignment 1: Basics]
  #v(0.3em)
  #text(fill: rgb("#4b5563"))[Writeup]
]

#v(1.2em)
#outline(title: [Contents])
#pagebreak()

#include "problems/P01_unicode1/question.typ"
#include "problems/P01_unicode1/answer.typ"
#include "problems/P02_unicode2/question.typ"
#include "problems/P02_unicode2/answer.typ"
#include "problems/P03_train_bpe_tinystories/question.typ"
#include "problems/P03_train_bpe_tinystories/answer.typ"
#include "problems/P04_train_bpe_owt/question.typ"
#include "problems/P04_train_bpe_owt/answer.typ"
#include "problems/P05_tokenizer_experiments/question.typ"
#include "problems/P05_tokenizer_experiments/answer.typ"
#include "problems/P06_transformer_accounting/question.typ"
#include "problems/P06_transformer_accounting/answer.typ"
#include "problems/P07_learning_rate_tuning/question.typ"
#include "problems/P07_learning_rate_tuning/answer.typ"
#include "problems/P08_adamw_accounting/question.typ"
#include "problems/P08_adamw_accounting/answer.typ"
#include "problems/P09_learning_rate/question.typ"
#include "problems/P09_learning_rate/answer.typ"
#include "problems/P10_batch_size_experiment/question.typ"
#include "problems/P10_batch_size_experiment/answer.typ"
#include "problems/P11_generate/question.typ"
#include "problems/P11_generate/answer.typ"
#include "problems/P12_layer_norm_ablation/question.typ"
#include "problems/P12_layer_norm_ablation/answer.typ"
#include "problems/P13_pre_norm_ablation/question.typ"
#include "problems/P13_pre_norm_ablation/answer.typ"
#include "problems/P14_no_pos_emb/question.typ"
#include "problems/P14_no_pos_emb/answer.typ"
#include "problems/P15_swiglu_ablation/question.typ"
#include "problems/P15_swiglu_ablation/answer.typ"
#include "problems/P16_main_experiment/question.typ"
#include "problems/P16_main_experiment/answer.typ"
#include "problems/P17_leaderboard/question.typ"
#include "problems/P17_leaderboard/answer.typ"
