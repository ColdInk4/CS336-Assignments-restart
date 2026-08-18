#import "../../template.typ": answer
#answer[
  (a) 训练的 wall-clock 时间约为 21.23 秒；GNU `time -v` 报告的峰值 RSS 为 188.5 MiB。
  最长的 token 是 `' accomplishment'`，它是一个完整且在儿童故事中常见的英文词，前导空格也符合 GPT-2 风格预分词会把空格并入词元的行为。
  
  (b) 本次训练最耗时的阶段是 pre-tokenization；在 worker 内，反复从 regex match 中提取完整匹配文本的 `Match.group()` 调用占据了主要时间。
]
