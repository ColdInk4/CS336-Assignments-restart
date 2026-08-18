#import "../../template.typ": answer
#answer[
  (a) 最长的 token 是一个 64 字节的序列，由重复 16 次的 `c383c382` 组成，显示为 `ÃÂÃÂ...`。原始 OpenWebText 数据中确实存在相同的编码伪影，是网页语料中的乱码；虽然没有语义意义，但对该语料来说是合理的。
  (b) TinyStories tokenizer 主要学习干净、可读的英文词元，例如 `b' accomplishment'`；OpenWebText 的词表更大、内容更杂，会学习网页格式、罕见字符串以及编码伪影等 token。
]
