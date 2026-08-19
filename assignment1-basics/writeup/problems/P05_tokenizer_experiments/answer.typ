#import "../../template.typ": answer
#answer[
  (a) 在 TinyStories 数据集上面，TinyStories Tokenizer 的压缩比为 4.22，OWT Tokenizer 的压缩比为 4.09。
  (b) 在 OWT 数据集上面，TinyStories Tokenizer 的压缩比为 3.21，OWT Tokenizer 的压缩比为 4.26。说明在 OWT 样本上，OWT tokenizer 的 bytes/token 高，表示同样字节数被编码成更少 token，压缩更好。
  (c) TinyStories Tokenizer 的吞吐大概是 5853714 bytes/second，大约 5.58 MiB/s，训练 Pile dataset(825GiB，搜了一下是这个大小) 需要大约 42 小时；而 OWT Tokenizer 的吞吐大概是 4696115 bytes/second，大约 4.47 MiB/s，训练 Pile dataset 需要大约 52 小时
  (d) 因为两个词表大小分别为 32K 和 10K，用 uint16 足以覆盖两者所有的 token ID，并且比再往后的 uint32，uint64 更省空间。
]
