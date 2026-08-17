#import "../../template.typ": answer
#answer[

(a) UTF-8是变长编码，而且是当前很多网页都在使用的主要编码。相较UTF-16和UTF-32更加省空间，并且 ASCII 文本不会像 UTF-16/UTF-32 那样出现大量填充零字节。

(b) 因为 UTF-8是变长的，而当前函数是尝试对每个输入的字节都进行单独decode，所以会出现问题。一个错误示例是"牛"，其utf-8表示为`b'\xe7\x89\x9b'`，在调用本函数时会出现`UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe7 in position 0: unexpected end of data`的报错。

(c) 对于UTF-8来说：起始字节的高位 1 的个数表示总字节数，后续所有字节必须严格以 10 开头。所以一个错误例子可以是 11100111 10001001（`b'\xe7\x89`），这里第一位告诉要3个字节，但只给了两个字节。
]