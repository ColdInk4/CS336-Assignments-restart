#import "../../template.typ": answer

#answer[

(a) 返回的是码点为`U+0000`，名称为`NUL`的控制字符，解释器以 `'\x00'` 这种转义表示来显示它。

(b) `__repr__()`用转义的可见文本描述字符，print 输出实际的字符，NUL字符通常没有可见字形，其 print 结果通常不可见。

(c) 在交互式解释器直接输入表达式时，解释器自动显示其 repr，而 print 写出字符串的实际内容，像`"this is a test" + chr(0) + "string"`里 `chr(0)` 这个字符插在文本中，只是打印时不可见。
#raw(">>> chr(0)
'\x00'
>>> print(chr(0))

>>> \"this is a test\" + chr(0) + \"string\"
'this is a test\x00string'
>>> print(\"this is a test\" + chr(0) + \"string\")
this is a teststring
"
, block: true, lang: "python")


]