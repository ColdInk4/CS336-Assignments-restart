#import "../../template.typ": problem, deliverable, note

#problem("unicode2", "Unicode Encodings", "3 points")[
(a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.

#deliverable[A one-to-two sentence response.]

(b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.

#raw("def decode_utf8_bytes_to_str_wrong(bytestring: bytes):\n    return \"\".join([bytes([b]).decode(\"utf-8\") for b in bytestring])\n\n>>> decode_utf8_bytes_to_str_wrong(\"hello\".encode(\"utf-8\"))\n'hello'", block: true, lang: "python")

#deliverable[An example input byte string for which `decode_utf8_bytes_to_str_wrong` produces incorrect output, with a one-sentence explanation of why the function is incorrect.]

(c) Give a two-byte sequence that does not decode to any Unicode character(s).

#deliverable[An example, with a one-sentence explanation.]
]
