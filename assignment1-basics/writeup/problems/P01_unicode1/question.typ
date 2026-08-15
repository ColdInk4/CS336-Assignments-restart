#import "../../template.typ": problem, deliverable

#problem("unicode1", "Understanding Unicode", "1 point")[
(a) What Unicode character does `chr(0)` return?

#deliverable[A one-sentence response.]

(b) How does this character's string representation (`__repr__()`) differ from its printed representation?

#deliverable[A one-sentence response.]

(c) What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:

#raw(">>> chr(0)\n>>> print(chr(0))\n>>> \"this is a test\" + chr(0) + \"string\"\n>>> print(\"this is a test\" + chr(0) + \"string\")", block: true, lang: "python")

#deliverable[A one-sentence response.]
]
