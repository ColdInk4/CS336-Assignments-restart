# Assignment 2 Writeup

Compile the writeup from the assignment directory:

```sh
cd assignment2-systems
typst compile writeup/main.typ writeup.pdf
```

Unlike Assignment 1, the prompt and your answer for each problem live in a
single Typst file:

- `problems/Pxx_*/solution.typ` contains both the assignment prompt
  (in the `Prompt` block) and your response (in the `Answer` block).
- `assets/` is reserved for figures used by that problem. Create it when
  the problem needs plots or images, then reference an asset from
  `solution.typ` with `#plot("assets/filename.png")`.

The `main.typ` entry point includes each problem's `solution.typ` in
assignment order. Keep all work in the matching `solution.typ` and that
problem's `assets/` directory.

## How to fill in an answer

Each `solution.typ` already ends with an empty `answer` block, ready to
be filled in:

```typst
#solution(
  ...,
  [
    ... prompt ...
  ],
  answer: [
    (a) 我的回答 ...
    (b) ...
  ],
)
```

Just write your response inside the `answer: [...]` array — there is no
need to remove any placeholder. The corresponding `Answer` block will
appear under the prompt in the compiled PDF.