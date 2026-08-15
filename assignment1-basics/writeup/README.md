# Assignment 1 Writeup

Compile the writeup from the assignment directory:

```sh
cd assignment1-basics
typst compile writeup/main.typ writeup.pdf
```

Each problem is self-contained under `writeup/problems/Pxx_*`:

- `question.typ` contains only the assignment prompt and deliverables.
- `answer.typ` is your response and is initially an empty placeholder.
- `assets/` is reserved for figures used by that problem. Create it when the problem needs plots or images, then reference an asset from `answer.typ` with `#plot("assets/filename.png")`.

The `main.typ` entry point includes the question and answer separately, in assignment order. Do not edit `question.typ` while solving; keep all work in the matching `answer.typ` and that problem's `assets/` directory.
