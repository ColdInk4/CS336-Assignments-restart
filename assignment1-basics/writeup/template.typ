#let section-label(body) = block(
  above: 0pt,
  below: 0.65em,
)[
  #text(size: 12pt, weight: "bold", fill: rgb("#1f2937"))[#body]
]

#let prompt-box(body) = block(
  width: 100%,
  fill: rgb("#f7f8fa"),
  stroke: rgb("#d8dee6"),
  inset: (x: 11pt, y: 10pt),
  radius: 4pt,
  above: 0.45em,
  below: 0.85em,
)[
  #body
]

#let answer-box(body) = block(
  width: 100%,
  fill: rgb("#ffffff"),
  stroke: rgb("#cfd7e2"),
  inset: (x: 12pt, y: 11pt),
  radius: 4pt,
  above: 0.35em,
  below: 1.15em,
)[
  #body
]

#let deliverable(body) = block(
  width: 100%,
  fill: rgb("#eef4ff"),
  stroke: rgb("#c8d7ee"),
  inset: (x: 8pt, y: 6pt),
  radius: 3pt,
  above: 0.55em,
  below: 0.65em,
)[
  #text(weight: "bold")[Deliverable:] #body
]

#let note(body) = block(
  width: 100%,
  fill: rgb("#fff8dc"),
  stroke: rgb("#e5d38a"),
  inset: (x: 8pt, y: 6pt),
  radius: 3pt,
  above: 0.55em,
  below: 0.65em,
)[
  #body
]

#let plot(path, width: 88%) = block(
  width: 100%,
  above: 0.75em,
  below: 0.9em,
)[
  #align(center)[#image(path, width: width)]
]

#let problem(id, title, points, body) = [
  #heading(level: 1)[Problem (#raw(id)): #title (#points)]
  #v(0.25em)
  #prompt-box[
    #section-label[Prompt]
    #body
  ]
]

#let answer(body) = answer-box[
  #section-label[Answer]
  #body
]

#let answer-placeholder() = answer[
  #text(fill: rgb("#6b7280"))[TODO: Write your response here.]
]
