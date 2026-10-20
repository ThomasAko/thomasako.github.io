# Patent analytics and search re-ranking

*Independent work · 2026*

**Background.** As a patent examiner at the European Patent Office I examined more than 100 applications in electrical test and measurement. The hardest part of search is the "last mile": going from a few hundred plausible documents to the handful that actually matter.

**What I'm building.**
- A **re-ranker** that reorders a pre-selected candidate set of patents on top of a normal search ranking, using claim-level semantic similarity
- Worked case studies on public patents in photonics, showing claim charts and prior-art reasoning
- Built entirely on public patent data

**Status.** In development.

```{note}
TODO: add the first public case study and a short demo of the re-ranker.
```

**Tools:** Python, sentence embeddings, patent databases (Espacenet, Google Patents)
