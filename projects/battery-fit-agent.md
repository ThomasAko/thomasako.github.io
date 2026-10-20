# Battery curve-fitting agent

*Generative AI course project (CPSC 57900), Lewis University · 2026*

**Problem.** Fitting capacity-fade curves across a large set of cells and spotting the unusual ones is slow, repetitive work that researchers usually do by hand.

**What I did.**
- Built an agent with **LangGraph** and a **local Llama model (Ollama)** that loads cycling data, fits capacity-fade models, checks fit quality, and flags cells that deserve a closer look
- Ran it on the public **Severson et al. fast-charging dataset** (124 LFP/graphite cells), with data extraction done locally and the agent run on a Colab T4 GPU
- Designed it so it can be reused on internal lab data without sending anything to an external API

**Role.** Sole developer.

```{note}
TODO: add a screenshot of the agent's output and a link to the repository once it's public.
```

**Tools:** LangGraph, LangChain, Ollama/Llama, Python, SciPy, Google Colab
