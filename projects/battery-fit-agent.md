# An AI agent that triages battery ageing data

**A LangGraph agent that fits degradation curves to 139 lithium-ion cells, finds the ones that behave unusually,
investigates each with tools, and explains what it found, with every number computed by code, not by the model.**

[Code on GitHub](https://github.com/ThomasAko/battery-fit-agent) ·
[Run it in Colab](https://colab.research.google.com/github/ThomasAko/battery-fit-agent/blob/main/notebooks/colab_investigator.ipynb) ·
Python, LangGraph, LangChain, SciPy, pandas · Groq, Gemini, Claude or local models

---

## At a glance

- **Problem:** in a battery test lab, an engineer fits curves to hundreds of capacity-fade curves and hunts for the
  odd ones out. It is slow, repetitive, and easy to get wrong at scale.
- **Approach:** three layers. Deterministic tools do all the numerics, fleet statistics decide what is unusual, and
  an LLM agent investigates only the shortlist, choosing its own tool calls, with a verifier checking its conclusions.
- **Result:** on ten hand-labelled cells, **9 of 10 verdicts match**, the verifier caught and corrected **3** first
  answers, and the report cites only numbers the tools produced. A first, simpler design flagged 139 of 139 cells,
  and analysing why it failed shaped the second.
- **Data:** the public fast-charging dataset of 139 LFP/graphite cells from Severson et al., *Nature Energy* (2019).
  The per-cell data ships with the repo, so it runs straight after cloning.

![Architecture](https://raw.githubusercontent.com/ThomasAko/battery-fit-agent/main/docs/figures/architecture.png)

## The problem

A battery loses capacity every time it is cycled. Plotted against cycle number, capacity usually falls slowly and
steadily, then bends at a *knee* and drops fast until the cell reaches end of life (80 % of its rated capacity).
Which mathematical law describes that curve, where the knee sits, and which cells deviate from the rest are basic
questions in battery diagnostics. Answering them by hand for a whole test campaign takes days.

## How it works

**1. Tools: every number, no AI.** For each cell, the code removes measurement glitches, estimates noise, fits seven
candidate laws (from linear to double exponential and a two-segment "knee" model), ranks them by the Bayesian
information criterion, and locates the knee and the end-of-life cycle.

**2. Fleet statistics: what counts as unusual.** Each cell is compared with the other 138 using robust z-scores:
is its misfit unusually large, its knee unusually early, its shape different from the fleet's typical law? Tests
that stopped long before end of life are kept out of the knee comparison, because their "knee" is measured against
an unfinished test. Result: 31 of 139 cells flagged, 7 at high priority.

**3. The agent: why it is unusual.** For each shortlisted cell, an LLM runs a tool-calling loop. It can ask for an
overview of the cell, check whether the test reached end of life, locate the outliers, compare any metric with the
fleet, read the fit table, or refit the curve over a window of cycles. It decides which tools to call and when it
knows enough, then returns a structured verdict: a category (real early ageing, unusual shape, test ended before end
of life, data quality, or normal), a confidence, evidence quoting tool outputs, and a recommended next step.

**4. The verifier.** Plain Python checks each verdict against the tool results. Calling a cell "early ageing" when
the tools say it never reached end of life, for example, sends the agent back with the specific contradiction.

**5. The report** is written from the structured verdicts only, never from raw data.

## What I found

### Version 1: the LLM as a judge

The first design ran every cell through a fixed pipeline and asked a local Llama 3.1 8B model to decide whether
each one deserved attention. On all 139 cells it **flagged 139 and called 138 "high priority"**; its stated
reasons sometimes contradicted the numbers it had been given; and its summary report recommended three cells, none
of them on the shortlist the fleet statistics produced, all from the end of the input list. That pattern points to
a prompt longer than the model's context window.

The fix was not a bigger model. It was a different job for the model: numbers and triage stay deterministic, and the
LLM investigates a short list with tools.

### Version 2: the LLM as an investigator

Run with GPT OSS 120B on Groq's free tier:

| | Version 1 | Version 2 |
|---|---|---|
| Cells flagged | 139 of 139 | 31 of 139 |
| Verdicts matching hand labels | n/a | 9 of 10 |
| Verdicts corrected by the verifier | no verifier | 3, all to the right category |
| Report | missed the shortlist | grouped all ten correctly, cited only tool numbers |

The one disagreement was a judgement call: the agent classed a cell as a data-quality case because its outliers sit
*after* end of life, at the very end of the test, which my own label had not considered.

<table><tr>
<td><img src="https://raw.githubusercontent.com/ThomasAko/battery-fit-agent/main/docs/figures/early_knee_000026_CH6.png" alt="Early ageing"></td>
<td><img src="https://raw.githubusercontent.com/ThomasAko/battery-fit-agent/main/docs/figures/no_eol_000045_CH2.png" alt="Test ended before end of life"></td>
</tr><tr>
<td><sub>Real early ageing: knee at cycle 168, end of life at 483.</sub></td>
<td><sub>Not early ageing: this test stopped at 20 % of the fade to end of life, so its "early knee" is an artefact.</sub></td>
</tr></table>

A simple rule-based classifier also scores 10 of 10 on these cells, and I say so in the repository: the labels and
the rules come from the same reading of the plots. What the agent adds is checkable evidence for each verdict and the
ability to handle cases no rule anticipated. Ten cells is a test of the design, not a benchmark.

## Engineering choices

- **Numbers come from code, judgement from the model.** The model never computes anything; it decides what to ask
  and how to interpret answers. That is what makes its output auditable.
- **A deterministic verifier around the LLM.** Cheap to write, and in practice it corrected 30 % of first answers.
- **Provider-agnostic.** One setting switches between Groq, Gemini, Claude, OpenRouter and local Ollama models; the
  multimodal providers also see each cell's diagnostic plot.
- **Built for real APIs.** Rate limiting for free tiers, retries when a provider is overloaded, and a clean stop
  when a daily quota runs out, all added after hitting those limits in practice.
- **Testable without an API.** Unit tests drive the full agent with a scripted stand-in model, including the
  verifier loop and API failures.
- **Every run leaves a trace.** Each investigation is saved step by step (tool calls, results, verifier feedback,
  verdict), and the notebook shows them next to the plots.

## Stack

Python · LangGraph (state graphs, tool-calling loop, subgraphs) · LangChain (tool definitions, structured output,
provider integrations) · Pydantic · NumPy, SciPy, pandas, Matplotlib · pytest · Groq, Gemini, Claude, Ollama
