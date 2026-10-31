# Thomas Ako, Ph.D.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/thomas8ako5/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white)](https://github.com/ThomasAko)
[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-4285F4?logo=googlescholar&logoColor=white)](https://scholar.google.com/citations?user=1bh6tdIAAAAJ&hl=en&authuser=1)
[![CV](https://img.shields.io/badge/CV-PDF-555555)](cv.md)

```{image} profile_pic.jpeg
:alt: Thomas Ako
:width: 200px
:align: right
```

**Physics-informed machine learning for batteries and other physical systems**

Welcome! I am a data scientist in the Data Science and Learning division at [Argonne National Laboratory](https://www.anl.gov), where I develop AI/ML models for lithium-ion battery diagnostics and prognostics. I hold an M.Sc. in Electrical Engineering from [KTH Royal Institute of Technology](https://www.kth.se/en) in Stockholm and a Ph.D. in Electrical Engineering from [Ghent University](https://www.ugent.be/en) in Belgium, and I complete an M.S. in Data Science at [Lewis University](https://www.lewisu.edu) in December 2026.

My path after the Ph.D. has been a bit unusual. After a stint as a photonics test engineer at [imec](https://www.imec-int.com/en), I spent several years in applications and technical sales roles at Dennemeyer, Sofics and Keysight Technologies. I then worked as a patent examiner at the [European Patent Office](https://www.epo.org/en), where I examined more than 100 patent applications in electrical test and measurement from companies including Airbus, Texas Instruments, Siemens and LG Energy Solution. Through all of it, the common thread has been measurement: the physics behind a signal, and turning noisy data into decisions.

Going forward, I want to build machine learning models that turn telemetry from physical hardware into decisions: health monitoring, fault detection and remaining useful life prediction for batteries, power electronics and other instrumented systems. I am most interested in models that respect the underlying physics, quantify their uncertainty, and hold up on real field data rather than only on clean lab benchmarks. Alongside this, I continue to work on patent information retrieval, using search and ranking models to find the prior art and technology trends that matter in batteries, BMS and photonics.

## What I work on

- **Battery diagnostics and prognostics:** physics-informed ML, state estimation and early-life prediction of degradation
- **Physics-informed ML more broadly:** building physical constraints into models, and testing honestly whether they help
- **Agentic tools for science:** LLM agents that automate repetitive analysis steps
- **Patent and technology analytics:** search, ranking and technical analysis of patents in batteries, BMS and photonics

## Featured projects

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} Predicting battery life from 100 cycles
:link: projects/fastcharge-classification
:link-type: doc
:img-top: _static/figs/battery/deltaq_vs_life.png
27 GB of cycling data → 32 physics features → ROC-AUC 0.954, with every early degrader in the test set caught.
:::

:::{grid-item-card} Physics-constrained molecular energy classification
:link: projects/qm7-energy-classification
:link-type: doc
:img-top: _static/figs/qm7/coulomb_matrix.png
Coulomb matrices, neural networks and a virial-theorem loss on 7,165 molecules.
:::

:::{grid-item-card} Battery curve-fitting agent
:link: projects/battery-fit-agent
:link-type: doc
LangGraph + local Llama agent that fits capacity-fade curves and flags outlier cells.
:::

::::

## Get in touch

I'm open to conversations about battery analytics, physics-informed ML, and technical patent advisory. The fastest way to reach me is [email](mailto:TODO) or [LinkedIn](https://www.linkedin.com/in/thomas8ako5/).
