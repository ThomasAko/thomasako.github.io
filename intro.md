# Thomas Ako, PhD

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/thomas8ako5/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white)](https://github.com/ThomasAko)
[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-4285F4?logo=googlescholar&logoColor=white)](https://scholar.google.com/citations?user=1bh6tdIAAAAJ&hl=en&authuser=1)
[![CV](https://img.shields.io/badge/CV-PDF-555555)](cv.md)

```{image} _static/profile.jpg
:alt: Thomas Ako
:width: 200px
:align: right
```

**Physics-informed machine learning for battery diagnostics.**

I'm a data scientist in the Data Science and Learning division at **Argonne National Laboratory**, where I develop AI/ML models for lithium-ion battery diagnostics and prognostics.

My path here is unusual. I hold a **PhD in Electrical Engineering from Ghent University** (integrated silicon photonics). After that I spent several years in **test and measurement**: photonics test engineering at IMEC, then applications and technical sales roles at Keysight, Sofics and Dennemeyer. I then worked as a **patent examiner at the European Patent Office**, where I examined more than 100 applications in electrical test and measurement from companies including Airbus, Texas Instruments, CATL and LG. Through all of it, the common thread has been measurement, the physics behind a signal, and turning noisy data into decisions.

I complete an **MS in Data Science** at Lewis University in December 2026.

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

:::{grid-item-card} Physics constraints for molecular ML: QM7 → QM9
:link: projects/qm7-energy-classification
:link-type: doc
:img-top: _static/figs/qm7/coulomb_matrix.png
Why a virial-theorem loss couldn't bind on QM7, and testing a NeurIPS 2025 energy loss on QM9 instead.
:::

:::{grid-item-card} Battery curve-fitting agent
:link: projects/battery-fit-agent
:link-type: doc
LangGraph + local Llama agent that fits capacity-fade curves and flags outlier cells.
:::

::::

## Get in touch

I'm open to conversations about battery analytics, physics-informed ML, and technical patent advisory. The fastest way to reach me is [email](mailto:TODO) or [LinkedIn](https://www.linkedin.com/in/thomas8ako5/).
