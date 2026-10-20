# Joint state-parameter estimation for LFP batteries

*Argonne National Laboratory · DOE ROVI/ACCESS program · 2026*

**Problem.** A battery management system has to track state of charge while the cell's parameters (capacity, resistance) drift with age. Doing both at once online is a joint estimation problem, and it raises a basic question: how good *can* the estimate be, given the data?

**What I did.**
- Built an Unscented Kalman Filter (UKF) for joint state-parameter estimation on lithium iron phosphate (LFP) cells
- Generated a 144-cell synthetic population via co-simulation of NREL's BLAST-Lite degradation model and Argonne's [Moirae](moirae.md) framework, so the true parameters are known
- Computed the posterior Cramér–Rao bound (PCRB) as a lower limit on achievable estimation error, and studied where and why the filter's error violates it across a reference performance test (cycling → HPPC → capacity check)
- Validated on experimental data from a CAMP NMC cell

**Role.** Lead author; designed the study, the simulation pipeline and the analysis.

**Status.** Manuscript in preparation for *Journal of Power Sources*. This is the first of a planned four-paper series (hybrid UKF-NN with a PCRB-regularized loss; a PyBaMM DFN truth tier with structural model mismatch; a multi-chemistry benchmark dataset).

```{note}
TODO: add one cleared figure (e.g. estimation error vs. PCRB over an RPT) once publication is approved.
```

**Tools:** Python, NumPy/SciPy, Moirae, BLAST-Lite, LCRC HPC clusters
