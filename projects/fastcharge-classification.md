# Predicting battery cycle life from the first 100 cycles

**Can we tell, after only 100 cycles, which cells will wear out early, and what is that prediction worth in dollars?**

```{image} ../_static/figs/battery/deltaq_vs_life.png
:alt: Capacity-loss proxy between cycles 10 and 100 plotted against cycle life
:width: 520px
:align: center
```

## Why it matters

Lithium-ion cells that look identical on day one can last 150 cycles or 2,200. For a manufacturer or a grid-storage operator, spotting the short-lived ones early means catching defects before deployment instead of after a failure in the field. The catch is that, early in life, the two groups look almost the same: at cycle 10 their state of health is indistinguishable.

## Data: 27 GB in, 100 kB out

I used the BEEP-structured version of the Severson et al. (2019) fast-charging dataset from Toyota Research Institute: **139 commercial LFP/graphite 18650 cells**, each cycled to end of life under one of dozens of two-step fast-charging protocols.

- The raw data is **139 JSON files totalling about 27 GB** (117–459 MB each). They contain bare `NaN` tokens that standard JSON parsers reject.
- I wrote a **streaming extractor** that cleans each file, pulls only the per-cycle summary statistics, and computes features from the **first 100 cycles only**. The result is a 139 × 32 feature matrix of about 100 kB.
- I **matched 96 cells to their published charging protocols** by fuzzy-matching cycle life against the paper's supplementary table, then parsed each protocol string (e.g. `5.6C(36%)-4.3C`) into three numeric design variables.

**Target:** end of life is when capacity drops below 80% of nominal (0.88 Ah). Cells below the median life of 787 cycles are labelled *early degraders*, giving 68 early degraders and 71 long-life cells.

**Features (32, all physically motivated):** capacity fade, internal resistance level and growth, temperature, coulombic efficiency, charge-time behaviour, and the charging protocol. Anything that would leak the answer, such as capacity at cycle 100, is excluded.

```{image} ../_static/figs/battery/cycle_life.png
:alt: Sorted cycle life for all 139 cells, split at the median of 787 cycles
:width: 480px
:align: center
```

## Models and a cost-aware decision rule

I compared **logistic regression** as the interpretable baseline, a **random forest**, and **XGBoost**, with the latter two tuned by Bayesian optimization (Optuna) using 5-fold stratified cross-validation. XGBoost learned how to handle the 43 cells with no protocol data directly, rather than having them imputed.

Accuracy alone is the wrong yardstick here, because the two kinds of error have very different costs. A **missed early degrader** risks a field failure, which I modelled at −$25,000. A **false alarm** means replacing a healthy cell, modelled at −$1,500. So I scored each model in **dollars per cell** under this cost matrix and set each model's decision threshold to maximize that value rather than using the default 0.5.

## Results (held-out test set of 28 cells)

| Model | ROC-AUC | PR-AUC | Utility-optimal threshold | Missed early degraders | False alarms | Value per cell |
|---|---|---|---|---|---|---|
| Logistic regression | 0.939 | 0.950 | 0.33 | 0 | 5 | $46 |
| **Random forest** | **0.954** | 0.949 | 0.25 | **0** | **2** | **$229** |
| XGBoost | 0.949 | 0.940 | 0.07 | 0 | 3 | $168 |

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item}
```{image} ../_static/figs/battery/roc.png
:alt: ROC curves for the three models
```
:::

:::{grid-item}
```{image} ../_static/figs/battery/utility_threshold.png
:alt: Expected value per cell as a function of decision threshold
```
:::
::::

All three models caught every early degrader in the test set. The utility-optimal thresholds sit far below 0.5, which is what you'd expect when a miss costs about 17 times more than a false alarm.

## What the models learned

```{image} ../_static/figs/battery/shap.png
:alt: SHAP beeswarm plot for the XGBoost model
:width: 520px
:align: center
```

- **Internal resistance right after formation (cycle 2) is the strongest single predictor.** It ranks first in both random-forest importance and SHAP. Physically, it reflects how well the initial SEI layer formed. In practice, a single impedance measurement after formation could serve as an incoming quality screen.
- **First-step charge rate shows a clear threshold.** In a one-at-a-time sensitivity sweep, predicted risk jumps from about 3% to about 20% once the first charging step exceeds **6C**. That is a concrete constraint for protocol design.
- **Early capacity fade acts as a step, not a slope.** Above about 3–5 mAh of loss between cycles 10 and 100, early degradation becomes far more likely. That threshold behaviour is why the tree models beat the linear baseline.

```{image} ../_static/figs/battery/sensitivity_controls.png
:alt: Sensitivity of predicted degradation risk to charging protocol variables
:width: 620px
:align: center
```

SHAP looks back at individual predictions, while the sensitivity sweeps ask what-if questions. Both point to the same physical story, which gives some confidence that the models are picking up electrochemistry rather than noise.

## Limitations and next steps

- **Small test set.** The difference between the random forest and XGBoost comes down to one cell, and their cross-validated AUCs are essentially tied (0.966 vs 0.969). Repeated, multi-seed cross-validation is the next step, along with choosing the threshold on validation data rather than the test set.
- **Single chemistry and temperature.** All cells are A123 LFP/graphite at 30 °C, so transfer to NMC cells or other operating conditions needs testing.
- **Possible batch effect.** Cycle-2 resistance may partly encode manufacturing batch, which would limit generalization to cells from other lots.
- **Illustrative costs.** The dollar values are plausible but assumed; a real deployment would plug in its own costs.

**Tools:** Python, pandas, scikit-learn, XGBoost, Optuna, SHAP, matplotlib/seaborn

```{note}
TODO: link the GitHub repository with the extractor and modelling code.
```
