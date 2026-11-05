# Predicting battery cycle life from the first 100 cycles

[![Repository](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white&style=for-the-badge)]([https://github.com/ThomasAko/physics-constraints-qm7-qm9](https://github.com/ThomasAko/battery-early-life-prediction))

**Can we tell, after only 100 cycles, which cells will wear out early, and what is that prediction worth in dollars?**

```{image} ../_static/figs/battery/deltaq_vs_life.png
:alt: Capacity lost between cycles 10 and 100 plotted against cycle life
:width: 520px
:align: center
```

## Why it matters

Lithium-ion cells that look identical on day one can last 150 cycles or 2,200. For a manufacturer or a grid-storage operator, spotting the short-lived ones early means catching them before deployment instead of after a failure in the field. The catch is that, early in life, the two groups look almost the same: at cycle 10 their state of health is indistinguishable.

## Data: 27 GB in, 100 kB out

I used the BEEP-structured version of the Severson et al. (2019) fast-charging dataset from Toyota Research Institute: **139 commercial LFP/graphite 18650 cells**, each cycled to end of life under one of dozens of two-step fast-charging protocols.

- The raw data is **139 JSON files totalling about 27 GB** (117–459 MB each). They contain bare `NaN` tokens that standard JSON parsers reject.
- I wrote an **extractor that processes the files one at a time**, cleans the invalid tokens, keeps only the per-cycle summary statistics, and computes features from the **first 100 cycles only**. The result is a feature matrix of about 100 kB.
- The file names don't say which charging protocol each cell received, so I **matched 96 cells to their published protocols** using cycle life, the one field the files share with the paper's supplementary table. The rule is deliberately conservative: nearest match within 15 cycles, and no second-best fallback when that cell is already taken. Each protocol string (e.g. `5.6C(36%)-4.3C`) becomes three numeric design variables.
- **Sanity check:** the 2018 test batch reproduces the paper's cycle lives exactly for all but one cell. The two 2017 batches sit one cycle short, consistent with a cycle-indexing convention in those files, and this changes at most one label out of 139.

**Target:** end of life is when capacity drops below 80% of nominal (0.88 Ah). Cells below the median life of 787 cycles are labelled *early degraders*, giving 68 early degraders and 71 long-life cells.

**Features (32, all physically motivated):** capacity fade, internal resistance level and growth, temperature, coulombic efficiency, charge-time behaviour, and the charging protocol. Anything that would leak the answer, such as capacity at cycle 100, is excluded.

```{image} ../_static/figs/battery/cycle_life.png
:alt: Sorted cycle life for all 139 cells, split at the median of 787 cycles
:width: 480px
:align: center
```

## Models and a cost-aware decision rule

I compared **logistic regression** as the interpretable baseline, a **random forest**, and **XGBoost**, with the latter two tuned by Bayesian optimization (Optuna) using 5-fold stratified cross-validation. XGBoost handled the 43 cells with no protocol data directly, learning where to route missing values instead of having them imputed.

Accuracy alone is the wrong yardstick here, because the two kinds of error have very different costs. A **missed early degrader** risks a field failure, which I modelled at −\$25,000. A **false alarm** means replacing a healthy cell, modelled at −\$1,500. So I scored each model in **dollars per cell** under this cost matrix and set each model's decision threshold to maximize that value rather than using the default 0.5. The threshold is chosen on **cross-validated predictions from the training set**, so the test set is used only once.

## Results

| Model | Test ROC-AUC | Repeated-CV ROC-AUC | Test: missed / false alarms | Test value per cell | Fleet simulation: value over best no-model policy |
|---|---|---|---|---|---|
| Logistic regression | 0.939 | 0.918 | 0 / 9 | −\$196 | +\$25 |
| Random forest | **0.954** | 0.963 | 1 / 2 | −\$682 | +\$148 |
| XGBoost | 0.949 | **0.968** | 1 / 3 | −\$743 | **+\$153** |

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item}
```{image} ../_static/figs/battery/roc.png
:alt: ROC curves for the three models on the held-out test set
```
:::

:::{grid-item}
```{image} ../_static/figs/battery/utility_threshold.png
:alt: Value per cell as a function of decision threshold
```
:::
::::

**Why are the test values negative?** With the threshold chosen honestly on the training data, each tree model lets one of the 14 early degraders through. On 28 cells, a single \$25,000 miss outweighs everything else: without it, random forest would be at +\$229 per cell. Zero isn't the right yardstick anyway. Replacing every test cell would score −\$500 per cell, and deploying every cell −\$12,400.

**One cell decides the test ranking**, so I didn't stop there. Across 50 repeated cross-validation splits of all 139 cells, random forest and XGBoost are tied (ROC-AUC 0.963 vs 0.968, with a fold-to-fold spread of about 0.03), and both are clearly ahead of logistic regression.

## What is the prediction worth?

The test table tells you how good the classifier is. An operator wants to know what screening new cells this way gains, how much that could vary, and how much it depends on the assumptions.

I simulated 5,000 fleets of 1,000 cells, built by resampling the real cells with **out-of-fold predictions** (each cell scored by a model that never saw it). Each simulation also re-picks the threshold on a bootstrap sample, so uncertainty in the threshold is included. Without a model, the best simple policy here is to replace every cell, at −\$522 per cell.

```{image} ../_static/figs/battery/fleet_monte_carlo.png
:alt: Distribution of value per cell across 5,000 simulated fleets for the three models
:width: 620px
:align: center
```

- **Every model beats the best no-model policy.** Random forest and XGBoost each add about **\$150 per cell**, logistic regression about \$25. The two tree models are effectively tied here too.
- **Under this cost matrix, screening is close to break-even in absolute terms.** With half the cells being early degraders and only \$200 earned for clearing a healthy cell, false alarms are expensive and misses are very expensive. In about a quarter of simulated fleets, a tree model does worse than simply replacing everything.
- **Screening wins whatever a miss costs; how much it wins depends on the costs.** Across miss costs from 5× to 100× a false alarm, every model still beats the no-model policy. Random forest's best threshold settles once a miss costs more than about 20× a false alarm, because the best rule then catches essentially every early degrader.
- **How common early degraders are matters a lot.** The 49% share here is an artefact of the median split. Screening is worth most when early degraders make up a few percent to about 20% of production, where the tree models gain several hundred to about a thousand dollars per cell over the no-model policy (an upper bound, since the threshold is tuned on the same cells). When they are very rare, deploying everything is already cheap, and the best rule flags only cells the model is quite sure about.

```{image} ../_static/figs/battery/prevalence_sweep.png
:alt: Value of the prediction and best threshold as the share of early degraders varies from 1% to 49%
:width: 680px
:align: center
```

## What the models learned

```{image} ../_static/figs/battery/shap.png
:alt: SHAP beeswarm plot for the XGBoost model
:width: 520px
:align: center
```

- **Internal resistance right after formation (cycle 2) is the strongest single predictor.** It ranks first in random-forest importance and in SHAP, whether SHAP is computed exactly in log-odds or on the predicted probability (the two SHAP methods agree on the top three features). Physically, it reflects how well the initial SEI layer formed. In practice, a single impedance measurement after formation could serve as an incoming quality screen.
- **First-step charge rate shows a clear threshold.** In a one-at-a-time sensitivity sweep on a typical cell, predicted risk jumps from about 3% to about 20% once the first charging step exceeds **6C**. That is a concrete constraint for protocol design.
- **Early capacity fade acts as a step, not a slope.** Above about 3–4 mAh of loss between cycles 10 and 100, early degradation becomes far more likely. Threshold behaviour like this is why the tree models beat the linear baseline.

```{image} ../_static/figs/battery/sensitivity_controls.png
:alt: Sensitivity of predicted degradation risk to charging protocol variables and early capacity loss
:width: 620px
:align: center
```

SHAP looks back at individual predictions, while the sensitivity sweeps ask what-if questions. Both point to the same physical story, which gives some confidence that the models are picking up electrochemistry rather than noise.

## Limitations and next steps

- **Small data.** 139 cells, 28 in the test set; one cell decides the test ranking. Hyperparameters were tuned on the training cells, so the cross-validated numbers are mildly optimistic; nested cross-validation and repeating the split over several seeds are the next steps.
- **Single chemistry and temperature.** All cells are A123 LFP/graphite at 30 °C, so transfer to NMC cells or other operating conditions needs testing.
- **Possible batch effect.** Cycle-2 resistance may partly encode manufacturing batch, which would limit generalization to cells from other lots.
- **Illustrative costs.** The dollar values are plausible but assumed. The sweeps show screening beats the no-model policy across a wide range of miss costs, but a real deployment would plug in its own numbers and its own share of early degraders.
- **Sensitivity sweeps are not causal.** They hold correlated features fixed; in a real cell, a different charging protocol would also change resistance and fade.

**Tools:** Python, pandas, scikit-learn, XGBoost, Optuna, SHAP, matplotlib/seaborn

**Code:** [github.com/ThomasAko/battery-early-life-prediction](https://github.com/ThomasAko/battery-early-life-prediction)
