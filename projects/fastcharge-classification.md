# Early prediction of battery degradation

*Advanced Data Mining, Lewis University · 2026*

**Problem.** Can we tell from early cycles which cells will degrade fast, and is acting on that prediction worth it in dollars?

**What I did.**
- Built classifiers on the Severson et al. fast-charging dataset to separate fast- from slow-degrading cells using early-cycle features
- Compared models and framed the choice with a **dollar-denominated utility function** instead of accuracy alone
- Recommended a Random Forest: **AUC 0.954**, with an estimated **$229 per cell** of value

**Role.** Individual project.

```{note}
TODO: add the ROC curve and the utility-vs-threshold plot, and link the notebook.
```

**Tools:** Python, scikit-learn, pandas, matplotlib
