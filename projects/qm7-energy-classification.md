# Physics constraints for molecular machine learning on QM7

**Can adding a physical law to a neural network's loss help it predict molecular energies? Testing a virial-theorem penalty on 7,165 molecules turned into a lesson in checking whether a result can be trusted at all.**

[![Open in Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?logo=googlecolab&logoColor=white&style=for-the-badge)](https://colab.research.google.com/github/ThomasAko/physics-constraints-qm7-qm9/blob/main/Physics_Constraints_QM7_to_QM9.ipynb)
[![View notebook](https://img.shields.io/badge/View-Notebook-2a5d8f?logo=jupyter&logoColor=white&style=for-the-badge)](https://github.com/ThomasAko/physics-constraints-qm7-qm9/blob/main/Physics_Constraints_QM7_to_QM9.ipynb)
[![Diagnosis notebook](https://img.shields.io/badge/Diagnosis-Notebook-5f6b7a?logo=jupyter&logoColor=white&style=for-the-badge)](https://github.com/ThomasAko/physics-constraints-qm7-qm9/blob/main/QM7_DeepChem_Diagnosis.ipynb)
[![Repository](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white&style=for-the-badge)](https://github.com/ThomasAko/physics-constraints-qm7-qm9)

:alt: Coulomb matrix heatmap for one molecule in QM7
:width: 420px
:align: center  

## Why it matters

Density functional theory (DFT) computes molecular energies accurately, but its cost grows steeply with system size. A machine-learning surrogate trained on DFT results can screen candidate molecules orders of magnitude faster. The open question is whether physics can be built into such a surrogate, so that its predictions are physically consistent as well as accurate.

## Setup

- **Data:** QM7, 7,165 small organic molecules (up to 23 atoms) with DFT atomization energies, loaded from the original `qm7.mat` and split at random 80/10/10.
- **Representation:** the **Coulomb matrix**. The diagonal holds the atomic self-energies (0.5·Z^2.4) and the off-diagonal entries the pairwise nuclear repulsion (ZᵢZⱼ/|Rᵢ−Rⱼ|). Each 23 × 23 matrix is flattened to 529 standardized features.
- **Models:** linear and random forest baselines on atom counts, random forests and dense neural networks on the Coulomb matrix, and a network with a Virial theorem penalty.

## 1. A broken data pipeline

The first version loaded QM7 through DeepChem's `load_qm7` and classified molecules as high or low energy. Every model, from random forests to a network found by a two-hour architecture search, stalled near 60% accuracy.

A composition only sanity check exposed the cause. Atomization energy is close to *extensive*: it is roughly a sum of contributions from each atom, so atom counts should predict it almost perfectly.

| Diagnostic | Original `qm7.mat` | DeepChem data |
|---|---|---|
| R² of a linear model on atom counts | 0.992 | 0.189 |
| Correlation of molecule size with energy | −0.971 | −0.267 |
| Smallest molecule | 4 atoms, at least one carbon | 1–2 hydrogen atoms only |

The features and labels did not match. On the original file, the same models with the same settings jump from about 60% to 98% accuracy.

## 2. From classification to regression

At 98%, classification can no longer tell models apart: atom counts alone reach 97.4%, and the remaining errors sit a median of 5.8 kcal/mol from the threshold, against 162 kcal/mol for a typical molecule. The study therefore predicts the energy itself and reports the mean absolute error (MAE) on 717 test molecules.

```{image} ../_static/figs/qm7/results_regression.png
:alt: Test MAE by model for QM7 atomization energy regression
:width: 680px
:align: center
```

| Model | MAE (kcal/mol) | RMSE (kcal/mol) |
|---|---|---|
| Mean predictor | 180.6 | 222.5 |
| Linear model, atom counts | 16.3 | 21.1 |
| Random forest, atom counts | 15.3 | 20.7 |
| Random forest, Coulomb matrix | 11.9 | 18.3 |
| Neural network (dropout, fixed learning rate) | 21.5 | 29.3 |
| Neural network (no dropout, learning-rate schedule) | 15.2 | 19.7 |
| **Random forest, sorted Coulomb matrix** | **10.5** | **15.6** |
| Neural network, sorted Coulomb matrix | 12.7 | 16.6 |
| Neural network + virial penalty, sorted | 13.9 | 18.3 |

- **Settings carried over from classification broke the network.** With dropout and a fixed learning rate it did worse than a five-coefficient linear model. Dropout adds noise straight to the predicted value, which classification tolerates and precise regression does not.
- **Atom ordering was a handicap.** Sorting each matrix by row norm, so the same position means the same kind of atom across molecules, cut the network's error by 17% and the random forest's by 11%. Only then did the network beat composition alone.

## 3. The Virial theorem penalty

For a molecule at equilibrium, 2⟨T⟩ + ⟨V⟩ = 0. I added two output heads for ⟨T⟩ and ⟨V⟩ and penalized violations of the theorem:

*Loss* = MSE(energy) + λ · mean(2T + V)², with λ = 0.1

**It gave no gain.** Across four runs, the virial network never beat the plain one; it was 0.2–1.5 kcal/mol worse.

**It cannot help.** QM7 has no labels for ⟨T⟩ or ⟨V⟩, so nothing ties the heads to real values. Setting the ⟨V⟩ head to −2 times the ⟨T⟩ head makes the residual vanish for every molecule, whatever the network has learned. In the notebook, rewriting the trained head this way drives the residual to about 10⁻⁷, float32 rounding, without touching the energy prediction. Small residuals are therefore not evidence of physical consistency.

## What I took away

- **Sanity-check the data before the models.** One composition-only baseline exposed a pipeline that had silently capped every result near 60%.
- **Choose a task that can separate models.** Once the data was right, classification was too easy to measure anything.
- **Training settings and representation mattered more than architecture.** The two-hour architecture search bought nothing; removing dropout and sorting the inputs mattered most.
- **A useful physics term must be anchored to data and impossible to satisfy trivially.** The Virial penalty met neither requirement.
- **Single runs are not evidence.** GPU training varied by about 1 kcal/mol from run to run, the same size as the Virial network's apparent effect.

[Notebook](https://github.com/ThomasAko/physics-constraints-qm7-qm9/blob/main/Physics_Constraints_QM7_to_QM9.ipynb) · [Diagnosis notebook](https://github.com/ThomasAko/physics-constraints-qm7-qm9/blob/main/QM7_DeepChem_Diagnosis.ipynb) · [Repository](https://github.com/ThomasAko/physics-constraints-qm7-qm9)

## What I'd do next

- **Test a physics term that meets both requirements.** The energy loss of *Energy Loss Functions for Physical Systems* (NeurIPS 2025, [arXiv:2511.02087](https://arxiv.org/abs/2511.02087)) charges a predicted geometry the energy needed to deform it into the reference. It is anchored to the true interatomic distances and is zero only for the correct geometry. QM9 geometry relaxation with an equivariant graph network is the natural test.
- **Put confidence intervals on the network results** with a multi-seed comparison.

**Tools:** Python, TensorFlow/Keras, scikit-learn, NumPy/SciPy, DeepChem and RDKit (diagnosis), Google Colab
