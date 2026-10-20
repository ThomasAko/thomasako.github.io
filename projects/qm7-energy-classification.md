# Can a physics constraint help a neural network classify molecular energies?

**Classifying 7,165 small molecules as high or low atomization energy from their Coulomb matrices, and testing whether adding the virial theorem to the loss helps a neural network.**

```{image} ../_static/figs/qm7/coulomb_matrix.png
:alt: Coulomb matrix heatmap for one molecule in QM7
:width: 420px
:align: center
```

## Why it matters

Density functional theory (DFT) computes molecular energies accurately, but its cost grows steeply with system size. A machine-learning surrogate trained on DFT results can screen candidate molecules orders of magnitude faster. The open question is whether physics can be built into such a surrogate so its predictions are not only accurate but physically consistent.

## Setup

- **Data:** the QM7 benchmark (via DeepChem/MoleculeNet), with 7,165 molecules of up to 23 atoms.
- **Representation:** the **Coulomb matrix**. The diagonal holds the atomic self-energies (0.5·Z^2.4) and the off-diagonal entries hold the pairwise nuclear repulsion (ZᵢZⱼ/|Rᵢ−Rⱼ|). Each 23 × 23 matrix is flattened to 529 features and standardized.
- **Task:** classify each molecule as above or below the median atomization energy. The median comes from the training set only, so no test information leaks in.
- **Split:** DeepChem's own train/validation/test sets (5,470 / 684 / 684 molecules).

## Models

1. **Random forest** (100 trees), a non-parametric baseline.
2. **Funnel neural network**: 529 → 256 → 128 → 64 → 32 → 2, with dropout and 179k parameters.
3. **Tuned neural network.** A Bayesian search (Keras Tuner, 412 trials over about two hours) settled on an unexpected *diamond* shape, 529 → 64 → 416 → 224 → 256 → 128 → 2, with 245k parameters. Repeated searches returned different architectures, which hints at a rugged loss landscape.
4. **Physics-regularized versions of both networks.** I added two auxiliary output heads for kinetic energy ⟨T⟩ and potential energy ⟨V⟩, and penalized violations of the **virial theorem**, 2⟨T⟩ + ⟨V⟩ = 0:

   *Loss* = *cross-entropy* + λ · mean(2T + V)², with λ = 0.01

## Results (test set, 684 molecules)

```{image} ../_static/figs/qm7/results.png
:alt: Accuracy and F1 for the five models
:width: 620px
:align: center
```

| Model | Accuracy | Precision (high) | Recall (high) | F1 |
|---|---|---|---|---|
| **Random forest** | **0.627** | **0.644** | 0.579 | **0.610** |
| NN, funnel | 0.591 | 0.608 | 0.523 | 0.563 |
| NN, tuned | 0.597 | 0.604 | 0.573 | 0.588 |
| Physics-regularized NN, funnel | 0.614 | 0.627 | 0.576 | 0.600 |
| Physics-regularized NN, tuned | 0.588 | 0.588 | 0.602 | 0.595 |

## What I took away

- **The random forest beat every neural network.** With about 5,000 training molecules and a flattened representation, trees found the decision boundary more reliably than networks with 180–245k parameters. The networks reached about 98% training accuracy but about 60% on validation, a clear sign of overfitting.
- **The physics term gave the funnel network a small lift, from 59.1% to 61.4%, but I wouldn't over-read it.** The gain is within the run-to-run spread I saw when re-training. And because ⟨T⟩ and ⟨V⟩ have no ground-truth labels, the network can satisfy the virial constraint trivially, for example by driving both toward zero. The near-zero residuals below can't tell that apart from genuine physics. Here the physics term behaves more like a regularizer on the shared layers than a real physical constraint.

```{image} ../_static/figs/qm7/virial_residual.png
:alt: Distribution of virial residuals 2T + V on the test set
:width: 640px
:align: center
```

- **The representation is the bottleneck.** Flattening the Coulomb matrix throws away its 2-D structure and makes it depend on atom ordering. It also says nothing directly about orbital hybridization or electronic shielding.

## What I'd do next

- **Make the physics constraint bite:** supervise ⟨T⟩ and ⟨V⟩ with computed values, or move to **regression** of the energy itself, where a continuous physical law can actually shape the predictions.
- **Use a better representation:** sorted or eigenvalue Coulomb features, a CNN treating the matrix as an image, or a graph neural network.
- **Reduce and structure the inputs:** PCA to denoise the 529 features, and clustering to find chemical families before classifying.

**Tools:** Python, DeepChem, RDKit, TensorFlow/Keras, Keras Tuner, scikit-learn, Google Colab

```{note}
TODO: link the notebook.
```
