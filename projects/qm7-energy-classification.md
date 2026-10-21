# Physics constraints for molecular machine learning: from QM7 to QM9

**One question, followed through two datasets: does building physics into a neural network's loss make it learn better? Part 1 found a constraint that looked like it helped but couldn't actually bind. Part 2 tests one that can.**

| | Part 1: QM7 | Part 2: QM9 |
|---|---|---|
| Task | classify molecules as high or low atomization energy | relax a distorted geometry back to equilibrium |
| Input | flattened Coulomb matrix | atom types and 3D coordinates |
| Model | random forest, dense neural networks | E(n)-equivariant graph network |
| Physics term | virial theorem 2⟨T⟩ + ⟨V⟩ = 0 on two auxiliary heads | energy loss on interatomic distances (NeurIPS 2025), plus two variations of my own |
| Status | complete, with diagnosis | in progress |

## Part 1: QM7

```{image} ../_static/figs/qm7/coulomb_matrix.png
:alt: Coulomb matrix heatmap for one molecule in QM7
:width: 380px
:align: center
```

**Setup.** QM7 holds 7,165 small molecules with DFT atomization energies. Each molecule is encoded as a **Coulomb matrix**: the diagonal holds the atomic self-energies (0.5·Z^2.4) and the off-diagonal entries the nuclear repulsion between atom pairs (ZᵢZⱼ/|Rᵢ−Rⱼ|). I flattened each matrix into 529 standardized features and classified molecules as above or below the training-set median energy. DeepChem's loader keeps 6,838 of the 7,165 molecules (RDKit rejects about 330 charged species) and splits them **randomly** into train/validation/test sets of 5,470 / 684 / 684.

**Models.**

* a random forest;
* a funnel network, 529 → 256 → 128 → 64 → 32 → 2, with 179k parameters;
* a network found by Bayesian architecture search (412 trials): an unexpected *diamond* shape, 529 → 64 → 416 → 224 → 256 → 128 → 2, with 245k parameters.

**The physics term.** At equilibrium the virial theorem links kinetic and potential energy: 2⟨T⟩ + ⟨V⟩ = 0. I gave each network two extra output heads for ⟨T⟩ and ⟨V⟩ and added λ·mean(2T + V)² to the loss.

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
| Virial NN, funnel | 0.614 | 0.627 | 0.576 | 0.600 |
| Virial NN, tuned | 0.588 | 0.588 | 0.602 | 0.595 |

### What worked

* A clean, leak-free pipeline: labels from the training median only, and scaling fitted on training data.
* A useful negative result on capacity. The random forest beat networks with 180–245k parameters, which reached about 98% training accuracy but stalled near 60% on validation.

### What didn't

* **The gain from the physics term isn't robust.** It lifted the funnel network from 59.1% to 61.4% in one run, but in a later rerun the virial version scored slightly *below* the plain one (60.2% vs 60.7%).
* **The constraint can't bind.** Both auxiliary heads read the same hidden vector **h**, so 2T + V = (2w_T + w_V)·**h** + (2b_T + b_V). Setting w_V = −2w_T and b_V = −2b_T makes the residual exactly zero for every molecule, whatever **h** contains. QM7 has no labels for ⟨T⟩ or ⟨V⟩, so nothing anchors them to real values. The tight residual distribution below therefore looked like physical consistency but isn't evidence of it.

```{image} ../_static/figs/qm7/virial_residual.png
:alt: Distribution of virial residuals 2T + V on the test set
:width: 620px
:align: center
```

### What I learned

A physics term only helps if it is:

1. **anchored to data**, comparing predictions with something measured rather than relating unlabelled outputs to each other; and
2. **not trivially satisfiable**, with no shortcut that zeroes the penalty without learning the physics.

A third point is about representation: the model should respect the problem's symmetries by construction. A flattened Coulomb matrix changes when the atoms are renumbered, which Montavon et al. (2012) handled by randomly permuting the matrices during training.

## Bridge: why QM9 and the energy loss

[*Energy Loss Functions for Physical Systems*](https://arxiv.org/abs/2511.02087) (NeurIPS 2025) treats each reference structure as the bottom of an energy well and charges a prediction the energy needed to deform the reference into it:

E(ŷ, y) = Σᵢ<ⱼ ½ kᵢⱼ ( ‖yᵢ − yⱼ‖ − ‖ŷᵢ − ŷⱼ‖ )²

It satisfies both requirements from Part 1. It is anchored to the true interatomic distances, and the only way to zero it is to reproduce the reference geometry. Because it uses distances alone, it is also invariant to rotation and translation without any alignment step.

It needs a model whose output is a **geometry**, which my QM7 setup didn't have. QM9 (about 130k molecules with 3D coordinates and bond lists) is the standard benchmark for geometry-aware models, and it's large enough to test the paper's claim that the gains are biggest with little data.

## Part 2: QM9 geometry relaxation (in progress)

**Task.** Displace the atoms of QM9 equilibrium geometries by Gaussian noise and train an E(n)-equivariant graph network to recover the relaxed structure.

**Losses compared:**

* coordinate MSE;
* MSE after optimal alignment;
* the paper's energy loss with distance-decay springs;
* **two variations of my own:**
  * **Coulomb-curvature springs** kᵢⱼ ∝ ZᵢZⱼ/d³ᵢⱼ, the second derivative of the same nuclear repulsion that defined the QM7 features;
  * **bond-graph springs**, stiff on bonded and angle-defining pairs and soft elsewhere, like a minimal force field.

**Two regimes.** In the *aligned* regime the noisy input keeps the reference orientation. In the *frame-free* regime the input is also randomly rotated, which mirrors the paper's argument that MSE breaks down when orientation is ambiguous.

**Evaluation**, deliberately independent of the training losses: aligned RMSD, bond-length and angle errors, whether covalent connectivity is recovered, and the MMFF force-field energy of each predicted geometry. Training sets of 1k, 10k and 50k molecules test the low-data claim.

```{note}
Results will be added here once the full QM9 runs are complete.
```

**Tools:** Python, DeepChem, RDKit, TensorFlow/Keras, Keras Tuner, scikit-learn, PyTorch, PyTorch Geometric, Google Colab
