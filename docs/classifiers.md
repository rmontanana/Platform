# Classifiers

Platform ships a catalog of classifiers that can be selected with
`-m/--model` in `b_main`, `b_grid`, and `b_best`. The list is printed by any
`--help` that involves a model, e.g. `b_main --help`:

```
-m, --model    Model to use: {A2DE, AODE, AODELd, AdaBoost, AdaBoostPy,
  BoostA2DE, BoostAODE, DecisionTree, KDB, KDBLd, Odte, RandomForest, SPODE,
  SPODELd, SPnDE, STree, SVC, TAN, TANLd, XA1DE, XA2DE, XBA2DE, XBAODE,
  XGBoost, XSP2DE, XSPODE}
```

All classifiers implement the same interface (`bayesnet::BaseClassifier`)
and are instantiated by name through the factory in `src/main/Models.cpp`
(registration happens in `src/main/modelRegister.h`).

## Families

### Bayesian-network classifiers (C++)

| Model | Family |
|-------|--------|
| `TAN` | Tree-Augmented Naive Bayes. |
| `TANLd` | TAN with a **l**earning **d**eferral / likelihood-distance structure. |
| `SPODE` | Super-Parent ODE (order 2). |
| `SPnDE` | SP-nDE (super-parent, configurable order). |
| `SPODELd` | SPODE with likelihood-distance variant. |
| `KDB` | Kernel Density estimator over a Bayesian network (uses `k`, `theta`). |
| `KDBLd` | KDB with likelihood-distance variant (uses `k`, `theta`, `ld_*`). |
| `XSPODE` / `XSP2DE` | Experimental SPODE / SP2DE variants. |

### AODE / ensemble family (C++)

Average-One-Dependent-Estimator and its boosted/cross variants:

| Model | Family |
|-------|--------|
| `AODE` | Average One-Dependent Estimator. |
| `A2DE` | A2DE (two-attribute dependence). |
| `AODELd` | AODE with likelihood-distance variant. |
| `BoostAODE` | Boosted AODE (feature selection, ordering, tolerance, …). |
| `BoostA2DE` | Boosted A2DE. |
| `XBAODE` / `XBA2DE` | Extended boosted AODE / A2DE. |
| `XA1DE` / `XA2DE` | Extended A1DE / A2DE ensembles. |

### Trees and boosting

| Model | Family |
|-------|--------|
| `DecisionTree` | Decision-tree classifier (experimental C++ implementation). |
| `AdaBoost` | AdaBoost (C++). |
| `AdaBoostPy` | AdaBoost (Python/Miniconda wrapper). |

### Python classifiers (require Miniconda)

These are wrappers around Python implementations and need a working
Miniconda environment (see [installation.md](installation.md#1-prerequisites)):

| Model | Notes |
|-------|-------|
| `STree` | Stochastic tree. |
| `Odte` | ODTE. |
| `SVC` | Support Vector Classifier (scikit-learn). |
| `RandomForest` | scikit-learn random forest. |
| `XGBoost` | XGBoost gradient boosting. |

> If you try to run one of the Python models without Miniconda configured,
> the wrapper fails to launch — set it up first.

## Hyperparameters

Hyperparameters are passed with `b_main --hyperparameters '<json>'` or
discovered by `b_grid`. The accepted keys depend on the model. A few
examples observed in real runs:

* **KDB / KDBLd**

  ```json
  { "k": 5, "theta": 0.02 }
  ```

  `k` = number of neighbors; `theta` = kernel width. The `Ld` variants also
  accept `ld_algorithm` (`BINU`, `BINQ`, …) and `ld_proposed_cuts`.

* **BoostAODE / BoostA2DE** (from a real grid file)

  ```json
  {
    "alpha_block": false,
    "block_update": true,
    "order": "asc",
    "maxTolerance": 3,
    "select_features": "CFS",
    "threshold": 1e-7
  }
  ```

  `select_features` can be `CFS`, `FCBF`, or `IWSS`; `order` one of
  `asc`, `desc`, `rand`.

* **AODE family with discretization** often exposes `mdlp_proposed_cuts`
  (number of MDLP cut points) and the smoothing/ld options.

To see exactly which hyperparameters a model accepts, run a small grid
search (`b_grid dump` / `b_grid search`) or inspect the model's
`setHyperparameters` implementation. Unknown keys are ignored by most
models rather than rejected, so a wrong key usually just has no effect.

## Choosing a smoothing strategy

For Bayesian-network models, node initialization uses a smoothing strategy
selected with `--smooth-strat` (or `smooth_strat` in `.env`):

| Value | Meaning |
|-------|---------|
| `ORIGINAL` | The platform's default smoothing. |
| `LAPLACE` | Add-one (Laplace) smoothing. |
| `CESTNIK` | Cestnik smoothing. |
