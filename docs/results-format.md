# Results: file naming and JSON format

This page documents the files Platform reads and writes, so you can script
against them, move them between machines, or fix them by hand.

## 1. File naming

| File | Where | Meaning |
|------|-------|---------|
| `results_<score>_<model>_<platform>_<YYYY-MM-DD>_<HH:MM:SS>_<n>_<rand>.json` | `results/` | One experiment run (one or more datasets). |
| `best_results_<score>_<model>.json` | `results/` | Per-dataset best run for a model (created by `b_best`, consumed by `--hyper-best`). |
| `grid_<model>_input.json` | `grid/` | Your hyperparameter search space (you create it). |
| `grid_<model>_output.json` | `grid/` | Grid-search results (created by `b_grid search`). |
| `results` folder `…_graph_<dataset>_<i>.dot` | `graphs/` | Graphviz network per dataset/fold (`b_main --graph`). |
| `datasets_experiment/<dataset>_<disc>_<strat>_<seed>_<fold>.json` | `datasets_experiment/` | Exact train/test splits (`b_main --generate-fold-files`). |
| `results.tex`, `results.md`, `post_hoc.tex`, `post_hoc.md` | `tex/` | `b_best --tex` exports. |
| `BestResults_<score>.xlsx`, `datasets.xlsx`, `some_results.xlsx` | `excel/` | Excel exports. |

`<disc>` is `disc_` (discretized) or `ndisc_`, `<strat>` is `strat_` or
`nstrat_`, and `<rand>` is a short random suffix that keeps file names
unique.

## 2. Result file — schema 1.0

A result file is a JSON object with `schema_version: "1.0"`. Top-level
fields (all required unless marked optional):

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | Always `"1.0"`. |
| `date` | string (`YYYY-MM-DD`) | Run date. |
| `time` | string (`HH:MM:SS`) | Run time. |
| `title` | string | Experiment title. |
| `language` | string | Implementation language (e.g. `c++`). |
| `language_version` | string | Compiler/library version. |
| `model` | string | Classifier name. |
| `platform` | string | Machine identifier from `.env`. |
| `score_name` | string | `accuracy` or `roc-auc-ovr`. |
| `version` | string | Platform version. |
| `folds` | int | Number of cross-validation folds. |
| `stratified` | bool | Whether stratified CV was used. |
| `discretized` | bool | Whether input was discretized. |
| `discretization_algorithm` | string | e.g. `mdlp` (optional, default `""`). |
| `smooth_strategy` | string | `ORIGINAL`/`LAPLACE`/`CESTNIK` (optional, default `ORIGINAL`). |
| `duration` | number | Total wall time in seconds. |
| `results` | array | One entry **per dataset** (see below). |

### Per-dataset entry (`results[i]`)

| Field | Type | Description |
|-------|------|-------------|
| `dataset` | string | Dataset name. |
| `samples` / `features` / `classes` | int | Dataset size. |
| `scores_train`, `scores_test` | number[] | Per-fold scores (length = `folds` × `seeds`). |
| `times_train`, `times_test` | number[] | Per-fold times. |
| `score`, `score_train` | number | Mean score over all folds. |
| `score_std`, `score_train_std` | number | Standard deviation of the scores. |
| `train_time`, `test_time` (+ `_std`) | number | Mean (± std) train/test time. |
| `time`, `time_std` | number | Total time statistics. |
| `hyperparameters` | object | The hyperparameters used (number/string values). |
| `nodes`, `leaves`, `depth` | number | Mean network-structure statistics (headers configurable in `.env`). |
| `notes` | string[] | Free-form notes (optional). |
| `graph` | string[] | Graphviz DOT source per fold (optional; used by `--graph`). |
| `confusion_matrices` | object[] | Per-fold confusion matrices (optional). |

Minimal example (abridged):

```json
{
  "date": "2026-09-28",
  "time": "13:45:44",
  "title": "Test iris TAN 5 folds",
  "model": "TAN",
  "platform": "MacBookpro16",
  "language": "c++",
  "language_version": "Apple clang",
  "score_name": "accuracy",
  "version": "1.0",
  "folds": 5,
  "stratified": false,
  "discretized": true,
  "discretization_algorithm": "mdlp",
  "smooth_strategy": "ORIGINAL",
  "duration": 0.07,
  "schema_version": "1.0",
  "results": [
    {
      "dataset": "iris",
      "samples": 150,
      "features": 4,
      "classes": 3,
      "hyperparameters": {},
      "scores_train": [0.9917, 0.975, 0.975, 0.975, 0.975],
      "scores_test":  [0.9333, 0.9667, 0.9667, 1.0,    0.9667],
      "times_train":  [0.02,   0.001,  0.001,  0.001,  0.001],
      "times_test":   [0.001,  0.001,  0.001,  0.001,  0.001],
      "train_time": 0.0049, "train_time_std": 0.0076,
      "test_time":  0.0015, "test_time_std":  0.0006,
      "score": 0.9667, "score_std": 0.0236,
      "score_train": 0.9783, "score_train_std": 0.0081,
      "time": 0.0063, "time_std": 0.0042,
      "nodes": 5.0, "leaves": 7.0, "depth": 16.6
    }
  ]
}
```

Use `b_results` to validate any file against this schema
(see [b_results.md](b_results.md)).

## 3. `best_results_<score>_<model>.json`

Maps each dataset to its best run:

```json
{
  "iris": {
    "score": 0.9533,
    "file": "results_accuracy_KDBLd_HugeFedora_2025-06-27_11:01:15_1.json",
    "hyperparameters": { "k": 3, "ld_proposed_cuts": 3 }
  },
  "adult": { "…": "…" }
}
```

* Created/refreshed by `b_best -m <model>`.
* Consumed by `b_main --hyper-best` and `b_grid experiment --hyper-best`,
  which look up the hyperparameters for each dataset they run.

## 4. Grid files

**Input** (`grid/grid_<model>_input.json`) — see
[b_grid.md](b_grid.md#1-the-input-file).

**Output** (`grid/grid_<model>_output.json`):

```json
{
  "model": "KDBLd",
  "score": "accuracy",
  "discretize": false,
  "stratified": false,
  "n_folds": 3,
  "seeds": [271],
  "date": "2026-09-28 13:48:47",
  "nested": 5,
  "platform": "MacBookpro16",
  "duration": "1.79 s",
  "results": {
    "iris": {
      "date": "2026-09-28 13:48:47",
      "grid": [ { "k": [3, 5, 7], "theta": [0.01, 0.03, 0.05] } ],
      "hyperparameters": { "k": 3, "theta": 0.03 },
      "score": 0.96
    }
  }
}
```

`b_main --hyper-file` accepts this same format, so a finished grid search
can be fed straight back into a normal experiment.

## 5. Fold files (`datasets_experiment/`)

`b_main --generate-fold-files` writes one JSON per dataset/seed/fold,
containing the exact train/test split that was used:

```
datasets_experiment/iris_disc_nstrat_271_0.json
```

Each file holds `seed`, `nfold`, and the `X_train` / `X_test` (and labels)
arrays. Use them to reproduce a run exactly in another tool.
