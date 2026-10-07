# Configuration

All Platform commands are driven by two things that live **in the directory
you run them from**:

1. a `.env` configuration file, and
2. a `datasets/` folder (or another source of data, see below).

Commands create the other folders (`results/`, `grid/`, `graphs/`, `excel/`,
`tex/`, `hidden_results/`, `datasets_experiment/`) automatically when needed.
A repository-root `.env.example` is provided as a template.

## 1. The `.env` file

`DotEnv` reads `.env` from the current working directory. If it is missing,
the program stops with:

```
File .env not found
```

Copy `.env.example` to `.env` and adapt it:

```bash
cp .env.example .env
```

### Keys

| Key | Valid values | Default* | Meaning |
|-----|--------------|----------|---------|
| `experiment` | `discretiz`, `odte`, `covid`, `Test` | — | Experiment name/type (used in titles and bookkeeping). |
| `score` | `accuracy`, `roc-auc-ovr` | `accuracy` | Primary evaluation metric for `b_main`/`b_best`. |
| `platform` | any string | — | System identifier recorded in every result (e.g. `MacBookpro16`). |
| `n_folds` | `5`, `10` | `5` | Number of cross-validation folds. |
| `stratified` | `0`, `1` | `0` | Use stratified K-fold cross validation. |
| `model` | any registered model | `TAN` | Default classifier name. |
| `source_data` | `Arff`, `Tanveer`, `Surcov`, `CsvJSON`, `Test` | `Arff` | Where and how datasets are loaded (see below). |
| `csv_json_path` | any path (optional) | — | **Required** when `source_data=CsvJSON`: folder with `*.csv` + `*_metadata.json`. |
| `seeds` | JSON list of ints, e.g. `[271]` | `[271]` | Random seeds for reproducibility. |
| `discretize` | `0`, `1` | `0` | Discretize the dataset before training. |
| `discretize_algo` | `mdlp`, `mdlp3`, `mdlp4`, `mdlp5`, `pkisqrt`, `pkilog`, `bin3u`…`bin10q/u` (see table) | `mdlp` | Discretization algorithm. |
| `ignore_nan` | `0`, `1` | `0` | Ignore NaN values when loading. |
| `smooth_strat` | `ORIGINAL`, `LAPLACE`, `CESTNIK` | `ORIGINAL` | Smoothing strategy for Bayes-network node initialization. |
| `nodes` | any string | `Nodes` | Column header used for the *nodes* statistic in reports. |
| `leaves` | any string | `Edges` | Column header for the *leaves/edges* statistic. |
| `depth` | any string | `States` | Column header for the *depth/states* statistic. |
| `fit_features` | `0`, `1` | `0` | Fit on features (model-dependent behavior). |
| `framework` | `bulma`, `bootstrap` | `bulma` | CSS framework for generated HTML reports. |
| `margin` | `0.1`, `0.2`, `0.3` | `0.1` | Margin used by some experimental classifiers. |

\* Defaults in *bold* are the values in `.env.example`; the rest are
enforced valid values in `DotEnv.h`.

### Discretization algorithms

MDLP variants (minimum description length, with different numbers of
iterations):

```
mdlp, mdlp3, mdlp4, mdlp5
```

PKI variants (parametric, using different root functions):

```
pkisqrt, pkilog
```

Binary binning with *n* bins, uniform (`u`) or quantile (`q`) cuts:

```
bin3u  bin3q  bin4u  bin4q  bin5u  bin5q  bin6u  bin6q  bin7u  bin7q  bin8u  bin8q  bin9u  bin9q  bin10u  bin10q
```

### Example

```ini
experiment=discretiz
score=accuracy
platform=MacBookpro16
n_folds=5
stratified=0
model=TAN
source_data=Arff
csv_json_path=
seeds=[271]
discretize=0
ignore_nan=0
nodes=Nodes
leaves=Edges
depth=States
fit_features=0
framework=bulma
margin=0.1
discretize_algo=mdlp
smooth_strat=ORIGINAL
```

## 2. The datasets folder

With `source_data=Arff` (the default), Platform reads datasets from
`datasets/` in the current directory.

### `all.txt` — the dataset catalog

`datasets/all.txt` lists every dataset the platform knows about, one per
line, **semicolon-separated**:

```
<name>;<class_feature>;<real_features>
```

* `<name>` — dataset name (used by all commands; the ARFF file is
  `<name>.arff`).
* `<class_feature>` — name of the class attribute inside the ARFF file.
* `<real_features>` — which features are numeric (continuous):
  * `all` — every feature is real-valued,
  * `none` — no real-valued features,
  * a JSON list of 0-based feature indices, e.g. `[0,3,6,7]`.

Example:

```
adult;class;[0,2,4,11,12,13]
iris;class;all
hayes-roth;class;none
```

Lines starting with `#` and blank lines are ignored. Dataset names are
matched case-insensitively and sorted alphabetically in reports.

### ARFF files

Each dataset is a standard **ARFF** file (`datasets/<name>.arff`). Mixed
discrete/continuous data is supported; continuous features are the ones
declared in `all.txt` (and in the ARFF `@attribute` declarations).

## 3. Other data sources (`source_data`)

| `source_data` | Folder | Format |
|---------------|--------|--------|
| `Arff` | `datasets/` | ARFF + `all.txt` |
| `Surcov` | `datasets/` | CSV + `all.txt` |
| `Tanveer` | `data/` | R-data + `all.txt` |
| `CsvJSON` | `csv_json_path` (from `.env`) | one `*.csv` + `<name>_metadata.json` per dataset; the metadata JSON must contain at least `target_name` |
| `Test` | `tests/data/` of the *Platform* source tree | ARFF (used by the unit tests) |

For `CsvJSON`, each dataset is a pair of files in the folder given by
`csv_json_path`:

```
mydata.csv
mydata_metadata.json     # { "target_name": "class", ... feature types ... }
```

## 4. Project layout at runtime

Created automatically as needed (relative to the working directory):

```
.
├── .env                      # your configuration (you create this)
├── datasets/                 # input data (you create this)
├── results/                  # results_*.json and best_results_*.json
├── hidden_results/           # intermediate per-fold data
├── grid/                     # grid_<model>_input.json / grid_<model>_output.json
├── graphs/                   # Graphviz .dot files (b_main --graph)
├── excel/                    # exported .xlsx reports
├── tex/                      # results.tex, results.md, post_hoc.* (b_best --tex)
└── datasets_experiment/      # fold files (b_main --generate-fold-files)
```

## 5. Choosing datasets on the command line

Every command that needs a dataset validates the name against `all.txt`.
The valid choices are always printed in the help and in error messages, e.g.:

```
-d, --dataset    Dataset file name: {adult, balance-scale, breast-w, …, wine}
```

Passing an unknown name fails fast:

```
Dataset must be one of: {adult, balance-scale, …}
```
