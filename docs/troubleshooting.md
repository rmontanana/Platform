# Troubleshooting

Common errors, what they mean, and how to fix them.

## Configuration and setup

### `File .env not found`

You are running Platform in a folder without a `.env` file. Create one (see
[configuration.md](configuration.md)) or `cd` into your experiments folder.

### `Unknown source.`

`source_data` in `.env` is not one of `Arff`, `Tanveer`, `Surcov`,
`CsvJSON`, `Test`.

### `csv_json_path is required in .env when source_data=CsvJSON`

Add a `csv_json_path=/path/to/csv+json` line pointing at the folder with
your `*.csv` and `*_metadata.json` files.

### `Unable to open catalog file. [datasets/all.txt]`

`datasets/all.txt` is missing or the `datasets/` folder is not in the
current directory. Check [configuration.md](configuration.md#2-the-datasets-folder).

### `Invalid catalog file format.`

A line in `all.txt` does not have 1–3 semicolon-separated fields. Expected
format: `<name>;<class>;<real_features>`.

### `Dataset must be one of: {…}` / `Model must be one of {…}`

Typo in a dataset or model name. The error message lists every valid value.
Python models (`STree`, `Odte`, `SVC`, `RandomForest`, `XGBoost`,
`AdaBoostPy`) are only registered when the build linked the Python
wrappers.

## Runtime errors

### `std::invalid_argument: dataset (X, y) must be of type Integer`

The classifier needs **discrete/integer** input but got floating-point
values. Typical for `TAN` and other discrete Bayesian networks on
continuous data.

**Fix:** add `--discretize` (or set `discretize=1` in `.env`):

```bash
b_main -d iris -m TAN -f 5 --discretize
```

### `X must be a floating point tensor`

The opposite case: the classifier (e.g. `KDBLd`) needs **continuous** input
but the data was discretized.

**Fix:** remove `--discretize` / set `discretize=0` in `.env`.

### `Filename is not set. Use save() method to generate a filename.`

`b_main --graph` without `--save`. Graph files are named after the result
file, so the result must be saved:

```bash
b_main -d iris -m TAN -f 5 --discretize --save --graph
```

### `Can't make the Friedman test with less than 3 models and/or less than 3 datasets.`

The Friedman test in `b_best --friedman` needs at least three models **and**
three datasets, each model having results for **every** dataset. Run the
missing experiments, or drop `--friedman`.

### `key '<dataset>' not found` (during `b_best --friedman`)

Same root cause as above: some model is missing results for a dataset, so
the statistics matrix cannot be built.

### `Friedman test can only be used with all models and all the datasets`

You combined `--friedman` with `-m` or `-d` filters. Re-run with
`-m any -d any`.

### `Number of folds must be greater than 1` / `Number of nested folds must be greater than 1`

`-f/--folds` and `--nested` must be integers ≥ 2.

### Python models fail to start

`STree`, `Odte`, `SVC`, `RandomForest`, `XGBoost`, `AdaBoostPy` need
Miniconda. Check the Python environment and the `libstdc++` shadowing
problem on Linux (see [installation.md](installation.md#1-prerequisites)).

### Linux: `libstdc++.so.6: version 'GLIBCXX_3.4.32' not found`

Miniconda's `libstdc++` is being loaded instead of the system one. Delete
the `libstdc++` library from the Miniconda installation — no rebuild needed.

## `b_grid` / MPI

### `b_grid` refuses to run without `mpirun`

`search` and `experiment` are MPI programs. Launch them with at least two
processes:

```bash
mpirun -np 4 b_grid search -m <model>
```

(One process works technically but defeats the purpose; `np=1` may also be
rejected depending on the MPI build.)

### `prterun detected that one or more processes exited with non-zero status`

A worker crashed — usually a data/model mismatch (see the tensor-type errors
above) or a dataset listed in the grid input that is not in `all.txt`.
Run without MPI (or with `--only` on one small dataset) to see the real
error message.

### Search results in the output file look stale

`b_grid search` **overwrites** `grid/grid_<model>_output.json`. Back it up
first (e.g. `cp grid/grid_M_output.json grid/grid_M_output.bak`) if you need
the previous run — `--continue <dataset>` resumes from it instead.

## `b_manage`

### `Make screen bigger to fit the results! N columns needed!`

The terminal window is narrower than the table. Enlarge the window (or use
a bigger font/monitor) and re-run.

### The list is empty

* No `results_*.json` files in the folder, or
* your filters (`-m`, `-s`, `--platform`, `--complete`, `--partial`) exclude
  everything. Re-run with no filters.

## Results and reporting

### `b_results` reports errors in files from an older Platform

Run `b_results --fix`; if a file still fails, re-run the experiment (old
files may lack required fields) or delete it if it is obsolete.

### `b_best` shows `N/A` in some table cells

That model simply has no saved result for that dataset yet — run it with
`b_main` first.

### Excel export doesn't open

The file is created in `excel/` (e.g. `excel/BestResults_accuracy.xlsx`)
and Platform calls the OS "open" handler. If no spreadsheet application is
associated, open the file manually from `excel/`.

## Still stuck?

* Re-run the failing command with `--help` to check the option names.
* Reproduce on a single small dataset (`-d iris -f 5 --quiet`) to isolate
  the problem.
* Check `b_results` to make sure the result files you depend on are valid.
