#!/bin/bash
# Parity test of b_main and b_grid experiment (Platform#8, Platform#9).
#
# Runs both programs with the same arguments on the test datasets and checks that the
# saved results are identical field by field, except dates and times.
#
# Usage: tests/parity/run_parity.sh [bin_dir]     (default: build_Debug/src)
# Needs mpirun. Exit code 0 when every case is identical.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
BIN="$(cd "${1:-$root/build_Debug/src}" && pwd)"
work="$(mktemp -d -t platform-parity)"
trap 'rm -rf "$work"' EXIT
cd "$work"
# source_data=Test reads the arff files of tests/data (path compiled into the binaries)
cat > .env <<'EOF'
experiment=discretiz
score=accuracy
platform=Parity
n_folds=5
stratified=0
model=TAN
source_data=Test
seeds=[42, 123, 456, 789, 1024]
discretize=0
ignore_nan=0
nodes=Nodes
leaves=Edges
depth=States
fit_features=0
margin=0.1
framework=bulma
discretize_algo=pkilog
smooth_strat=ORIGINAL
EOF
failed=0
run_case() {
    local name=$1; shift
    mkdir -p "$name/main" "$name/grid"
    if ! "$BIN/b_main" "$@" --quiet --save --title "parity $name" --folder "$name/main/" > "$name/main.log" 2>&1; then
        echo "FAILED $name: b_main"; tail -3 "$name/main.log"; failed=1; return
    fi
    if ! mpirun -np 3 "$BIN/b_grid" experiment "$@" --quiet --save --title "parity $name" --folder "$name/grid/" > "$name/grid.log" 2>&1; then
        echo "FAILED $name: b_grid"; tail -3 "$name/grid.log"; failed=1; return
    fi
    python3 "$here/compare.py" "$name"/main/*.json "$name"/grid/*.json "$name" || failed=1
}
# glass is left out: its line in tests/data/all.txt reads "glass;Type,all"
run_case boost-notes   -m XBAODE --datasets iris ecoli diabetes --discretize --stratified
run_case boost-pairs   -m XBA2DE --datasets iris ecoli diabetes --discretize --stratified
run_case no-train      -m XBAODE --datasets iris ecoli diabetes --discretize --stratified --no-train-score
run_case auc           -m TAN -s roc-auc-ovr --datasets iris ecoli diabetes --discretize --stratified
run_case continuous    -m TANLd --datasets iris ecoli --stratified
run_case graph         -m TAN --datasets iris --discretize --graph
# A failing task stops b_grid with a non-zero code and names the task, instead of hanging
if timeout 300 mpirun -np 3 "$BIN/b_grid" experiment -m XBA2DE --datasets iris --discretize --hyperparameters '{"beta": -1}' \
    --quiet --title failure --folder ./ > failure.log 2>&1; then
    echo "FAILED failure: b_grid ended with code 0"; failed=1
elif grep -q "b_grid: task .* failed on rank .*Invalid beta" failure.log; then
    echo "SAME failure: b_grid stops and names the task"
else
    echo "FAILED failure: no task message (or timeout)"; tail -5 failure.log; failed=1
fi
exit $failed
