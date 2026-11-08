"""Compares a b_main result with a b_grid experiment result, field by field.

Only times and dates may differ. Exits with code 1 on any other difference.
"""
import json
import sys

IGNORED_HEADER = {"date", "time", "duration"}
IGNORED_RESULT = {"times_train", "times_test", "train_time", "train_time_std", "test_time",
                       "test_time_std", "time", "time_std"}


def diff(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        if list(a) != list(b):
            out.append(f"{path}: keys {list(a)} != {list(b)}")
        for k in a:
            if k in b:
                diff(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out)
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")


def main(fa, fb, name):
    a, b = json.load(open(fa)), json.load(open(fb))
    out = []
    for j in (a, b):
        for k in IGNORED_HEADER:
            j.pop(k, None)
        for r in j["results"]:
            for k in IGNORED_RESULT:
                r.pop(k, None)
    diff(a, b, "$", out)
    n_notes = sum(len(r["notes"]) for r in a["results"])
    n_cm = sum(len(r.get("confusion_matrices", [])) for r in a["results"])
    n_cmt = sum(len(r.get("confusion_matrices_train", [])) for r in a["results"])
    summary = (f"{name}: {len(a['results'])} datasets, {n_notes} notes, {n_cm} test matrices, "
               f"{n_cmt} train, score_name={a['score_name']}")
    if out:
        print(f"DIFFERENT {summary}")
        for line in out[:40]:
            print("   ", line)
        sys.exit(1)
    print(f"SAME {summary}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
