# Installation

Platform is built with **CMake + Make**, and its dependencies are managed
with **Conan**. It compiles on **Linux** and **macOS**.

## 1. Prerequisites

Install these before building:

| Dependency | Why | Notes |
|------------|-----|-------|
| **Conan** (2.x) | Installs libtorch, nlohmann_json, Catch2, CLI11, libxlsxwriter, … | `pip install conan` |
| **CMake / Make / C++20 compiler** | Build system | `g++` ≥ 11 or `Apple clang` ≥ 14 |
| **MPI** (OpenMPI or MPICH) | Parallel grid search (`b_grid search/experiment`) | Linux: `openmpi` + `openmpi-devel`; macOS: `brew install mpich` |
| **Boost** | Python integration and utilities | Linux: `dnf install boost-devel`; or set `BOOST_ROOT` |
| **Miniconda** *(optional)* | Python-backed classifiers (`STree`, `Odte`, `SVC`, `RandomForest`, `XGBoost`, `AdaBoostPy`) | Only needed if you use those models |
| **Graphviz `dot`** *(optional)* | Render the `.dot` model graphs to images | `brew install graphviz` / `apt install graphviz` |

### Miniconda

To run the Python classifiers (STree, ODTE, SVC, etc.) a Miniconda
installation is required. Install it from
[conda.io](https://docs.conda.io/en/latest/miniconda.html), preferably in
your home folder.

> **Linux note:** if the `b_xxxx` executables fail with
> `libstdc++.so.6: version 'GLIBCXX_3.4.32' not found (required by b_xxxx)`,
> the `libstdc++` shipped inside Miniconda is shadowing the system one.
> Delete the `libstdc++` library from the Miniconda installation; no
> recompilation is needed.

### MPI

* **Linux:** install `openmpi` and `openmpi-devel`. If your distro uses
  modulefiles, load them with `module load mpi/openmpi-x86_64`. If CMake
  cannot find OpenMPI (e.g. Oracle Linux), export
  `MPI_HOME=/usr/lib64/openmpi`.
* **macOS:** `brew install mpich`. If CMake doesn't find it, edit the
  `mpicxx` wrapper and remove `,-commons,use_dylibs` from `final_ldflags`:
  `vi /opt/homebrew/bin/mpicxx`.

### Boost

The easiest option is your distribution's package
(`sudo dnf install boost-devel`). If you installed Boost from a compressed
archive instead, point the build at it:

```bash
export BOOST_ROOT=/path/to/boost-library/
```

If you built Boost from source with `./bootstrap.sh --prefix=... && ./b2
install`, set `BOOST_ROOT` to the `prefix` you chose. Add the `export`
line to `~/.bashrc` (or equivalent) to make it permanent.

## 2. Build

All targets are Makefile targets; `make help` lists them.

```bash
make help          # show all targets
make init          # conan install for both Release and Debug
make debug         # configure + build build_Debug/ (testing enabled)
make release       # configure + build build_Release/
make install       # build Release, copy the b_* binaries to ~/bin (default)
make clean         # remove build directories
```

* `make debug` / `make release` **recreate** their build folder from scratch
  (`build_Debug/`, `build_Release/`).
* `make install` accepts a custom destination: `make install dest=/opt/bin`.
* The build is parallelized automatically (CPU count minus a small margin).

The executables are produced in `build_<type>/src/`:

```
build_Debug/src/b_main
build_Debug/src/b_grid
build_Debug/src/b_best
build_Debug/src/b_list
build_Debug/src/b_manage
build_Debug/src/b_results
```

`make install` copies them to `~/bin` by default, so you can call them by
name from anywhere.

## 3. Tests and coverage

The unit tests use **Catch2** and are built as
`build_Debug/tests/unit_tests_platform`.

```bash
make test                        # build debug, run all tests
make test opt="-s"               # verbose test output
make test opt="-c='Test Name'"   # run a single test section
make coverage                    # run tests + generate a coverage report
```

The coverage report (gcovr) is written under `build_Debug/` — open
`build_Debug/coverage/index.html` (or the gcovr XML/HTML output) in a browser.

## 4. Extra targets

| Target | What it does |
|--------|--------------|
| `make example` | Builds and runs `PlatformSample` (BoostAODE on iris, discretized, stratified). |
| `make dependency` | Creates a CMake dependency graph (`build_Debug/dependency.png`). |
| `make clang-uml` | Generates UML class/sequence diagrams from `.clang-uml`. |
| `make setup` | Installs `gcovr`/`lcov` (needed for coverage). |

## 5. Verify the installation

```bash
b_list --version     # or any other binary, e.g. b_main --version
```

Expect something like:

```
b_list, version 1.0.0
```

If the command runs and `b_list datasets` (inside an experiments folder with
a `datasets/all.txt`) prints your dataset table, the installation is working.
See [configuration.md](configuration.md) to set up the experiments folder.
