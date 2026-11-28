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
| **Python 3 and network access** | The first build downloads and builds the Python 3.12 environment and Boost.Python (see below) | Any `python3` ≥ 3.6 to run the bootstrap scripts |
| **Graphviz `dot`** *(optional)* | Render the `.dot` model graphs to images | `brew install graphviz` / `apt install graphviz` |

Python 3.12, the Python classifiers and Boost are **not** prerequisites: the
Makefile builds them inside the repository.

### MPI

* **Linux:** install `openmpi` and `openmpi-devel`. If your distro uses
  modulefiles, load them with `module load mpi/openmpi-x86_64`. If CMake
  cannot find OpenMPI (e.g. Oracle Linux), export
  `MPI_HOME=/usr/lib64/openmpi`.
* **macOS:** `brew install mpich`. If CMake doesn't find it, edit the
  `mpicxx` wrapper and remove `,-commons,use_dylibs` from `final_ldflags`:
  `vi /opt/homebrew/bin/mpicxx`.

### Python and Boost for the Python classifiers

Platform uses PyClassifiers 1.1, which is built for one CPython minor version
and one exact Boost version; Platform has to be built and run with the same
pair. Since 1.1.7 both are pinned (CPython 3.12, Boost 1.92.0) and the Makefile
builds them itself the first time, inside the repository and without `sudo`:

| What | Where | How |
|------|-------|-----|
| CPython 3.12 with headers | `build_deps/python` | Standalone build installed by a pinned, checksummed [uv](https://docs.astral.sh/uv/) downloaded to `build_deps/uv` |
| Python environment | `.venv` | venv of that CPython with the packages pinned in `requirements.txt` (NumPy, scikit-learn, STree, Odte, XGBoost) |
| Boost.Python / Boost.NumPy 1.92.0 | `build_deps/boost` | Built from the official sources (checksum verified); log in `build_deps/logs` |

`make env` builds whatever is missing and every build target (`debug`,
`release`, `buildd`, `buildr`, `install`, `test`) runs it first, so
`make release && make install` is enough on a new machine. The first run takes
a few minutes (most of it compiling Boost.Python); later runs find everything
in place. The line `>>> Python .../.venv/bin/python (ABI 3.12), Boost 1.92.0`
shows what is being used.

The executables record where that Python and Boost are, so they run from
`~/bin` with no environment variables. Consequently `.venv` and `build_deps`
must stay where they were built: if you move or delete the repository, run
`make release && make install` again.

To use an existing pair instead of the pinned one, give both
`PYCLASSIFIERS_PYTHON` (an interpreter with the Python classifiers installed)
and `BOOST_ROOT` (the prefix of a Boost.Python built for that interpreter, with
its CMake config files); `BOOST_VERSION=x.y.z` if it cannot be read from
`$BOOST_ROOT/include/boost/version.hpp`. At run time `PYCLASSIFIERS_PYTHON`
still selects another interpreter of the same Python minor version.

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
