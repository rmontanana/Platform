# AGENTS.md

This file provides guidance to agentic coding agents working with code in this repository.

## Build System

The project uses CMake with Make as the primary build system:

- **Release build**: `make release` (creates `build_Release/` directory)
- **Debug build**: `make debug` (creates `build_Debug/` directory with testing enabled)
- **Clean project**: `make clean` (removes build directories)
- **Initialize dependencies**: `make init` (runs conan install for both configurations)

### Testing

- **Run all tests**: `make test`
- **Run single test with verbose output**: `make test opt="-s"`
- **Run specific test section**: `make test opt="-c='Test Name'"`
- **Coverage report**: `make coverage` (generates HTML report in `build_Debug/coverage/index.html`)

Test executables are built from `tests/` and located in `build_Debug/tests/` after building.

### Build Targets

Main executables (built from `src/commands/`):
- `b_main`: Main experiment runner
- `b_grid`: Grid search over hyperparameters
- `b_best`: Best results analysis and comparison
- `b_list`: Dataset listing and properties
- `b_manage`: Results management interface
- `b_results`: Results processing

## Code Style Guidelines

### General

- **C++ Standard**: C++20
- **Namespace**: `platform` for main code, `bayesnet` for classifiers
- **Include guards**: Use `#ifndef NAME_H` / `#define NAME_H` / `#endif` pattern
- **File headers**: Include SPDX headers with copyright and license information

### Naming Conventions

- **Classes**: PascalCase (e.g., `Dataset`, `Experiment`, `DecisionTree`)
- **Functions**: PascalCase (e.g., `getFeatures()`, `computeStd()`)
- **Variables**: snake_case (e.g., `n_samples`, `max_depth`, `fileType`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_SIZE`)
- **Members**: snake_case with trailing underscore optional (e.g., `loaded_` or `loaded`)

### Types

- Use `int`, `long`, `float`, `double` for primitives
- Use `std::string` for strings
- Use `std::vector<T>`, `std::map<K,V>` for containers
- Use `torch::Tensor` for PyTorch tensors
- Use `std::shared_ptr` for ownership transfer
- Use `const` references for input parameters where appropriate

### Imports/Includes

- System headers: `<vector>`, `<string>`, `<iostream>`, etc.
- Project headers: `"common/Dataset.h"`, `"main/Experiment.h"`
- External libraries: `<bayesnet/BaseClassifier.h>`, `<torch/torch.h>`, `<nlohmann/json.hpp>`

Include order: 1) main header, 2) project headers, 3) external libraries

### Formatting

- **Indentation**: 4 spaces (no tabs)
- **Line length**: 100 characters maximum
- **Braces**: K&R style (opening brace on same line as control statement)
- **Spaces**: Around operators (`a + b`), after commas, after control keywords

### Error Handling

- Use `std::invalid_argument` for invalid parameters
- Use `std::runtime_error` for runtime errors
- Check dataset loading before operations: throw if not loaded
- Use `exit(1)` for critical initialization errors in main executables

### Special Patterns

- **Factory pattern**: Use `Models::instance()->create()` for classifier instantiation
- **Automatic registration**: Add registrar in `modelRegister.h` for new classifiers
- **Singleton pattern**: `Models` uses static instance pattern with deleted copy operations

## Testing

- **Framework**: Catch2
- **Test location**: `tests/Test*.cpp`
- **Test naming**: `Test[Component]` pattern
- **Test sections**: Use `TEST_CASE("[Category]")` with descriptive names
- **Assertions**: `REQUIRE()`, `CHECK()`, `REQUIRE_FALSE()`

## Configuration Files

- `.clang-tidy`: Static analysis with C++ Core Guidelines
- `Makefile`: Build automation
- `CMakeLists.txt`: CMake configuration
- `.env`: Experiment configuration (not in repo)

## Key Directories

- `src/main/`: Core experiment framework
- `src/common/`: Utility classes (Dataset, Datasets, Discretization)
- `src/experimental_clfs/`: Custom classifier implementations
- `src/grid/`: Hyperparameter grid search
- `src/results/`: Result storage and processing
- `src/reports/`: Report generation (console, Excel)
- `tests/`: Unit tests

## Dependencies

Managed via Conan:
- **libtorch**: PyTorch C++ backend
- **nlohmann_json**: JSON processing
- **Catch2**: Unit testing
- **CLI11/argparse**: Command-line parsing
- **BayesNet**: Core Bayesian network classifiers (external)
- **libxlsxwriter**: Excel file generation