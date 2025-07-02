# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Platform is a C++ machine learning framework for running experiments with Bayesian Networks and other classifiers. It supports both research-focused experimental classifiers and production-ready models through a unified interface.

## Build System

The project uses CMake with Make as the primary build system:

- **Release build**: `make release` (creates `build_Release/` directory)
- **Debug build**: `make debug` (creates `build_Debug/` directory with testing and coverage enabled)
- **Install binaries**: `make install` (copies executables to `~/bin` by default)
- **Clean project**: `make clean` (removes build directories)
- **Initialize dependencies**: `make init` (runs conan install for both Release and Debug)

### Testing

- **Run tests**: `make test` (builds debug version and runs all tests)
- **Coverage report**: `make coverage` (runs tests and generates coverage with gcovr)
- **Single test with options**: `make test opt="-s"` (verbose) or `make test opt="-c='Test Name'"` (specific test)

### Build Targets

Main executables (built from `src/commands/`):
- `b_main`: Main experiment runner
- `b_grid`: Grid search over hyperparameters 
- `b_best`: Best results analysis and comparison
- `b_list`: Dataset listing and properties
- `b_manage`: Results management interface
- `b_results`: Results processing

## Dependencies

The project uses Conan for package management with these key dependencies:
- **libtorch**: PyTorch C++ backend for tensor operations
- **nlohmann_json**: JSON processing
- **catch2**: Unit testing framework
- **cli11**: Command-line argument parsing (replacement for argparse)

Custom dependencies (not available in ConanCenter):
- **fimdlp**: MDLP discretization library (needs manual integration)
- **folding**: Cross-validation utilities (needs manual integration)
- **arff-files**: ARFF dataset file handling (needs manual integration)

External dependencies (managed separately):
- **BayesNet**: Core Bayesian network classifiers (from `../lib/`)
- **PyClassifiers**: Python classifier wrappers (from `../lib/`)
- **MPI**: Message Passing Interface for parallel processing
- **Boost**: Python integration and utilities

**Note**: Some dependencies (fimdlp, folding, arff-files) are not available in ConanCenter and need to be:
- Built as custom Conan packages, or
- Integrated using CMake FetchContent, or
- Built separately and found via find_package

## Architecture

### Core Components

**Experiment Framework** (`src/main/`):
- `Experiment.cpp/h`: Main experiment orchestration
- `Models.cpp/h`: Classifier factory and registration system
- `Scores.cpp/h`: Performance metrics calculation
- `HyperParameters.cpp/h`: Parameter management
- `ArgumentsExperiment.cpp/h`: Command-line argument handling

**Data Handling** (`src/common/`):
- `Dataset.cpp/h`: Individual dataset representation
- `Datasets.cpp/h`: Dataset collection management
- `Discretization.cpp/h`: Data discretization utilities

**Classifiers** (`src/experimental_clfs/`):
- `AdaBoost.cpp/h`: Multi-class SAMME AdaBoost implementation
- `DecisionTree.cpp/h`: Decision tree base classifier
- `XA1DE.cpp/h`: Extended AODE variants
- Experimental implementations of Bayesian network classifiers

**Grid Search** (`src/grid/`):
- `GridSearch.cpp/h`: Hyperparameter optimization
- `GridExperiment.cpp/h`: Grid search experiment management
- Uses MPI for parallel hyperparameter evaluation

**Results & Reporting** (`src/results/`, `src/reports/`):
- JSON-based result storage with schema validation
- Excel export capabilities via libxlsxwriter
- Console and paginated result display

### Model Registration System

The framework uses a factory pattern with automatic registration:
- All classifiers inherit from `bayesnet::BaseClassifier`
- Registration happens in `src/main/modelRegister.h`
- Factory creates instances by string name via `Models::create()`

## Configuration

**Environment Configuration** (`.env` file):
- `experiment`: Experiment name/type
- `n_folds`: Cross-validation folds (default: 5)
- `seeds`: Random seeds for reproducibility
- `model`: Default classifier name
- `score`: Primary evaluation metric
- `platform`: System identifier for results

**Grid Search Configuration**:
- `grid_<model_name>_input.json`: Hyperparameter search space
- `grid_<model_name>_output.json`: Search results

## Data Format

**Dataset Requirements**:
- ARFF format files in `datasets/` directory
- `all.txt` file listing datasets: `<name>,<class_name>,<real_features>`
- Supports both discrete and continuous features
- Automatic discretization available via MDLP

**Experimental Data**:
- Results stored in JSON format with versioned schemas
- Test data in `tests/data/` for unit testing
- Sample datasets: iris, diabetes, ecoli, glass, etc.

## Development Workflow

1. **Setup**: Run `make init` to install dependencies via Conan
2. **Development**: Use `make debug` for development builds with testing
3. **Testing**: Run `make test` after changes
4. **Release**: Use `make release` for optimized builds
5. **Experiments**: Use `.env` configuration and run `b_main` with appropriate flags

## Key Features

- **Multi-threaded**: Uses MPI for parallel grid search and experiments
- **Cross-platform**: Supports Linux and macOS via vcpkg
- **Extensible**: Easy classifier registration and integration
- **Research-focused**: Designed for machine learning experimentation
- **Visualization**: DOT graph generation for decision trees and networks