SHELL := /bin/bash
.DEFAULT_GOAL := help
.PHONY: env env-python env-boost init clean coverage setup help build test clean debug release buildr buildd install dependency testp testb clang-uml example

f_release = build_Release
f_debug = build_Debug
app_targets = b_best b_list b_main b_manage b_grid b_results
test_targets = unit_tests_platform
# Set the number of parallel jobs to the number of available processors minus 7
CPUS := $(shell getconf _NPROCESSORS_ONLN 2>/dev/null \
                 || nproc --all 2>/dev/null \
                 || sysctl -n hw.ncpu)

# --- Your desired job count: CPUs – 7, but never less than 1 --------------
JOBS := $(shell n=$(CPUS); [ $${n} -gt 7 ] && echo $$((n-7)) || echo 1)

# --- Python and Boost of pyclassifiers --------------------------------------
# pyclassifiers >= 1.1 is built for one CPython minor version and one exact
# Boost version, and Platform has to be built and run with the same pair. Both
# are pinned and built on demand inside the workspace (`make env`, which every
# build target runs first), so nothing has to be installed beforehand:
#   $(ENV_DIR)            venv of a uv-managed CPython 3.12 with requirements.txt
#   $(DEPS_DIR)/boost     Boost.Python / Boost.NumPy 1.92.0
#   $(DEPS_DIR)/{uv,python,cache,sources,logs}   what they are built from
# To use an existing pair instead, give both:
#   PYCLASSIFIERS_PYTHON  interpreter (with the Python classifiers installed)
#   BOOST_ROOT            prefix of a Boost.Python built for that interpreter
#   BOOST_VERSION         its exact version, if $(BOOST_ROOT)/include does not tell
DEPS_DIR ?= build_deps
ENV_DIR ?= .venv
DEV_PYTHON ?= python3
ifeq ($(strip $(PYCLASSIFIERS_PYTHON)),)
PYTHON := $(CURDIR)/$(ENV_DIR)/bin/python
PY_ABI := 3.12
env_python := env-python
else
PYTHON := $(PYCLASSIFIERS_PYTHON)
PY_ABI = $(shell "$(PYTHON)" -c "import sys; print('%d.%d' % sys.version_info[:2])")
env_python :=
endif
ifeq ($(strip $(BOOST_ROOT)),)
BOOST_ROOT := $(CURDIR)/$(DEPS_DIR)/boost
BOOST_VERSION ?= 1.92.0
env_boost := env-boost
else
BOOST_VERSION ?= $(shell awk '/define BOOST_VERSION /{v=$$3; printf "%d.%d.%d\n", v/100000, (v/100)%1000, v%100; exit}' \
	"$(BOOST_ROOT)/include/boost/version.hpp" 2>/dev/null)
env_boost :=
endif
export PYCLASSIFIERS_PYTHON := $(PYTHON)
export BOOST_ROOT
pyclfs_options = -o "pyclassifiers/*:python_abi=$(PY_ABI)" -o "pyclassifiers/*:boost_version=$(BOOST_VERSION)"

define ClearTests
	@for t in $(test_targets); do \
		if [ -f $(f_debug)/tests/$$t ]; then \
			echo ">>> Cleaning $$t..." ; \
			rm -f $(f_debug)/tests/$$t ; \
		fi ; \
	done
	@nfiles="$(find . -name "*.gcda" -print0)" ; \
	if test "${nfiles}" != "" ; then \
		find . -name "*.gcda" -print0 | xargs -0 rm 2>/dev/null ;\
	fi ; 
endef

define build_target
	@echo ">>> Building the project for $(1)..."
	@if [ -d $(2) ]; then rm -fr $(2); fi
	@echo ">>> Python $(PYTHON) (ABI $(PY_ABI)), Boost $(BOOST_VERSION)"
	@conf=$$($(DEV_PYTHON) scripts/python_conf.py "$(PYTHON)") || exit 1; \
	conan install . --build=missing -of $(2) -s build_type=$(1) $(pyclfs_options) $${conf:+-c "$$conf"}
	@cmake -S . -B $(2) -DCMAKE_TOOLCHAIN_FILE=$(2)/build/$(1)/generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=$(1) -D$(3)
	@echo ">>> Will build using $(JOBS) parallel jobs"
	echo ">>> Done"
endef

define compile_target
	@echo ">>> Compiling for $(1)..."
		if [ "$(3)" != "" ]; then \
		target="-t$(3)"; \
	else \
		target=""; \
	fi
	@cmake --build $(2) --config $(1) --parallel $(JOBS) $(target)
	@echo ">>> Done"
endef

env: $(env_python) $(env_boost) ## Build what is missing of the Python 3.12 environment and the Boost.Python that the Python classifiers need

env-python:
	@if [ ! -x "$(PYTHON)" ]; then \
		echo ">>> Creating $(ENV_DIR) with a uv-managed CPython $(PY_ABI)..."; \
		base=$$($(DEV_PYTHON) scripts/bootstrap_python.py --base "$(DEPS_DIR)") || exit 1; \
		"$$base" -m venv "$(ENV_DIR)" || exit 1; \
	fi
	@if ! cmp -s requirements.txt "$(ENV_DIR)/.requirements"; then \
		echo ">>> Installing requirements.txt in $(ENV_DIR)..."; \
		PIP_CACHE_DIR="$(CURDIR)/$(DEPS_DIR)/pip-cache" "$(PYTHON)" -m pip install --quiet --disable-pip-version-check \
			--requirement requirements.txt || exit 1; \
		cp requirements.txt "$(ENV_DIR)/.requirements"; \
	fi

env-boost: $(env_python)
	@if [ ! -d "$(BOOST_ROOT)/lib/cmake/Boost-$(BOOST_VERSION)" ]; then \
		echo ">>> Building Boost.Python $(BOOST_VERSION) in $(DEPS_DIR)/boost (log: $(DEPS_DIR)/logs/boost-build.log)..."; \
		"$(PYTHON)" scripts/bootstrap_boost.py --python "$(PYTHON)" --jobs "$(JOBS)" --base "$(DEPS_DIR)" || exit 1; \
	fi

init: env ## Initialize the project installing dependencies
	@echo ">>> Installing dependencies with Conan"
	@conan install . --output-folder=build --build=missing -s build_type=Release $(pyclfs_options)
	@conan install . --output-folder=build_debug --build=missing -s build_type=Debug $(pyclfs_options)
	@echo ">>> Done"

clean: ## Clean the project
	@echo ">>> Cleaning the project..."
	@if test -f CMakeCache.txt ; then echo "- Deleting CMakeCache.txt"; rm -f CMakeCache.txt; fi
	@for folder in $(f_release) $(f_debug) build build_debug install_test ; do \
	if test -d "$$folder" ; then \
		echo "- Deleting $$folder folder" ; \
		rm -rf "$$folder"; \
	fi; \
	done
	$(call ClearTests)
	@echo ">>> Done";
setup: ## Install dependencies for tests and coverage
	@if [ "$(shell uname)" = "Darwin" ]; then \
		brew install gcovr; \
		brew install lcov; \
	fi
	@if [ "$(shell uname)" = "Linux" ]; then \
		pip install gcovr; \
	fi

dest ?= ${HOME}/bin
install: env ## Copy binary files to bin folder
	@echo "Destination folder: $(dest)"
	@make buildr
	@echo "*******************************************"
	@echo ">>> Copying files to $(dest)"
	@echo "*******************************************"
	@for item in $(app_targets); do \
		echo ">>> Copying $$item" ; \
		cp $(f_release)/src/$$item $(dest) || { \
            echo "*** Error copying $$item" ; \
        } ; \
	done

dependency: ## Create a dependency graph diagram of the project (build/dependency.png)
	@echo ">>> Creating dependency graph diagram of the project...";
	$(MAKE) debug
	cd $(f_debug) && cmake .. --graphviz=dependency.dot && dot -Tpng dependency.dot -o dependency.png

buildd: env ## Build the debug targets (configuring them first if needed)
	@[ -f $(f_debug)/CMakeCache.txt ] || $(MAKE) debug
	@$(call compile_target,"Debug","$(f_debug)")

buildr: env ## Build the release targets (configuring them first if needed)
	@[ -f $(f_release)/CMakeCache.txt ] || $(MAKE) release
	@$(call compile_target,"Release","$(f_release)")

clang-uml: ## Create uml class and sequence diagrams
	clang-uml -p --add-compile-flag -I /usr/lib/gcc/x86_64-redhat-linux/8/include/

debug: env ## Build a debug version of the project with Conan
	@$(call build_target,"Debug","$(f_debug)", "ENABLE_TESTING=ON")

release: env ## Build a Release version of the project with Conan
	@$(call build_target,"Release","$(f_release)", "ENABLE_TESTING=OFF")


opt = ""
test: env ## Run tests (opt="-s") to verbose output the tests, (opt="-c='Test Maximum Spanning Tree'") to run only that section
	@echo ">>> Running Platform tests...";
	@$(MAKE) debug
	@$(call compile_target, "Debug", "$(f_debug)", $(test_targets))
	@for t in $(test_targets); do \
		if [ -f $(f_debug)/tests/$$t ]; then \
			cd $(f_debug)/tests ; \
			./$$t $(opt) ; \
		fi ; \
	done
	@echo ">>> Done";

fname = iris
example: ## Build sample
	@echo ">>> Building Sample...";
	@cmake --build $(f_release) -t PlatformSample
	$(f_release)/sample/PlatformSample --model BoostAODE --dataset $(fname) --discretize --stratified
	@echo ">>> Done";


coverage: ## Run tests and generate coverage report (build/index.html)
	@echo ">>> Building tests with coverage..."
	@$(MAKE) test
	@gcovr $(f_debug)/tests
	@echo ">>> Done";		


help: ## Show help message
	@IFS=$$'\n' ; \
	help_lines=(`grep -Fh "##" $(MAKEFILE_LIST) | grep -Fv fgrep | sed -e 's/\\$$//' | sed -e 's/##/:/'`); \
	printf "%s\n\n" "Usage: make [task]"; \
	printf "%-20s %s\n" "task" "help" ; \
	printf "%-20s %s\n" "------" "----" ; \
	for help_line in $${help_lines[@]}; do \
		IFS=$$':' ; \
		help_split=($$help_line) ; \
		help_command=`echo $${help_split[0]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
		help_info=`echo $${help_split[2]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
		printf '\033[36m'; \
		printf "%-20s %s" $$help_command ; \
		printf '\033[0m'; \
		printf "%s\n" $$help_info; \
	done
