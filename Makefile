SHELL := /bin/bash
.DEFAULT_GOAL := help
.PHONY: init clean coverage setup help build test clean debug release buildr buildd install dependency testp testb clang-uml example

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
	@conan install . --build=missing -of $(2) -s build_type=$(1)
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

init: ## Initialize the project installing dependencies
	@echo ">>> Installing dependencies with Conan"
	@conan install . --output-folder=build --build=missing -s build_type=Release
	@conan install . --output-folder=build_debug --build=missing -s build_type=Debug
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
install: ## Copy binary files to bin folder
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

buildd: ## Build the debug targets
	@$(call compile_target,"Debug","$(f_debug)")

buildr: ## Build the release targets
	@$(call compile_target,"Release","$(f_release)")

clang-uml: ## Create uml class and sequence diagrams
	clang-uml -p --add-compile-flag -I /usr/lib/gcc/x86_64-redhat-linux/8/include/

debug: ## Build a debug version of the project with Conan
	@$(call build_target,"Debug","$(f_debug)", "ENABLE_TESTING=ON")

release: ## Build a Release version of the project with Conan
	@$(call build_target,"Release","$(f_release)", "ENABLE_TESTING=OFF")


opt = ""
test: ## Run tests (opt="-s") to verbose output the tests, (opt="-c='Test Maximum Spanning Tree'") to run only that section
	@echo ">>> Running Platform tests...";
	@$(MAKE) clean
	@$(MAKE) debug
	@$(call "Compile_target", "Debug", "$(f_debug)", $(test_targets))
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
	@cmake --build $(f_release) -t sample
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
