SHELL := /bin/bash
.DEFAULT_GOAL := help
.PHONY: coverage setup help build test clean debug release submodules buildr buildd install dependency testp testb clang-uml

f_release = build_release
f_debug = build_debug
app_targets = b_best b_list b_main b_manage b_grid
test_targets = unit_tests_platform
n_procs = -j 16

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


sub-init: ## Initialize submodules
	@git submodule update --init --recursive

sub-update: ## Initialize submodules
	@git submodule update --remote --merge
	@git submodule foreach git pull origin master

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
	make buildr
	@echo "*******************************************"
	@echo ">>> Copying files to $(dest)"
	@echo "*******************************************"
	@for item in $(app_targets); do \
		echo ">>> Copying $$item" ; \
		cp $(f_release)/src/$$item $(dest) ; \
	done

dependency: ## Create a dependency graph diagram of the project (build/dependency.png)
	@echo ">>> Creating dependency graph diagram of the project...";
	$(MAKE) debug
	cd $(f_debug) && cmake .. --graphviz=dependency.dot && dot -Tpng dependency.dot -o dependency.png

buildd: ## Build the debug targets
	cmake --build $(f_debug) -t $(app_targets) PlatformSample $(n_procs)

buildr: ## Build the release targets
	cmake --build $(f_release) -t $(app_targets) $(n_procs)

clean: ## Clean the tests info
	@echo ">>> Cleaning Debug Platform tests...";
	$(call ClearTests)
	@echo ">>> Done";

clang-uml: ## Create uml class and sequence diagrams
	clang-uml -p --add-compile-flag -I /usr/lib/gcc/x86_64-redhat-linux/8/include/

debug: ## Build a debug version of the project
	@echo ">>> Building Debug Platform...";
	@if [ -d ./$(f_debug) ]; then rm -rf ./$(f_debug); fi
	@mkdir $(f_debug); 
	@cmake -S . -B $(f_debug) -D CMAKE_BUILD_TYPE=Debug -D ENABLE_TESTING=ON -D CODE_COVERAGE=ON
	@echo ">>> Done";

release: ## Build a Release version of the project
	@echo ">>> Building Release Platform...";
	@if [ -d ./$(f_release) ]; then rm -rf ./$(f_release); fi
	@mkdir $(f_release); 
	@cmake -S . -B $(f_release) -D CMAKE_BUILD_TYPE=Release
	@echo ">>> Done";	

opt = ""
test: ## Run tests (opt="-s") to verbose output the tests, (opt="-c='Test Maximum Spanning Tree'") to run only that section
	@echo ">>> Running Platform tests...";
	@$(MAKE) clean
	@cmake --build $(f_debug) -t $(test_targets) $(n_procs)
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
	@cmake --build build_debug -t sample
	build_debug/sample/PlatformSample --model BoostAODE --dataset $(fname) --discretize --stratified
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
