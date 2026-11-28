cmake_minimum_required(VERSION 3.20)

project(Platform
  VERSION 1.1.7
  DESCRIPTION "Platform to run Experiments with classifiers."
  HOMEPAGE_URL "https://github.com/rmontanana/platform"
  LANGUAGES CXX
)

# Global CMake variables
# ----------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED            ON)
set(CMAKE_CXX_EXTENSIONS                  OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS          ON)
set(CMAKE_POSITION_INDEPENDENT_CODE        ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Options
# -------
option(ENABLE_TESTING "Unit testing build"                        OFF)
option(CODE_COVERAGE "Collect coverage from test library"         OFF)

# CMakes modules
# --------------
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules ${CMAKE_MODULE_PATH})

# MPI
find_package(MPI REQUIRED)
message("MPI_CXX_LIBRARIES=${MPI_CXX_LIBRARIES}")
message("MPI_CXX_INCLUDE_DIRS=${MPI_CXX_INCLUDE_DIRS}")

cmake_policy(SET CMP0135 NEW)

# External libraries - dependencies of Platform
# ---------------------------------------------
find_package(nlohmann_json CONFIG REQUIRED)
find_package(argparse CONFIG REQUIRED)
find_package(Torch CONFIG REQUIRED)
find_package(arff-files CONFIG REQUIRED)
find_package(fimdlp CONFIG REQUIRED)
find_package(folding CONFIG REQUIRED)
find_package(bayesnet CONFIG REQUIRED)
# PyClassifiers >= 1.1 discovers Python and Boost itself and attaches them to
# its target: the CPython minor version and SOABI and the exact Boost version
# must be the ones the package was built with (its python_abi and boost_version
# options, set by the Makefile). Select the interpreter with PYCLASSIFIERS_PYTHON
# and a Boost outside the default prefixes with BOOST_ROOT.
find_package(PyClassifiers CONFIG REQUIRED)
find_package(libxlsxwriter CONFIG REQUIRED)
message("Python3_EXECUTABLE=${Python3_EXECUTABLE}")
message("Python3_VERSION=${Python3_VERSION}")
message("Boost_VERSION=${Boost_VERSION}")
message("Boost_INCLUDE_DIRS=${Boost_INCLUDE_DIRS}")
include_directories(${Boost_INCLUDE_DIRS})
# `make install` copies the executables out of the build tree, so they have to
# find the Boost.Python and the libpython they were built with on their own:
# neither is a Conan package nor, usually, in a system library directory.
get_filename_component(PLATFORM_BOOST_LIBRARY_DIR "${Boost_DIR}/../.." ABSOLUTE)
list(APPEND CMAKE_BUILD_RPATH "${PLATFORM_BOOST_LIBRARY_DIR}" ${Python3_LIBRARY_DIRS})
message("Runtime library path=${CMAKE_BUILD_RPATH}")

# Subdirectories
# --------------
## Configure test data path
cmake_path(SET TEST_DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
configure_file(src/common/SourceData.h.in "${CMAKE_BINARY_DIR}/configured_files/include/SourceData.h")
add_subdirectory(config)
add_subdirectory(src)
add_subdirectory(sample)
file(GLOB Platform_SOURCES CONFIGURE_DEPENDS ${Platform_SOURCE_DIR}/src/*.cpp)

# Testing
# -------
if (ENABLE_TESTING)
  MESSAGE("Testing enabled")
  set(CMAKE_CXX_FLAGS_DEBUG " ${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage -O0 -g")
  enable_testing()
  find_package(Catch2 CONFIG REQUIRED)
  set(CODE_COVERAGE ON)
  include(CTest)
  add_subdirectory(tests)
endif (ENABLE_TESTING)
if (CODE_COVERAGE)
    MESSAGE("Code coverage enabled")
    include(CodeCoverage)
    SET(GCC_COVERAGE_LINK_FLAGS " ${GCC_COVERAGE_LINK_FLAGS} -lgcov --coverage")
endif (CODE_COVERAGE)
