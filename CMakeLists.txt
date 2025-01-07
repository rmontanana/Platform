cmake_minimum_required(VERSION 3.20)

project(Platform
  VERSION 1.1.0
  DESCRIPTION "Platform to run Experiments with classifiers."
  HOMEPAGE_URL "https://github.com/rmontanana/platform"
  LANGUAGES CXX
)

find_package(Torch REQUIRED)

if (POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif ()

# Global CMake variables
# ----------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED            ON)
set(CMAKE_CXX_EXTENSIONS                  OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS          ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast")
set(CMAKE_CXX_FLAGS_DEBUG " ${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage -O0 -g")

# Options
# -------
option(ENABLE_CLANG_TIDY "Enable to add clang tidy."              OFF)
option(ENABLE_TESTING "Unit testing build"                        OFF)
option(CODE_COVERAGE "Collect coverage from test library"         OFF)

# MPI
find_package(MPI REQUIRED)
message("MPI_CXX_LIBRARIES=${MPI_CXX_LIBRARIES}")
message("MPI_CXX_INCLUDE_DIRS=${MPI_CXX_INCLUDE_DIRS}")

# Boost Library
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost 1.66.0 REQUIRED COMPONENTS python3 numpy3) 
if(Boost_FOUND)
    message("Boost_INCLUDE_DIRS=${Boost_INCLUDE_DIRS}")
    include_directories(${Boost_INCLUDE_DIRS}) 
endif()

# Python
find_package(Python3 3.11 COMPONENTS Interpreter Development REQUIRED)
message("Python3_LIBRARIES=${Python3_LIBRARIES}")

# CMakes modules
# --------------
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules ${CMAKE_MODULE_PATH})
include(AddGitSubmodule)

if (CODE_COVERAGE)
    enable_testing()
    include(CodeCoverage)
    MESSAGE("Code coverage enabled")
    SET(GCC_COVERAGE_LINK_FLAGS " ${GCC_COVERAGE_LINK_FLAGS} -lgcov --coverage")
endif (CODE_COVERAGE)

if (ENABLE_CLANG_TIDY)
  include(StaticAnalyzers) # clang-tidy
endif (ENABLE_CLANG_TIDY)

# External libraries - dependencies of Platform
# ---------------------------------------------
add_git_submodule("lib/argparse")
add_git_submodule("lib/mdlp")

find_library(XLSXWRITER_LIB NAMES libxlsxwriter.dylib libxlsxwriter.so PATHS ${Platform_SOURCE_DIR}/lib/libxlsxwriter/lib)
message("XLSXWRITER_LIB=${XLSXWRITER_LIB}")

find_library(PyClassifiers NAMES libPyClassifiers PyClassifiers libPyClassifiers.a PATHS ${Platform_SOURCE_DIR}/../lib/lib REQUIRED)
find_path(PyClassifiers_INCLUDE_DIRS REQUIRED NAMES pyclassifiers PATHS ${Platform_SOURCE_DIR}/../lib/include)
find_library(BayesNet NAMES libBayesNet BayesNet libBayesNet.a PATHS ${Platform_SOURCE_DIR}/../lib/lib REQUIRED)
find_path(Bayesnet_INCLUDE_DIRS REQUIRED NAMES bayesnet PATHS ${Platform_SOURCE_DIR}/../lib/include)

message(STATUS "PyClassifiers=${PyClassifiers}")
message(STATUS "PyClassifiers_INCLUDE_DIRS=${PyClassifiers_INCLUDE_DIRS}")
message(STATUS "BayesNet=${BayesNet}")
message(STATUS "Bayesnet_INCLUDE_DIRS=${Bayesnet_INCLUDE_DIRS}")

# Subdirectories
# --------------
## Configure test data path
cmake_path(SET TEST_DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
configure_file(src/common/SourceData.h.in "${CMAKE_BINARY_DIR}/configured_files/include/SourceData.h")
add_subdirectory(config)
add_subdirectory(src)
# add_subdirectory(sample)
file(GLOB Platform_SOURCES CONFIGURE_DEPENDS ${Platform_SOURCE_DIR}/src/*.cpp)

# Testing
# -------
if (ENABLE_TESTING)
  MESSAGE("Testing enabled")
  if (NOT TARGET Catch2::Catch2)
    add_git_submodule("lib/catch2")
  endif (NOT TARGET Catch2::Catch2)
  include(CTest)
  add_subdirectory(tests)
endif (ENABLE_TESTING)
