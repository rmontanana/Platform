cmake_minimum_required(VERSION 3.20)

project(Platform
  VERSION 1.1.0
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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast")
set(CMAKE_CXX_FLAGS_DEBUG " ${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage -O0 -g")

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

# Boost Library
cmake_policy(SET CMP0135 NEW)
cmake_policy(SET CMP0167 NEW) # For FindBoost
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 

find_package(Boost 1.66.0 REQUIRED COMPONENTS python3 numpy3) 

# # Python
find_package(Python3 REQUIRED COMPONENTS Development)
# # target_include_directories(MyTarget SYSTEM PRIVATE ${Python3_INCLUDE_DIRS})
# message("Python_LIBRARIES=${Python_LIBRARIES}")

# # Boost Python
# find_package(boost_python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR} CONFIG REQUIRED COMPONENTS python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR})
# # target_link_libraries(MyTarget PRIVATE Boost::python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR})


if(Boost_FOUND)
    message("Boost_INCLUDE_DIRS=${Boost_INCLUDE_DIRS}")
    message("Boost_LIBRARIES=${Boost_LIBRARIES}")
    message("Boost_VERSION=${Boost_VERSION}")
    include_directories(${Boost_INCLUDE_DIRS}) 
endif()



# External libraries - dependencies of Platform
# ---------------------------------------------

find_library(XLSXWRITER_LIB NAMES libxlsxwriter.dylib libxlsxwriter.so PATHS ${Platform_SOURCE_DIR}/lib/libxlsxwriter/lib)
# find_path(XLSXWRITER_INCLUDE_DIR xlsxwriter.h)
# find_library(XLSXWRITER_LIBRARY xlsxwriter)
# message("XLSXWRITER_INCLUDE_DIR=${XLSXWRITER_INCLUDE_DIR}")
# message("XLSXWRITER_LIBRARY=${XLSXWRITER_LIBRARY}")
find_package(Torch CONFIG REQUIRED)
find_package(fimdlp CONFIG REQUIRED)
find_package(folding CONFIG REQUIRED)
find_package(argparse CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS python)
find_package(arff-files CONFIG REQUIRED)

# BayesNet
find_library(bayesnet NAMES libbayesnet bayesnet libbayesnet.a PATHS ${Platform_SOURCE_DIR}/../lib/lib REQUIRED)
find_path(Bayesnet_INCLUDE_DIRS REQUIRED NAMES bayesnet PATHS ${Platform_SOURCE_DIR}/../lib/include)
add_library(bayesnet::bayesnet UNKNOWN IMPORTED)
set_target_properties(bayesnet::bayesnet PROPERTIES
  IMPORTED_LOCATION ${bayesnet}
  INTERFACE_INCLUDE_DIRECTORIES ${Bayesnet_INCLUDE_DIRS})
message(STATUS "BayesNet=${bayesnet}")
message(STATUS "BayesNet_INCLUDE_DIRS=${Bayesnet_INCLUDE_DIRS}")

# PyClassifiers
find_library(PyClassifiers NAMES libPyClassifiers PyClassifiers libPyClassifiers.a PATHS ${Platform_SOURCE_DIR}/../lib/lib REQUIRED)
find_path(PyClassifiers_INCLUDE_DIRS REQUIRED NAMES pyclassifiers PATHS ${Platform_SOURCE_DIR}/../lib/include)
message(STATUS "PyClassifiers=${PyClassifiers}")
message(STATUS "PyClassifiers_INCLUDE_DIRS=${PyClassifiers_INCLUDE_DIRS}")

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
  enable_testing()
  MESSAGE("Testing enabled")
  find_package(Catch2 CONFIG REQUIRED)
  include(CTest)
  add_subdirectory(tests)
endif (ENABLE_TESTING)
if (CODE_COVERAGE)
    include(CodeCoverage)
    MESSAGE("Code coverage enabled")
    SET(GCC_COVERAGE_LINK_FLAGS " ${GCC_COVERAGE_LINK_FLAGS} -lgcov --coverage")
endif (CODE_COVERAGE)
