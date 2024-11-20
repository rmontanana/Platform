include_directories(
    ## Libs
    ${Platform_SOURCE_DIR}/lib/Files
    ${Platform_SOURCE_DIR}/lib/folding
    ${Platform_SOURCE_DIR}/lib/mdlp/src
    ${Platform_SOURCE_DIR}/lib/argparse/include
    ${Platform_SOURCE_DIR}/lib/json/include
    ${Platform_SOURCE_DIR}/lib/libxlsxwriter/include
    ${Python3_INCLUDE_DIRS}
    ${MPI_CXX_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${CMAKE_BINARY_DIR}/configured_files/include
    ${PyClassifiers_INCLUDE_DIRS}
    ${Bayesnet_INCLUDE_DIRS}
    ${FImdlp_INCLUDE_DIRS}
    ## Platform
    ${Platform_SOURCE_DIR}/src
    ${Platform_SOURCE_DIR}/results
)

# b_best
add_executable(
    b_best commands/b_best.cpp best/Statistics.cpp
    best/BestResultsExcel.cpp best/BestResultsTex.cpp best/BestResultsMd.cpp best/BestResults.cpp
    common/Datasets.cpp common/Dataset.cpp common/Discretization.cpp 
    main/Models.cpp main/Scores.cpp
    reports/ReportExcel.cpp reports/ReportBase.cpp reports/ExcelFile.cpp
    results/Result.cpp
)
target_link_libraries(b_best Boost::boost "${PyClassifiers}" "${BayesNet}" "${FImdlp}" ${Python3_LIBRARIES} "${TORCH_LIBRARIES}" ${LIBTORCH_PYTHON} Boost::python Boost::numpy "${XLSXWRITER_LIB}")

# b_grid
set(grid_sources GridSearch.cpp GridData.cpp)
list(TRANSFORM grid_sources PREPEND grid/)
add_executable(b_grid commands/b_grid.cpp ${grid_sources} 
    common/Datasets.cpp common/Dataset.cpp common/Discretization.cpp
    main/HyperParameters.cpp main/Models.cpp 
)
target_link_libraries(b_grid ${MPI_CXX_LIBRARIES} "${PyClassifiers}" "${BayesNet}" "${FImdlp}" ${Python3_LIBRARIES} "${TORCH_LIBRARIES}" ${LIBTORCH_PYTHON} Boost::python Boost::numpy)

# b_list
add_executable(b_list commands/b_list.cpp
    common/Datasets.cpp common/Dataset.cpp common/Discretization.cpp
    main/Models.cpp main/Scores.cpp
    reports/ReportExcel.cpp reports/ExcelFile.cpp reports/ReportBase.cpp reports/DatasetsExcel.cpp reports/DatasetsConsole.cpp reports/ReportsPaged.cpp
    results/Result.cpp results/ResultsDatasetExcel.cpp results/ResultsDataset.cpp results/ResultsDatasetConsole.cpp
)
target_link_libraries(b_list "${PyClassifiers}" "${BayesNet}" "${FImdlp}" ${Python3_LIBRARIES} "${TORCH_LIBRARIES}" ${LIBTORCH_PYTHON} Boost::python Boost::numpy "${XLSXWRITER_LIB}")

# b_main
set(main_sources Experiment.cpp Models.cpp HyperParameters.cpp Scores.cpp)
list(TRANSFORM main_sources PREPEND main/)
add_executable(b_main commands/b_main.cpp ${main_sources} 
    common/Datasets.cpp common/Dataset.cpp common/Discretization.cpp
    reports/ReportConsole.cpp reports/ReportBase.cpp 
    results/Result.cpp
)
target_link_libraries(b_main "${PyClassifiers}" "${BayesNet}" "${FImdlp}" ${Python3_LIBRARIES} "${TORCH_LIBRARIES}" ${LIBTORCH_PYTHON} Boost::python Boost::numpy)

# b_manage
set(manage_sources ManageScreen.cpp OptionsMenu.cpp ResultsManager.cpp)
list(TRANSFORM manage_sources PREPEND manage/)
add_executable(
    b_manage commands/b_manage.cpp ${manage_sources} 
    common/Datasets.cpp common/Dataset.cpp common/Discretization.cpp
    reports/ReportConsole.cpp reports/ReportExcel.cpp reports/ReportExcelCompared.cpp reports/ReportBase.cpp reports/ExcelFile.cpp reports/DatasetsConsole.cpp reports/ReportsPaged.cpp
    results/Result.cpp results/ResultsDataset.cpp results/ResultsDatasetConsole.cpp
    main/Scores.cpp
)
target_link_libraries(b_manage "${TORCH_LIBRARIES}" "${XLSXWRITER_LIB}" "${FImdlp}" "${BayesNet}")
