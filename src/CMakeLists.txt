include_directories(
    ${Python3_INCLUDE_DIRS}
    ${MPI_CXX_INCLUDE_DIRS}
    ${CMAKE_BINARY_DIR}/configured_files/include
    ${Platform_SOURCE_DIR}/src
)

# b_best
add_executable(
    b_best commands/b_best.cpp best/Statistics.cpp
    best/BestResultsExcel.cpp best/BestResultsTex.cpp best/BestResultsMd.cpp best/BestResults.cpp
    common/Datasets.cpp common/Dataset.cpp common/Discretization.cpp 
    main/Models.cpp main/Scores.cpp
    reports/ReportExcel.cpp reports/ReportBase.cpp reports/ExcelFile.cpp
    results/Result.cpp
    experimental_clfs/XA1DE.cpp
    experimental_clfs/ExpClf.cpp
    experimental_clfs/DecisionTree.cpp
    experimental_clfs/AdaBoost.cpp
)
target_link_libraries(b_best Boost::boost  bayesnet::bayesnet argparse::argparse fimdlp::fimdlp ${Python3_LIBRARIES} torch::torch Boost::python Boost::numpy libxlsxwriter::libxlsxwriter)

# b_grid
set(grid_sources GridSearch.cpp GridData.cpp GridExperiment.cpp GridBase.cpp )
list(TRANSFORM grid_sources PREPEND grid/)
add_executable(b_grid commands/b_grid.cpp ${grid_sources} 
    common/Datasets.cpp common/Dataset.cpp common/Discretization.cpp
    main/HyperParameters.cpp main/Models.cpp main/Experiment.cpp main/Scores.cpp main/ArgumentsExperiment.cpp
    reports/ReportConsole.cpp reports/ReportBase.cpp 
    results/Result.cpp
    experimental_clfs/XA1DE.cpp
    experimental_clfs/ExpClf.cpp
    experimental_clfs/DecisionTree.cpp
    experimental_clfs/AdaBoost.cpp
)
target_link_libraries(b_grid ${MPI_CXX_LIBRARIES}  bayesnet::bayesnet argparse::argparse fimdlp::fimdlp ${Python3_LIBRARIES} torch::torch Boost::python Boost::numpy)

# b_list
add_executable(b_list commands/b_list.cpp
    common/Datasets.cpp common/Dataset.cpp common/Discretization.cpp
    main/Models.cpp main/Scores.cpp
    reports/ReportExcel.cpp reports/ExcelFile.cpp reports/ReportBase.cpp reports/DatasetsExcel.cpp reports/DatasetsConsole.cpp reports/ReportsPaged.cpp
    results/Result.cpp results/ResultsDatasetExcel.cpp results/ResultsDataset.cpp results/ResultsDatasetConsole.cpp
    experimental_clfs/XA1DE.cpp
    experimental_clfs/ExpClf.cpp
    experimental_clfs/DecisionTree.cpp
    experimental_clfs/AdaBoost.cpp
)
target_link_libraries(b_list  bayesnet::bayesnet argparse::argparse fimdlp::fimdlp ${Python3_LIBRARIES} torch::torch Boost::python Boost::numpy libxlsxwriter::libxlsxwriter)

# b_main
set(main_sources Experiment.cpp Models.cpp HyperParameters.cpp Scores.cpp ArgumentsExperiment.cpp)
list(TRANSFORM main_sources PREPEND main/)
add_executable(b_main commands/b_main.cpp ${main_sources} 
    common/Datasets.cpp common/Dataset.cpp common/Discretization.cpp
    reports/ReportConsole.cpp reports/ReportBase.cpp 
    results/Result.cpp
    experimental_clfs/XA1DE.cpp
    experimental_clfs/ExpClf.cpp
    experimental_clfs/ExpClf.cpp
    experimental_clfs/DecisionTree.cpp
    experimental_clfs/AdaBoost.cpp
)
target_link_libraries(b_main PRIVATE nlohmann_json::nlohmann_json  bayesnet::bayesnet argparse::argparse fimdlp::fimdlp ${Python3_LIBRARIES} torch::torch Boost::python Boost::numpy)

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
target_link_libraries(b_manage torch::torch libxlsxwriter::libxlsxwriter fimdlp::fimdlp bayesnet::bayesnet argparse::argparse)

# b_results
add_executable(b_results commands/b_results.cpp)
target_link_libraries(b_results torch::torch libxlsxwriter::libxlsxwriter fimdlp::fimdlp bayesnet::bayesnet argparse::argparse)
