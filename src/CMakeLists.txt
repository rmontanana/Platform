include_directories(
    ## Libs
    ${Platform_SOURCE_DIR}/lib/PyClassifiers/lib/BayesNet/src
    ${Platform_SOURCE_DIR}/lib/PyClassifiers/lib/BayesNet/lib/folding
    ${Platform_SOURCE_DIR}/lib/PyClassifiers/lib/BayesNet/lib/mdlp
    ${Platform_SOURCE_DIR}/lib/PyClassifiers/lib/BayesNet/lib/json/include
    ${Platform_SOURCE_DIR}/lib/PyClassifiers/src
    ${Platform_SOURCE_DIR}/lib/Files
    ${Platform_SOURCE_DIR}/lib/mdlp
    ${Platform_SOURCE_DIR}/lib/argparse/include
    ${Platform_SOURCE_DIR}/lib/json/include
    ${Platform_SOURCE_DIR}/lib/libxlsxwriter/include
    ${Python3_INCLUDE_DIRS}
    ${MPI_CXX_INCLUDE_DIRS}
    ${CMAKE_BINARY_DIR}/configured_files/include
    ## Platform
    ${Platform_SOURCE_DIR}/src/common
    ${Platform_SOURCE_DIR}/src/best
    ${Platform_SOURCE_DIR}/src/grid
    ${Platform_SOURCE_DIR}/src/main
    ${Platform_SOURCE_DIR}/src/manage
    ${Platform_SOURCE_DIR}/src/reports
)

# b_best
set(best_sources b_best.cc BestResults.cc  Statistics.cc BestResultsExcel.cc)
list(TRANSFORM best_sources PREPEND best/)
add_executable(b_best ${best_sources} main/Result.cc reports/ReportExcel.cc reports/ReportBase.cc reports/ExcelFile.cc common/Datasets.cc common/Dataset.cc)
target_link_libraries(b_best Boost::boost "${TORCH_LIBRARIES}" "${XLSXWRITER_LIB}" ArffFiles mdlp)

# b_grid
set(grid_sources b_grid.cc GridSearch.cc GridData.cc)
list(TRANSFORM grid_sources PREPEND grid/)
add_executable(b_grid ${grid_sources} main/HyperParameters.cc main/Models.cc common/Datasets.cc common/Dataset.cc)
target_link_libraries(b_grid PyClassifiers ${MPI_CXX_LIBRARIES} ArffFiles)

# b_list
set(list_sources b_list.cc DatasetsExcel.cc)
list(TRANSFORM list_sources PREPEND list/)
add_executable(b_list ${list_sources} common/Datasets.cc common/Dataset.cc reports/ReportExcel.cc reports/ExcelFile.cc reports/ReportBase.cc)
target_link_libraries(b_list "${TORCH_LIBRARIES}" "${XLSXWRITER_LIB}" ArffFiles mdlp)

# b_main
set(main_sources b_main.cc Experiment.cc Models.cc HyperParameters.cc)
list(TRANSFORM main_sources PREPEND main/)
add_executable(b_main ${main_sources} common/Datasets.cc common/Dataset.cc reports/ReportConsole.cc reports/ReportBase.cc main/Result.cc)
target_link_libraries(b_main PyClassifiers BayesNet ArffFiles mdlp)

# b_manage
set(manage_sources b_manage.cc ManageResults.cc CommandParser.cc Results.cc)
list(TRANSFORM manage_sources PREPEND manage/)
add_executable(b_manage ${manage_sources} main/Result.cc reports/ReportConsole.cc reports/ReportExcel.cc reports/ReportBase.cc reports/ExcelFile.cc common/Datasets.cc common/Dataset.cc)
target_link_libraries(b_manage "${TORCH_LIBRARIES}" "${XLSXWRITER_LIB}" ArffFiles mdlp)