include_directories(
    ${Platform_SOURCE_DIR}/lib/PyClassifiers/lib/BayesNet/src/BayesNet
    ${Platform_SOURCE_DIR}/lib/PyClassifiers/lib/BayesNet/lib/folding
    ${Platform_SOURCE_DIR}/lib/PyClassifiers/lib/BayesNet/lib/mdlp
    ${Platform_SOURCE_DIR}/lib/PyClassifiers/lib/BayesNet/lib/json/include
    ${Platform_SOURCE_DIR}/lib/PyClassifiers/src/PyClassifiers
    ${Platform_SOURCE_DIR}/src/Platform
    ${Platform_SOURCE_DIR}/lib/Files
    ${Platform_SOURCE_DIR}/lib/mdlp
    ${Platform_SOURCE_DIR}/lib/argparse/include
    ${Platform_SOURCE_DIR}/lib/json/include
    ${Platform_SOURCE_DIR}/lib/libxlsxwriter/include
    ${Python3_INCLUDE_DIRS}
    ${MPI_CXX_INCLUDE_DIRS}
    ${CMAKE_BINARY_DIR}/configured_files/include
)

add_executable(b_best b_best.cc BestResults.cc Result.cc Statistics.cc BestResultsExcel.cc ReportExcel.cc ReportBase.cc Datasets.cc Dataset.cc ExcelFile.cc)
add_executable(b_grid b_grid.cc GridSearch.cc GridData.cc HyperParameters.cc Datasets.cc Dataset.cc Models.cc)
add_executable(b_list b_list.cc Datasets.cc Dataset.cc)
add_executable(b_main b_main.cc Experiment.cc Datasets.cc Dataset.cc Models.cc HyperParameters.cc ReportConsole.cc ReportBase.cc)
add_executable(b_manage b_manage.cc Results.cc ManageResults.cc CommandParser.cc Result.cc ReportConsole.cc ReportExcel.cc ReportBase.cc Datasets.cc Dataset.cc ExcelFile.cc)

target_link_libraries(b_best Boost::boost "${XLSXWRITER_LIB}" "${TORCH_LIBRARIES}" ArffFiles mdlp)
target_link_libraries(b_grid PyClassifiers ${MPI_CXX_LIBRARIES})
target_link_libraries(b_list ArffFiles mdlp "${TORCH_LIBRARIES}")
target_link_libraries(b_main PyClassifiers BayesNet ArffFiles mdlp "${TORCH_LIBRARIES}")
target_link_libraries(b_manage "${TORCH_LIBRARIES}" "${XLSXWRITER_LIB}" ArffFiles mdlp)