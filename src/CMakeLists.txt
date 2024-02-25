include_directories(
    ${Platform_SOURCE_DIR}/lib/PyClassifiers/lib/BayesNet/src
    ${Platform_SOURCE_DIR}/lib/PyClassifiers/lib/BayesNet/lib/folding
    ${Platform_SOURCE_DIR}/lib/PyClassifiers/lib/BayesNet/lib/mdlp
    ${Platform_SOURCE_DIR}/lib/PyClassifiers/lib/BayesNet/lib/json/include
    ${Platform_SOURCE_DIR}/lib/PyClassifiers/src
    ${Platform_MODULES}
    ${Platform_SOURCE_DIR}/lib/Files
    ${Platform_SOURCE_DIR}/lib/mdlp
    ${Platform_SOURCE_DIR}/lib/argparse/include
    ${Platform_SOURCE_DIR}/lib/json/include
    ${Platform_SOURCE_DIR}/lib/libxlsxwriter/include
    ${Python3_INCLUDE_DIRS}
    ${MPI_CXX_INCLUDE_DIRS}
    ${CMAKE_BINARY_DIR}/configured_files/include
)

set(best_sources BestResults.cc Result.cc Statistics.cc BestResultsExcel.cc ReportExcel.cc ReportBase.cc Datasets.cc Dataset.cc ExcelFile.cc)
list(TRANSFORM best_sources PREPEND ${Platform_MODULES})
add_executable(b_best b_best.cc ${best_sources})
set(grid_sources GridSearch.cc GridData.cc HyperParameters.cc Datasets.cc Dataset.cc Models.cc)
list(TRANSFORM grid_sources PREPEND ${Platform_MODULES})
add_executable(b_grid b_grid.cc ${grid_sources})
add_executable(b_list b_list.cc ${Platform_MODULES}Datasets.cc ${Platform_MODULES}Dataset.cc)
set(main_sources Experiment.cc Datasets.cc Dataset.cc Models.cc HyperParameters.cc ReportConsole.cc ReportBase.cc Result.cc)
list(TRANSFORM main_sources PREPEND ${Platform_MODULES})
add_executable(b_main b_main.cc ${main_sources})
set(manage_sources Results.cc ManageResults.cc CommandParser.cc Result.cc ReportConsole.cc ReportExcel.cc ReportBase.cc Datasets.cc Dataset.cc ExcelFile.cc)
list(TRANSFORM manage_sources PREPEND ${Platform_MODULES})
add_executable(b_manage b_manage.cc ${manage_sources})

target_link_libraries(b_best Boost::boost "${XLSXWRITER_LIB}" "${TORCH_LIBRARIES}" ArffFiles mdlp)
target_link_libraries(b_grid PyClassifiers ${MPI_CXX_LIBRARIES} ArffFiles)
target_link_libraries(b_list ArffFiles mdlp "${TORCH_LIBRARIES}")
target_link_libraries(b_main PyClassifiers BayesNet ArffFiles mdlp "${TORCH_LIBRARIES}")
target_link_libraries(b_manage "${TORCH_LIBRARIES}" "${XLSXWRITER_LIB}" ArffFiles mdlp)