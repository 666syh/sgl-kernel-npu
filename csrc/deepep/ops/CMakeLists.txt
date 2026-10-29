cmake_minimum_required(VERSION 3.16.0)
project(opp)
find_package(ASC REQUIRED)
include(${CMAKE_CURRENT_SOURCE_DIR}/common/cmake/DeepEPCannVersion.cmake)
deepep_detect_cann_version(DEEPEP_CANN_VERSION_NUM)
set(DEEPEP_LAYERED_MIN_CANN_VERSION 90200000)
add_compile_definitions(
    DEEPEP_CANN_VERSION_NUM=${DEEPEP_CANN_VERSION_NUM}
    DEEPEP_LAYERED_MIN_CANN_VERSION=${DEEPEP_LAYERED_MIN_CANN_VERSION})
set(package_name ${vendor_name})

npu_op_package(${package_name}
    TYPE RUN
    CONFIG
        INSTALL_PATH ${CMAKE_BINARY_DIR}/
)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/framework)
    add_subdirectory(framework)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/op_host)
    add_subdirectory(op_host)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel)
    add_subdirectory(op_kernel)
endif()
