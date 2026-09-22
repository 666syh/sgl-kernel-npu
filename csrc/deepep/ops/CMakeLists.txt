cmake_minimum_required(VERSION 3.16.0)
project(opp)
find_package(ASC REQUIRED)

# The top-level build and this OPC project are configured independently. Read
# the switch from the environment/command line so both projects use the same
# A5 fused_deep_moe setting.
if(NOT DEFINED DEEPEP_ENABLE_A5_FUSED_DEEP_MOE)
    if(DEFINED ENV{DEEPEP_ENABLE_A5_FUSED_DEEP_MOE} AND
       NOT "$ENV{DEEPEP_ENABLE_A5_FUSED_DEEP_MOE}" STREQUAL "")
        set(DEEPEP_ENABLE_A5_FUSED_DEEP_MOE
            "$ENV{DEEPEP_ENABLE_A5_FUSED_DEEP_MOE}" CACHE STRING
            "Build the Ascend 950 fused_deep_moe operator")
    else()
        set(DEEPEP_ENABLE_A5_FUSED_DEEP_MOE ON CACHE STRING
            "Build the Ascend 950 fused_deep_moe operator")
    endif()
endif()
if(NOT "${DEEPEP_ENABLE_A5_FUSED_DEEP_MOE}" STREQUAL "ON" AND
   NOT "${DEEPEP_ENABLE_A5_FUSED_DEEP_MOE}" STREQUAL "OFF")
    message(FATAL_ERROR "DEEPEP_ENABLE_A5_FUSED_DEEP_MOE must be ON or OFF")
endif()
message(STATUS "A5 fused_deep_moe: ${DEEPEP_ENABLE_A5_FUSED_DEEP_MOE}")

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
