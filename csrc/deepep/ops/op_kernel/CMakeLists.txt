# set custom compile options
if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    npu_op_kernel_options(ALL OPTIONS -g -O0 )
endif()

message(STATUS "ASCEND include path: ${ASCEND_CANN_PACKAGE_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ascendc/moe_distribute_dispatch")

set(_DISPATCH_FFN_INC_OPTS)
if (EXISTS ${CMAKE_SOURCE_DIR}/third_party/catlass/include)
    list(APPEND _DISPATCH_FFN_INC_OPTS -I${CMAKE_SOURCE_DIR}/third_party/catlass/include)
else()
    message(FATAL_ERROR "dependency catlass is missing")
endif()
message(WARNING "_DISPATCH_FFN_INC_OPTS: ${_DISPATCH_FFN_INC_OPTS}")

# This OPC project is configured separately from the host CMake project. The
# parent build exports the switch, while this fallback keeps direct cmake
# invocations backward compatible.
if(NOT DEFINED DEEPEP_ENABLE_A5_FUSED_DEEP_MOE OR
   "${DEEPEP_ENABLE_A5_FUSED_DEEP_MOE}" STREQUAL "")
    set(DEEPEP_ENABLE_A5_FUSED_DEEP_MOE ON)
endif()

# FusedDeepMoe A5 on Ascend 950 reuses the same Catlass dependency fetched by
# deepep/ops/build.sh for A3 kernels. These options must be appended to the
# final global option set because OPC's per-op compiler inherits that set.
set(_A5_FUSED_DEEP_MOE_INC_OPTS)
if ("${ASCEND_COMPUTE_UNIT}" MATCHES "^ascend950" AND
    DEEPEP_ENABLE_A5_FUSED_DEEP_MOE)
    set(_A5_SYSTEM_MOE_BASE_COMMON_DIR "${ASCEND_CANN_PACKAGE_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common")
    set(_A5_SYSTEM_MOE_BASE_OP_KERNEL "${_A5_SYSTEM_MOE_BASE_COMMON_DIR}/op_kernel/moe_distribute_base.h")
    set(_A5_SYSTEM_MOE_BASE_INC_KERNEL "${_A5_SYSTEM_MOE_BASE_COMMON_DIR}/inc/kernel/moe_distribute_base.h")

    if (EXISTS "${_A5_SYSTEM_MOE_BASE_OP_KERNEL}")
        list(APPEND _A5_FUSED_DEEP_MOE_INC_OPTS
            -DDEEPEP_A5_SYSTEM_MOE_BASE_USE_OP_KERNEL=1)
    elseif(EXISTS "${_A5_SYSTEM_MOE_BASE_INC_KERNEL}")
        list(APPEND _A5_FUSED_DEEP_MOE_INC_OPTS
            -DDEEPEP_A5_SYSTEM_MOE_BASE_USE_INC_KERNEL=1)
    else()
        message(FATAL_ERROR
            "Cannot find system moe_distribute_base.h for A5 fused. "
            "Checked: ${_A5_SYSTEM_MOE_BASE_OP_KERNEL} and ${_A5_SYSTEM_MOE_BASE_INC_KERNEL}")
    endif()

    list(APPEND _A5_FUSED_DEEP_MOE_INC_OPTS
        -I$ENV{ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common/
        -DCATLASS_ARCH=3510
        -DDEEPEP_CATLASS_A5_COMPAT=1
        -DDEEPEP_SKIP_DISPATCH_FFN_COMBINE=1)
endif()

if ("${ASCEND_COMPUTE_UNIT}" MATCHES "^ascend950" AND
    NOT DEEPEP_ENABLE_A5_FUSED_DEEP_MOE)
    # KERNEL_DIR performs a recursive scan and would pull the canonical
    # fused_deep_moe.cpp back into the A5 manifest. Use the explicit source
    # form so the fused entry and its private tree are absent from the device
    # build altogether.
    file(GLOB_RECURSE _DEEPEP_KERNEL_FILES
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
    list(FILTER _DEEPEP_KERNEL_FILES EXCLUDE REGEX "^fused_deep_moe\\.cpp$")
    list(FILTER _DEEPEP_KERNEL_FILES EXCLUDE REGEX "^fused_deep_moe_a5(/|$)")
    message(STATUS "A5 fused_deep_moe disabled; excluded fused kernel sources")
    npu_op_kernel_sources(ascendc_kernels
        OP_TYPE DeepEP
        KERNEL_FILE "${_DEEPEP_KERNEL_FILES}")
else()
    message(STATUS "A5 fused_deep_moe enabled or non-A5 build; using recursive kernel sources")
    npu_op_kernel_sources(ascendc_kernels
        KERNEL_DIR ./)
endif()

npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}/
    TILING_LIBRARY cust_optiling
)

npu_op_package_add(${package_name}
    LIBRARY ascendc_kernels
)

npu_op_kernel_options("" ALL OPTIONS
    -DASCENDC_DUMP=0
    --cce-auto-sync=off
    ${_DISPATCH_FFN_INC_OPTS}
    ${_A5_FUSED_DEEP_MOE_INC_OPTS})
