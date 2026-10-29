function(deepep_detect_cann_version out_var)
    if(NOT DEFINED ENV{ASCEND_HOME_PATH} OR "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
        message(FATAL_ERROR "The ASCEND_HOME_PATH environment variable is required to detect CANN_VERSION_NUM")
    endif()

    set(_ascend_home "$ENV{ASCEND_HOME_PATH}")
    set(_version_header "${_ascend_home}/include/version/cann_version.h")
    if(NOT EXISTS "${_version_header}")
        message(FATAL_ERROR "Cannot find CANN version header: ${_version_header}")
    endif()

    file(STRINGS "${_version_header}" _version_lines
         REGEX "^[ \\t]*#define[ \\t]+CANN_VERSION_NUM[ \\t]+")
    if(NOT _version_lines)
        message(FATAL_ERROR "Cannot find CANN_VERSION_NUM in ${_version_header}")
    endif()

    string(REGEX MATCH
           "([0-9]+)[ \\t]*\\*[ \\t]*10000000[^0-9]+([0-9]+)[ \\t]*\\*[ \\t]*100000[^0-9]+([0-9]+)[ \\t]*\\*[ \\t]*1000"
           _version_match "${_version_lines}")
    if(NOT _version_match)
        message(FATAL_ERROR
                "Cannot parse CANN_VERSION_NUM from ${_version_header}; expected CANN's major*10000000 + minor*100000 + patch*1000 expression")
    endif()

    set(_cann_major ${CMAKE_MATCH_1})
    set(_cann_minor ${CMAKE_MATCH_2})
    set(_cann_patch ${CMAKE_MATCH_3})
    if(_cann_major GREATER 999 OR _cann_minor GREATER 999 OR _cann_patch GREATER 999)
        message(FATAL_ERROR "CANN version fields exceed the 3-digit encoding in ${_version_header}")
    endif()

    math(EXPR _version_num
         "${_cann_major} * 10000000 + ${_cann_minor} * 100000 + ${_cann_patch} * 1000")
    set(${out_var} "${_version_num}" PARENT_SCOPE)
    message(STATUS "Detected DeepEP CANN version=${_version_num} from ${_version_header}")
endfunction()
