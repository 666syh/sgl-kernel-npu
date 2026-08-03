/*
 * SPDX-License-Identifier: MIT
 */
#ifndef DEEPEP_PROFILING_CORE_PROFILE_DEBUG_HPP
#define DEEPEP_PROFILING_CORE_PROFILE_DEBUG_HPP

#include <string>

namespace deep_ep::profiling::debug {

bool IsEnabled();
void Print(const std::string &msg);

}  // namespace deep_ep::profiling::debug

#endif  // DEEPEP_PROFILING_CORE_PROFILE_DEBUG_HPP
