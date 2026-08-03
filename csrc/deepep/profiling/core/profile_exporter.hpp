/*
 * SPDX-License-Identifier: MIT
 */
#ifndef DEEPEP_PROFILING_CORE_PROFILE_EXPORTER_HPP
#define DEEPEP_PROFILING_CORE_PROFILE_EXPORTER_HPP

#include <string>

#include <torch/types.h>

#include "profiling/core/profile_schema.hpp"

namespace deep_ep::profiling::exporter {

void ExportBufferToTrace(const at::Tensor &profileBuffer, int64_t rank, const std::string &profileTraceDir,
                         int64_t numWarmups, int64_t launchCountCaptured, const ProfileSchema &schema);

}  // namespace deep_ep::profiling::exporter

#endif  // DEEPEP_PROFILING_CORE_PROFILE_EXPORTER_HPP
