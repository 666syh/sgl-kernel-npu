/*
 * SPDX-License-Identifier: MIT
 */
#ifndef DEEPEP_PROFILING_ADAPTERS_FUSED_DEEP_MOE_A5_PROFILE_ADAPTER_HPP
#define DEEPEP_PROFILING_ADAPTERS_FUSED_DEEP_MOE_A5_PROFILE_ADAPTER_HPP

#include <cstdint>
#include <string>

#include "profiling/core/profile_runtime.hpp"

namespace deep_ep::profiling::fused_deep_moe_a5 {

using LaunchContext = runtime::ProfileLaunchContext;

bool IsDebugEnabled();
void DebugPrint(const std::string &msg);
bool IsActive();

void BeginProfile(int64_t numProfileSkipLaunches, int64_t numProfileActiveLaunches, const std::string &profileTraceDir);
void EndProfile(int64_t rank);

LaunchContext PrepareLaunch(int64_t numExperts, int64_t numRanks, bool profileEnable);
void CompleteLaunch(const LaunchContext &ctx, int64_t rank);

int64_t GetExpectedLaunches();
int64_t GetCapturedLaunches();
uint32_t GetLaunchCountCapacity();
std::string GetProfileTraceDir();
int64_t GetNumProfileSkipLaunches();
int64_t GetNumProfileActiveLaunches();

}  // namespace deep_ep::profiling::fused_deep_moe_a5

#endif  // DEEPEP_PROFILING_ADAPTERS_FUSED_DEEP_MOE_A5_PROFILE_ADAPTER_HPP
