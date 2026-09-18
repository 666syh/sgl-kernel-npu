#ifndef DEEPEP_PROFILING_ADAPTERS_MOE_LOW_LATENCY_DISPATCH_V2_A5_PROFILE_ADAPTER_HPP
#define DEEPEP_PROFILING_ADAPTERS_MOE_LOW_LATENCY_DISPATCH_V2_A5_PROFILE_ADAPTER_HPP

#include <cstdint>

#include "profiling/core/profile_runtime.hpp"

namespace deep_ep::profiling::moe_low_latency_dispatch_v2_a5 {

using LaunchContext = runtime::ProfileLaunchContext;

LaunchContext PrepareLaunch(bool profileEnable);
void CompleteLaunch(const LaunchContext &ctx, int64_t rank);

}  // namespace deep_ep::profiling::moe_low_latency_dispatch_v2_a5

#endif  // DEEPEP_PROFILING_ADAPTERS_MOE_LOW_LATENCY_DISPATCH_V2_A5_PROFILE_ADAPTER_HPP
