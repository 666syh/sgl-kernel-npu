#include "profiling/adapters/moe_low_latency_dispatch_v2_a5/moe_low_latency_dispatch_v2_a5_profile_adapter.hpp"

#include "profiling/adapters/moe_low_latency_dispatch_v2_a5/moe_low_latency_dispatch_v2_a5_profile_traits.hpp"

namespace deep_ep::profiling::moe_low_latency_dispatch_v2_a5 {

LaunchContext PrepareLaunch(bool profileEnable)
{
    ProfileLaunchConfig launchConfig{};
    launchConfig.groupCountCapacity = 1U;
    launchConfig.stageLayout = BuildStageLayout();
    return runtime::PrepareLaunch(GetProfileRegistration(), launchConfig, profileEnable);
}

void CompleteLaunch(const LaunchContext &ctx, int64_t rank)
{
    runtime::CompleteLaunch(ctx, rank);
}

}  // namespace deep_ep::profiling::moe_low_latency_dispatch_v2_a5
