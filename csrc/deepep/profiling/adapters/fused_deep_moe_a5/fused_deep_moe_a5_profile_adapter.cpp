/*
 * SPDX-License-Identifier: MIT
 */

#include "profiling/adapters/fused_deep_moe_a5/fused_deep_moe_a5_profile_adapter.hpp"

#include "profiling/adapters/fused_deep_moe_a5/fused_deep_moe_a5_profile_traits.hpp"
#include "profiling/core/profile_runtime.hpp"
#include "profiling/core/profile_session.hpp"

namespace deep_ep::profiling::fused_deep_moe_a5 {

bool IsDebugEnabled()
{
    return runtime::IsDebugEnabled();
}
void DebugPrint(const std::string &msg)
{
    runtime::DebugPrint(msg);
}
bool IsActive()
{
    return runtime::IsSessionActive();
}

void BeginProfile(int64_t numProfileSkipLaunches, int64_t numProfileActiveLaunches, const std::string &profileTraceDir)
{
    runtime::BeginSession(GetProfileSchema(), numProfileSkipLaunches, numProfileActiveLaunches, profileTraceDir);
}

void EndProfile(int64_t rank)
{
    runtime::EndSession(rank);
}

LaunchContext PrepareLaunch(int64_t numExperts, int64_t numRanks, bool profileEnable)
{
    uint32_t groupCountCapacity = GetGroupCountCapacity(numExperts, numRanks);
    auto stageLayout = BuildStageLayout(groupCountCapacity);
    return runtime::PrepareLaunch(profileEnable, groupCountCapacity, stageLayout, GetProfileSchema());
}

void CompleteLaunch(const LaunchContext &ctx, int64_t rank)
{
    runtime::CompleteLaunch(ctx, rank, GetProfileSchema());
}

int64_t GetExpectedLaunches()
{
    return session::GetExpectedLaunches();
}
int64_t GetCapturedLaunches()
{
    return session::GetCapturedLaunches();
}
uint32_t GetLaunchCountCapacity()
{
    return session::GetLaunchCountCapacity();
}
std::string GetProfileTraceDir()
{
    return session::GetProfileTraceDir();
}
int64_t GetNumProfileSkipLaunches()
{
    return session::GetNumProfileSkipLaunches();
}
int64_t GetNumProfileActiveLaunches()
{
    return session::GetNumProfileActiveLaunches();
}

}  // namespace deep_ep::profiling::fused_deep_moe_a5
