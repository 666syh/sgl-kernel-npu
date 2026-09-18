#ifndef DEEPEP_OP_KERNEL_PROFILING_ADAPTERS_MOE_LOW_LATENCY_DISPATCH_V2_A5_PROFILE_STAGE_H
#define DEEPEP_OP_KERNEL_PROFILING_ADAPTERS_MOE_LOW_LATENCY_DISPATCH_V2_A5_PROFILE_STAGE_H

#include <cstdint>

namespace deep_ep::profiling::moe_low_latency_dispatch_v2_a5 {

enum class ProfileStage : uint32_t {
    AlltoallDispatch = 0,
    SetStatus = 1,
    WaitDispatch = 2,
    LocalWindowCopy = 3,
    AllgatherSetStatusAndWait = 4,
    AllgatherProcessOut = 5,
    UpdateTokenNumsOut = 6,
    Count = 7,
};

constexpr uint32_t kStageCount = static_cast<uint32_t>(ProfileStage::Count);

}  // namespace deep_ep::profiling::moe_low_latency_dispatch_v2_a5

#endif  // DEEPEP_OP_KERNEL_PROFILING_ADAPTERS_MOE_LOW_LATENCY_DISPATCH_V2_A5_PROFILE_STAGE_H
