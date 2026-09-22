#ifndef DEEPEP_PROFILING_ADAPTERS_MOE_LOW_LATENCY_COMBINE_V2_A5_PROFILE_STAGE_H
#define DEEPEP_PROFILING_ADAPTERS_MOE_LOW_LATENCY_COMBINE_V2_A5_PROFILE_STAGE_H

#include <cstdint>

namespace deep_ep::profiling::moe_low_latency_combine_v2_a5 {

enum class ProfileStage : uint32_t {
    ReduceScatter = 0,
    BufferInit = 1,
    SetTpStatusAndDispatch = 2,
    AlltoAllBufferInitAndMaskCal = 3,
    LocalWindowCopy = 4,
    Count = 5,
};

constexpr uint32_t kStageCount = static_cast<uint32_t>(ProfileStage::Count);

}  // namespace deep_ep::profiling::moe_low_latency_combine_v2_a5

#endif  // DEEPEP_PROFILING_ADAPTERS_MOE_LOW_LATENCY_COMBINE_V2_A5_PROFILE_STAGE_H
