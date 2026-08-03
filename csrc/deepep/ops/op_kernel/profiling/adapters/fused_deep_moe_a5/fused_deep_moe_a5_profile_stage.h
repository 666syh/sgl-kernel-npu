/*
 * SPDX-License-Identifier: MIT
 */
#ifndef DEEPEP_OP_KERNEL_PROFILING_ADAPTERS_FUSED_DEEP_MOE_A5_PROFILE_STAGE_H
#define DEEPEP_OP_KERNEL_PROFILING_ADAPTERS_FUSED_DEEP_MOE_A5_PROFILE_STAGE_H

#include <cstdint>

namespace deep_ep::profiling::fused_deep_moe_a5 {

enum class ProfileStage : uint32_t {
    Dispatch = 0,
    Gmm1 = 1,
    SwigluQuant = 2,
    Gmm2 = 3,
    Combine = 4,
    WeightSum = 5,
    Count = 6,
};

constexpr uint32_t kStageCount = static_cast<uint32_t>(ProfileStage::Count);

}  // namespace deep_ep::profiling::fused_deep_moe_a5

#endif  // DEEPEP_OP_KERNEL_PROFILING_ADAPTERS_FUSED_DEEP_MOE_A5_PROFILE_STAGE_H
