/*
 * SPDX-License-Identifier: MIT
 */

#include "profiling/adapters/fused_deep_moe_a5/fused_deep_moe_a5_profile_traits.hpp"

#include <sstream>

#include "exception.hpp"

namespace deep_ep::profiling::fused_deep_moe_a5 {

const ProfileSchema &GetProfileSchema()
{
    static const ProfileSchema schema{
        "fused_deep_moe",
        kStageCount,
        Cam::PROFILE_ACTIVE_STAGE_CAPACITY,
        {Cam::PROFILE_AIC_COUNT_CAPACITY, Cam::PROFILE_AIV_COUNT_CAPACITY, Cam::PROFILE_LOGICAL_CORE_COUNT_CAPACITY},
        &GetStageName,
        &GetStageDisplayName,
    };
    return schema;
}

const char *GetStageName(uint64_t stageId)
{
    switch (static_cast<ProfileStage>(stageId)) {
        case ProfileStage::Dispatch:
            return "dispatch";
        case ProfileStage::Gmm1:
            return "gmm1";
        case ProfileStage::SwigluQuant:
            return "swiglu_quant";
        case ProfileStage::Gmm2:
            return "gmm2";
        case ProfileStage::Combine:
            return "combine";
        case ProfileStage::WeightSum:
            return "weight_sum";
        default:
            return "unknown";
    }
}

std::string GetStageDisplayName(uint64_t stageId, uint64_t occurrenceId, const Cam::ProfileStageLayout &stageLayout)
{
    std::ostringstream oss;
    oss << GetStageName(stageId);
    uint32_t stageOccurrenceCount = Cam::GetProfileStageOccurrenceCount(stageLayout, static_cast<uint32_t>(stageId));
    auto stage = static_cast<ProfileStage>(stageId);
    if (stage == ProfileStage::Dispatch || stage == ProfileStage::Gmm1 || stage == ProfileStage::Gmm2 ||
        stage == ProfileStage::Combine) {
        oss << "[group=" << occurrenceId << "]";
    } else if (stageOccurrenceCount > 1U || occurrenceId != 0U) {
        oss << "[occ=" << occurrenceId << "]";
    }
    return oss.str();
}

Cam::ProfileStageLayout BuildStageLayout(uint32_t groupCountCapacity)
{
    EP_HOST_ASSERT_S(groupCountCapacity >= 1U && groupCountCapacity <= Cam::PROFILE_MAX_GROUP_COUNT_CAPACITY,
                     "groupCountCapacity must be in [1, 64].");
    Cam::ProfileStageLayout layout{};
    layout.stageCount = static_cast<uint16_t>(kStageCount);
    layout.activeStageCapacity = static_cast<uint16_t>(Cam::PROFILE_ACTIVE_STAGE_CAPACITY);
    EP_HOST_ASSERT_S(
        Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::Dispatch), groupCountCapacity),
        "invalid dispatch occurrence capacity.");
    EP_HOST_ASSERT_S(
        Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::Gmm1), groupCountCapacity),
        "invalid gmm1 occurrence capacity.");
    EP_HOST_ASSERT_S(Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::SwigluQuant), 1U),
                     "invalid swiglu occurrence capacity.");
    EP_HOST_ASSERT_S(
        Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::Gmm2), groupCountCapacity),
        "invalid gmm2 occurrence capacity.");
    EP_HOST_ASSERT_S(
        Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::Combine), groupCountCapacity),
        "invalid combine occurrence capacity.");
    EP_HOST_ASSERT_S(Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::WeightSum), 1U),
                     "invalid weight sum occurrence capacity.");
    return layout;
}

uint32_t GetGroupCountCapacity(int64_t numExperts, int64_t numRanks)
{
    EP_HOST_ASSERT_S(numRanks > 0, "num_ranks must be positive for fused deep moe profiling.");
    EP_HOST_ASSERT_S(numExperts > 0, "num_experts must be positive for fused deep moe profiling.");
    EP_HOST_ASSERT_S(numExperts % numRanks == 0,
                     "num_experts must be divisible by num_ranks for fused deep moe profiling.");
    return static_cast<uint32_t>(numExperts / numRanks);
}

}  // namespace deep_ep::profiling::fused_deep_moe_a5
