#include "profiling/adapters/moe_low_latency_dispatch_v2_a5/moe_low_latency_dispatch_v2_a5_profile_traits.hpp"

#include "exception.hpp"

namespace deep_ep::profiling::moe_low_latency_dispatch_v2_a5 {

const ProfileSchema &GetProfileSchema()
{
    static const ProfileSchema schema{
        "moe_low_latency_dispatch_v2",
        kStageCount,
        Cam::PROFILE_ACTIVE_STAGE_CAPACITY,
        {Cam::PROFILE_AIC_COUNT_CAPACITY, Cam::PROFILE_AIV_COUNT_CAPACITY, Cam::PROFILE_LOGICAL_CORE_COUNT_CAPACITY},
        &GetStageName,
        &GetStageDisplayName,
        &GetPrivateDataJson,
    };
    return schema;
}

const ProfileOpRegistration &GetProfileRegistration()
{
    static const ProfileOpRegistration registration{
        "moe_low_latency_dispatch_v2_a5",
        &GetProfileSchema,
        &GetLaunchEventName,
    };
    return registration;
}

const char *GetLaunchEventName()
{
    return "moe_low_latency_dispatch_v2_launch";
}

const char *GetStageName(uint64_t stageId)
{
    switch (static_cast<ProfileStage>(stageId)) {
        case ProfileStage::AlltoallDispatch:
            return "alltoall_dispatch";
        case ProfileStage::SetStatus:
            return "set_status";
        case ProfileStage::WaitDispatch:
            return "wait_dispatch";
        case ProfileStage::LocalWindowCopy:
            return "local_window_copy";
        case ProfileStage::AllgatherSetStatusAndWait:
            return "allgather_set_status_and_wait";
        case ProfileStage::AllgatherProcessOut:
            return "allgather_process_out";
        case ProfileStage::UpdateTokenNumsOut:
            return "update_token_nums_out";
        default:
            return "unknown";
    }
}

std::string GetStageDisplayName(uint64_t stageId, uint64_t occurrenceId, const Cam::ProfileStageLayout &stageLayout)
{
    (void)occurrenceId;
    (void)stageLayout;
    return GetStageName(stageId);
}

std::string GetPrivateDataJson(uint64_t stageId, uint64_t occurrenceId, const Cam::ProfileRecord &record,
                               const Cam::ProfileStageLayout &stageLayout)
{
    (void)stageId;
    (void)occurrenceId;
    (void)record;
    (void)stageLayout;
    return {};
}

Cam::ProfileStageLayout BuildStageLayout()
{
    Cam::ProfileStageLayout layout{};
    layout.stageCount = static_cast<uint16_t>(kStageCount);
    layout.activeStageCapacity = static_cast<uint16_t>(Cam::PROFILE_ACTIVE_STAGE_CAPACITY);
    for (uint32_t stage = 0U; stage < kStageCount; ++stage) {
        EP_HOST_ASSERT_S(Cam::SetProfileStageOccurrenceCount(layout, stage, 1U),
                         "invalid low-latency dispatch stage occurrence capacity.");
    }
    return layout;
}

}  // namespace deep_ep::profiling::moe_low_latency_dispatch_v2_a5
