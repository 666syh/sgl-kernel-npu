#ifndef DEEPEP_PROFILING_ADAPTERS_MOE_LOW_LATENCY_COMBINE_V2_A5_PROFILE_TRAITS_HPP
#define DEEPEP_PROFILING_ADAPTERS_MOE_LOW_LATENCY_COMBINE_V2_A5_PROFILE_TRAITS_HPP

#include <string>

#include "profiling/adapters/moe_low_latency_combine_v2_a5/moe_low_latency_combine_v2_a5_profile_stage.h"
#include "profiling/common/profile_protocol_common.h"
#include "profiling/core/profile_schema.hpp"

namespace deep_ep::profiling::moe_low_latency_combine_v2_a5 {

static_assert(kStageCount <= Cam::PROFILE_ACTIVE_STAGE_CAPACITY,
              "moe_low_latency_combine_v2_a5 stage count must fit in active profiling stage capacity");

const ProfileSchema &GetProfileSchema();
const ProfileOpRegistration &GetProfileRegistration();
const char *GetLaunchEventName();
const char *GetStageName(uint64_t stageId);
std::string GetStageDisplayName(uint64_t stageId, uint64_t occurrenceId, const Cam::ProfileStageLayout &stageLayout);
std::string GetPrivateDataJson(uint64_t stageId, uint64_t occurrenceId, const Cam::ProfileRecord &record,
                               const Cam::ProfileStageLayout &stageLayout);
Cam::ProfileStageLayout BuildStageLayout();

}  // namespace deep_ep::profiling::moe_low_latency_combine_v2_a5

#endif  // DEEPEP_PROFILING_ADAPTERS_MOE_LOW_LATENCY_COMBINE_V2_A5_PROFILE_TRAITS_HPP
