/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */
#ifndef DEEPEP_PROFILING_ADAPTERS_FUSED_DEEP_MOE_A5_PROFILE_COMPAT_H
#define DEEPEP_PROFILING_ADAPTERS_FUSED_DEEP_MOE_A5_PROFILE_COMPAT_H

#include "profiling/common/profile_protocol_common.h"
#include "profiling/adapters/fused_deep_moe_a5/fused_deep_moe_a5_profile_stage.h"

namespace Cam {

constexpr uint64_t FUSED_DEEP_MOE_PROFILE_MAGIC = PROFILE_MAGIC;
constexpr uint64_t FUSED_DEEP_MOE_PROFILE_VERSION = PROFILE_VERSION;
constexpr uint64_t FUSED_DEEP_MOE_PROFILE_CYCLE_TO_US = PROFILE_CYCLE_TO_US;
constexpr uint64_t FUSED_DEEP_MOE_PROFILE_FLAG_SESSION_BUFFER = PROFILE_FLAG_SESSION_BUFFER;
constexpr uint64_t FUSED_DEEP_MOE_PROFILE_CORE_TYPE_AIC = PROFILE_CORE_TYPE_AIC;
constexpr uint64_t FUSED_DEEP_MOE_PROFILE_CORE_TYPE_AIV = PROFILE_CORE_TYPE_AIV;
constexpr uint32_t FUSED_DEEP_MOE_PROFILE_AIC_COUNT_CAPACITY = PROFILE_AIC_COUNT_CAPACITY;
constexpr uint32_t FUSED_DEEP_MOE_PROFILE_AIV_COUNT_CAPACITY = PROFILE_AIV_COUNT_CAPACITY;
constexpr uint32_t FUSED_DEEP_MOE_PROFILE_LOGICAL_CORE_COUNT_CAPACITY = PROFILE_LOGICAL_CORE_COUNT_CAPACITY;
constexpr uint32_t FUSED_DEEP_MOE_PROFILE_ACTIVE_STAGE_CAPACITY = PROFILE_ACTIVE_STAGE_CAPACITY;
constexpr uint32_t FUSED_DEEP_MOE_PROFILE_RESERVED_STAGE_CAPACITY = PROFILE_RESERVED_STAGE_CAPACITY;
constexpr uint32_t FUSED_DEEP_MOE_PROFILE_MAX_GROUP_COUNT_CAPACITY = PROFILE_MAX_GROUP_COUNT_CAPACITY;

using FusedDeepMoeProfileStage = deep_ep::profiling::fused_deep_moe_a5::ProfileStage;
constexpr uint32_t FUSED_DEEP_MOE_PROFILE_STAGE_COUNT = static_cast<uint32_t>(FusedDeepMoeProfileStage::Count);

using FusedDeepMoeProfileStageLayout = ProfileStageLayout;
using FusedDeepMoeProfileHeader = ProfileHeader;
using FusedDeepMoeProfileRecord = ProfileRecord;

inline constexpr uint32_t GetFusedDeepMoeProfileStageBaseOffset(const FusedDeepMoeProfileStageLayout &layout,
                                                                uint32_t stageId)
{
    return GetProfileStageBaseOffset(layout, stageId);
}

inline constexpr uint32_t GetFusedDeepMoeProfileTotalOccurrences(const FusedDeepMoeProfileStageLayout &layout)
{
    return GetProfileTotalOccurrences(layout);
}

inline constexpr uint32_t GetFusedDeepMoeProfileRecordsPerLaunch(uint32_t logicalCoreCount,
                                                                 const FusedDeepMoeProfileStageLayout &layout)
{
    return GetProfileRecordsPerLaunch(logicalCoreCount, layout);
}

inline constexpr uint64_t GetFusedDeepMoeProfileLaunchOffset(uint32_t dataOffset, uint32_t recordsPerLaunch,
                                                             uint32_t recordBytes, uint64_t launchId)
{
    return GetProfileLaunchOffset(dataOffset, recordsPerLaunch, recordBytes, launchId);
}

inline constexpr uint32_t GetFusedDeepMoeProfileLogicalCoreLinear(uint64_t coreType, uint64_t coreIdx)
{
    return GetProfileLogicalCoreLinear(coreType, coreIdx);
}

inline constexpr uint32_t GetFusedDeepMoeProfileDataOffset()
{
    return GetProfileDataOffset();
}

}  // namespace Cam

#endif  // DEEPEP_PROFILING_ADAPTERS_FUSED_DEEP_MOE_A5_PROFILE_COMPAT_H
