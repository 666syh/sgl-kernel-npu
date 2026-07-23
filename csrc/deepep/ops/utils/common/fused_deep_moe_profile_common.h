/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */
#ifndef OPS_UTILS_COMMON_FUSED_DEEP_MOE_PROFILE_COMMON_H
#define OPS_UTILS_COMMON_FUSED_DEEP_MOE_PROFILE_COMMON_H

#include <cstdint>

#if defined(__CCE_AICORE__)
#define FUSED_DEEP_MOE_PROFILE_INLINE __aicore__ inline
#else
#define FUSED_DEEP_MOE_PROFILE_INLINE inline
#endif

namespace Cam {

constexpr uint64_t FUSED_DEEP_MOE_PROFILE_MAGIC = 0x46444D5035413031ULL;  // FDMP5A01
constexpr uint64_t FUSED_DEEP_MOE_PROFILE_VERSION = 1;
constexpr uint64_t FUSED_DEEP_MOE_PROFILE_CYCLE_TO_US = 1000;
constexpr uint64_t FUSED_DEEP_MOE_PROFILE_FLAG_SESSION_BUFFER = 0x1ULL;
constexpr uint64_t FUSED_DEEP_MOE_PROFILE_CORE_TYPE_AIC = 1ULL;
constexpr uint64_t FUSED_DEEP_MOE_PROFILE_CORE_TYPE_AIV = 2ULL;
constexpr uint32_t FUSED_DEEP_MOE_PROFILE_GROUP_COUNT_CAPACITY = 64U;
constexpr uint32_t FUSED_DEEP_MOE_PROFILE_AIC_PER_GROUP = 1U;
constexpr uint32_t FUSED_DEEP_MOE_PROFILE_AIV_PER_GROUP = 2U;
constexpr uint32_t FUSED_DEEP_MOE_PROFILE_AIC_COUNT_CAPACITY =
    FUSED_DEEP_MOE_PROFILE_GROUP_COUNT_CAPACITY * FUSED_DEEP_MOE_PROFILE_AIC_PER_GROUP;
constexpr uint32_t FUSED_DEEP_MOE_PROFILE_AIV_COUNT_CAPACITY =
    FUSED_DEEP_MOE_PROFILE_GROUP_COUNT_CAPACITY * FUSED_DEEP_MOE_PROFILE_AIV_PER_GROUP;
constexpr uint32_t FUSED_DEEP_MOE_PROFILE_STAGE_COUNT = 5U;
constexpr uint32_t FUSED_DEEP_MOE_PROFILE_LOGICAL_CORE_COUNT_CAPACITY =
    FUSED_DEEP_MOE_PROFILE_AIC_COUNT_CAPACITY + FUSED_DEEP_MOE_PROFILE_AIV_COUNT_CAPACITY;
constexpr uint32_t FUSED_DEEP_MOE_PROFILE_RECORDS_PER_LAUNCH =
    FUSED_DEEP_MOE_PROFILE_LOGICAL_CORE_COUNT_CAPACITY * FUSED_DEEP_MOE_PROFILE_STAGE_COUNT;

FUSED_DEEP_MOE_PROFILE_INLINE constexpr uint64_t PackProfileLaunchCounts(uint32_t launchCountCapacity,
                                                                         uint32_t launchCountCaptured)
{
    return (static_cast<uint64_t>(launchCountCapacity) << 32) | static_cast<uint64_t>(launchCountCaptured);
}

FUSED_DEEP_MOE_PROFILE_INLINE constexpr uint64_t PackProfileLayout0(uint16_t stageCount, uint16_t groupCountCapacity,
                                                                    uint16_t aicPerGroup, uint16_t aivPerGroup)
{
    return (static_cast<uint64_t>(stageCount) << 48) | (static_cast<uint64_t>(groupCountCapacity) << 32) |
           (static_cast<uint64_t>(aicPerGroup) << 16) | static_cast<uint64_t>(aivPerGroup);
}

FUSED_DEEP_MOE_PROFILE_INLINE constexpr uint64_t PackProfileLayout1(uint32_t logicalCoreCountCapacity,
                                                                    uint32_t recordsPerLaunch)
{
    return (static_cast<uint64_t>(logicalCoreCountCapacity) << 32) | static_cast<uint64_t>(recordsPerLaunch);
}

FUSED_DEEP_MOE_PROFILE_INLINE constexpr uint32_t UnpackProfileLaunchCapacity(uint64_t launchCountsPacked)
{
    return static_cast<uint32_t>(launchCountsPacked >> 32);
}

FUSED_DEEP_MOE_PROFILE_INLINE constexpr uint32_t UnpackProfileLaunchCaptured(uint64_t launchCountsPacked)
{
    return static_cast<uint32_t>(launchCountsPacked & 0xFFFFFFFFULL);
}

FUSED_DEEP_MOE_PROFILE_INLINE constexpr uint16_t UnpackProfileStageCount(uint64_t layoutPacked0)
{
    return static_cast<uint16_t>((layoutPacked0 >> 48) & 0xFFFFULL);
}

FUSED_DEEP_MOE_PROFILE_INLINE constexpr uint16_t UnpackProfileGroupCountCapacity(uint64_t layoutPacked0)
{
    return static_cast<uint16_t>((layoutPacked0 >> 32) & 0xFFFFULL);
}

FUSED_DEEP_MOE_PROFILE_INLINE constexpr uint16_t UnpackProfileAicPerGroup(uint64_t layoutPacked0)
{
    return static_cast<uint16_t>((layoutPacked0 >> 16) & 0xFFFFULL);
}

FUSED_DEEP_MOE_PROFILE_INLINE constexpr uint16_t UnpackProfileAivPerGroup(uint64_t layoutPacked0)
{
    return static_cast<uint16_t>(layoutPacked0 & 0xFFFFULL);
}

FUSED_DEEP_MOE_PROFILE_INLINE constexpr uint32_t UnpackProfileLogicalCoreCountCapacity(uint64_t layoutPacked1)
{
    return static_cast<uint32_t>(layoutPacked1 >> 32);
}

FUSED_DEEP_MOE_PROFILE_INLINE constexpr uint32_t UnpackProfileRecordsPerLaunch(uint64_t layoutPacked1)
{
    return static_cast<uint32_t>(layoutPacked1 & 0xFFFFFFFFULL);
}

FUSED_DEEP_MOE_PROFILE_INLINE constexpr uint32_t GetFusedDeepMoeProfileLogicalCoreLinear(uint64_t coreType,
                                                                                         uint64_t coreIdx)
{
    if (coreType == FUSED_DEEP_MOE_PROFILE_CORE_TYPE_AIC) {
        return (coreIdx < FUSED_DEEP_MOE_PROFILE_AIC_COUNT_CAPACITY) ? static_cast<uint32_t>(coreIdx) : UINT32_MAX;
    }
    if (coreType == FUSED_DEEP_MOE_PROFILE_CORE_TYPE_AIV) {
        return (coreIdx < FUSED_DEEP_MOE_PROFILE_AIV_COUNT_CAPACITY)
                   ? (FUSED_DEEP_MOE_PROFILE_AIC_COUNT_CAPACITY + static_cast<uint32_t>(coreIdx))
                   : UINT32_MAX;
    }
    return UINT32_MAX;
}

enum class FusedDeepMoeProfileStage : uint32_t {
    Dispatch = 0,
    Gmm1 = 1,
    SwigluQuant = 2,
    Gmm2 = 3,
    Combine = 4,
    Count = 5,
};

struct FusedDeepMoeProfileHeader {
    uint64_t magic;
    uint64_t version;
    uint64_t cycleToUs;
    uint64_t launchCountsPacked;
    uint64_t droppedLaunches;
    uint64_t layoutPacked0;
    uint64_t layoutPacked1;
    uint64_t flags;
};

struct FusedDeepMoeProfileRecord {
    uint64_t coreType;
    uint64_t coreIdx;
    uint64_t stageId;
    uint64_t launchId;
    uint64_t startCycle;
    uint64_t endCycle;
    uint64_t reserved0;
    uint64_t reserved1;
};

static_assert(sizeof(FusedDeepMoeProfileHeader) == 64, "Unexpected fused profile header size");
static_assert(sizeof(FusedDeepMoeProfileRecord) == 64, "Unexpected fused profile record size");

}  // namespace Cam

#undef FUSED_DEEP_MOE_PROFILE_INLINE

#endif  // OPS_UTILS_COMMON_FUSED_DEEP_MOE_PROFILE_COMMON_H
