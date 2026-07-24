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
constexpr uint64_t FUSED_DEEP_MOE_PROFILE_VERSION = 2;
constexpr uint64_t FUSED_DEEP_MOE_PROFILE_CYCLE_TO_US = 1000;
constexpr uint64_t FUSED_DEEP_MOE_PROFILE_FLAG_SESSION_BUFFER = 0x1ULL;
constexpr uint64_t FUSED_DEEP_MOE_PROFILE_CORE_TYPE_AIC = 1ULL;
constexpr uint64_t FUSED_DEEP_MOE_PROFILE_CORE_TYPE_AIV = 2ULL;
constexpr uint32_t FUSED_DEEP_MOE_PROFILE_AIC_COUNT_CAPACITY = 36U;
constexpr uint32_t FUSED_DEEP_MOE_PROFILE_AIV_COUNT_CAPACITY = 72U;
constexpr uint32_t FUSED_DEEP_MOE_PROFILE_STAGE_COUNT = 5U;
constexpr uint32_t FUSED_DEEP_MOE_PROFILE_LOGICAL_CORE_COUNT_CAPACITY =
    FUSED_DEEP_MOE_PROFILE_AIC_COUNT_CAPACITY + FUSED_DEEP_MOE_PROFILE_AIV_COUNT_CAPACITY;
constexpr uint32_t FUSED_DEEP_MOE_PROFILE_STAGE_OCCURRENCE_BITS = 12U;
constexpr uint64_t FUSED_DEEP_MOE_PROFILE_STAGE_OCCURRENCE_MASK =
    (1ULL << FUSED_DEEP_MOE_PROFILE_STAGE_OCCURRENCE_BITS) - 1ULL;

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

FUSED_DEEP_MOE_PROFILE_INLINE constexpr uint64_t PackProfileStageOccurrences(uint16_t dispatchOccurrence,
                                                                             uint16_t gmm1Occurrence,
                                                                             uint16_t swigluOccurrence,
                                                                             uint16_t gmm2Occurrence,
                                                                             uint16_t combineOccurrence)
{
    return (static_cast<uint64_t>(dispatchOccurrence) << (0U * FUSED_DEEP_MOE_PROFILE_STAGE_OCCURRENCE_BITS)) |
           (static_cast<uint64_t>(gmm1Occurrence) << (1U * FUSED_DEEP_MOE_PROFILE_STAGE_OCCURRENCE_BITS)) |
           (static_cast<uint64_t>(swigluOccurrence) << (2U * FUSED_DEEP_MOE_PROFILE_STAGE_OCCURRENCE_BITS)) |
           (static_cast<uint64_t>(gmm2Occurrence) << (3U * FUSED_DEEP_MOE_PROFILE_STAGE_OCCURRENCE_BITS)) |
           (static_cast<uint64_t>(combineOccurrence) << (4U * FUSED_DEEP_MOE_PROFILE_STAGE_OCCURRENCE_BITS));
}

FUSED_DEEP_MOE_PROFILE_INLINE constexpr uint64_t PackProfileFlags(uint32_t flags, uint32_t droppedLaunches)
{
    return (static_cast<uint64_t>(droppedLaunches) << 32) | static_cast<uint64_t>(flags);
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

FUSED_DEEP_MOE_PROFILE_INLINE constexpr uint32_t UnpackProfileFlags(uint64_t flagsPacked)
{
    return static_cast<uint32_t>(flagsPacked & 0xFFFFFFFFULL);
}

FUSED_DEEP_MOE_PROFILE_INLINE constexpr uint32_t UnpackProfileDroppedLaunches(uint64_t flagsPacked)
{
    return static_cast<uint32_t>(flagsPacked >> 32);
}

FUSED_DEEP_MOE_PROFILE_INLINE constexpr uint32_t UnpackProfileStageOccurrenceCount(uint64_t stageOccurrencesPacked,
                                                                                   uint32_t stageId)
{
    if (stageId >= FUSED_DEEP_MOE_PROFILE_STAGE_COUNT) {
        return 0U;
    }
    return static_cast<uint32_t>((stageOccurrencesPacked >> (stageId * FUSED_DEEP_MOE_PROFILE_STAGE_OCCURRENCE_BITS)) &
                                 FUSED_DEEP_MOE_PROFILE_STAGE_OCCURRENCE_MASK);
}

FUSED_DEEP_MOE_PROFILE_INLINE constexpr uint32_t GetFusedDeepMoeProfileStageBaseOffset(uint64_t stageOccurrencesPacked,
                                                                                       uint32_t stageId)
{
    uint32_t base = 0U;
    for (uint32_t i = 0U; i < stageId && i < FUSED_DEEP_MOE_PROFILE_STAGE_COUNT; ++i) {
        base += UnpackProfileStageOccurrenceCount(stageOccurrencesPacked, i);
    }
    return base;
}

FUSED_DEEP_MOE_PROFILE_INLINE constexpr uint32_t GetFusedDeepMoeProfileTotalOccurrences(uint64_t stageOccurrencesPacked)
{
    uint32_t total = 0U;
    for (uint32_t i = 0U; i < FUSED_DEEP_MOE_PROFILE_STAGE_COUNT; ++i) {
        total += UnpackProfileStageOccurrenceCount(stageOccurrencesPacked, i);
    }
    return total;
}

FUSED_DEEP_MOE_PROFILE_INLINE constexpr uint32_t GetFusedDeepMoeProfileRecordsPerLaunch(uint32_t logicalCoreCount,
                                                                                        uint64_t stageOccurrencesPacked)
{
    return logicalCoreCount * GetFusedDeepMoeProfileTotalOccurrences(stageOccurrencesPacked);
}

FUSED_DEEP_MOE_PROFILE_INLINE constexpr uint64_t GetFusedDeepMoeProfileLaunchOffset(uint32_t headerBytes,
                                                                                    uint32_t recordsPerLaunch,
                                                                                    uint32_t recordBytes,
                                                                                    uint64_t launchId)
{
    return static_cast<uint64_t>(headerBytes) +
           launchId * static_cast<uint64_t>(recordsPerLaunch) * static_cast<uint64_t>(recordBytes);
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
    uint64_t layoutPacked0;
    uint64_t layoutPacked1;
    uint64_t stageOccurrencesPacked;
    uint64_t flagsPacked;
};

struct FusedDeepMoeProfileRecord {
    uint64_t coreType;
    uint64_t coreIdx;
    uint64_t stageId;
    uint64_t occurrenceId;
    uint64_t launchId;
    uint64_t startCycle;
    uint64_t endCycle;
    uint64_t reserved0;
};

static_assert(sizeof(FusedDeepMoeProfileHeader) == 64, "Unexpected fused profile header size");
static_assert(sizeof(FusedDeepMoeProfileRecord) == 64, "Unexpected fused profile record size");

}  // namespace Cam

#undef FUSED_DEEP_MOE_PROFILE_INLINE

#endif  // OPS_UTILS_COMMON_FUSED_DEEP_MOE_PROFILE_COMMON_H
