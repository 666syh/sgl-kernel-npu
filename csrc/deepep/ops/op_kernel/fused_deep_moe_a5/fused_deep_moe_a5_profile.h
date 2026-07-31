/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */
#ifndef FUSED_DEEP_MOE_A5_PROFILE_H
#define FUSED_DEEP_MOE_A5_PROFILE_H

#include "../../utils/common/fused_deep_moe_profile_common.h"
#include <kernel_operator.h>

namespace Cam {

struct FusedDeepMoeProfileWriter {
    __gm__ FusedDeepMoeProfileRecord *records{nullptr};
    bool enabled{false};
    uint32_t coreIdx{0};
    uint32_t coreType{0};
    uint32_t stageCount{0};
    uint32_t logicalCoreCount{0};
    uint32_t recordsPerLaunch{0};
    uint32_t launchId{0};
    uint64_t profileBufferBytes{0};
    __gm__ const FusedDeepMoeProfileStageLayout *stageLayout{nullptr};

    __aicore__ inline void Init(GM_ADDR profileGM, bool enable, uint32_t launchId_, uint32_t coreType_,
                                uint64_t profileBufferBytes_)
    {
        enabled = enable;
        if (!enabled) {
            return;
        }
        if (profileGM == nullptr) {
            enabled = false;
            return;
        }
        coreIdx = AscendC::GetBlockIdx();
        coreType = coreType_;
        launchId = launchId_;
        profileBufferBytes = profileBufferBytes_;
        stageCount = static_cast<uint32_t>(FusedDeepMoeProfileStage::Count);
        auto *base = reinterpret_cast<__gm__ uint8_t *>(profileGM);
        auto *header = reinterpret_cast<__gm__ FusedDeepMoeProfileHeader *>(base);
        if (header == nullptr || header->magic != FUSED_DEEP_MOE_PROFILE_MAGIC ||
            header->version != FUSED_DEEP_MOE_PROFILE_VERSION || header->cycleToUs == 0 ||
            (UnpackProfileFlags(header->flagsPacked) &
             static_cast<uint32_t>(FUSED_DEEP_MOE_PROFILE_FLAG_SESSION_BUFFER)) == 0U) {
            enabled = false;
            return;
        }
        uint64_t launchCountCapacity = UnpackProfileLaunchCapacity(header->launchCountsPacked);
        uint32_t layoutStageCount = UnpackProfileStageCount(header->layoutPacked0);
        uint32_t layoutGroupCountCapacity = UnpackProfileGroupCountCapacity(header->layoutPacked0);
        logicalCoreCount = UnpackProfileLogicalCoreCountCapacity(header->layoutPacked1);
        recordsPerLaunch = UnpackProfileRecordsPerLaunch(header->layoutPacked1);
        stageLayout =
            reinterpret_cast<__gm__ const FusedDeepMoeProfileStageLayout *>(base + sizeof(FusedDeepMoeProfileHeader));
        uint32_t layoutActiveStageCapacity = static_cast<uint32_t>(stageLayout->activeStageCapacity);
        uint32_t layoutRecordsPerLaunch = GetFusedDeepMoeProfileRecordsPerLaunch(logicalCoreCount, *stageLayout);
        if (launchId >= launchCountCapacity || layoutStageCount == 0U ||
            layoutStageCount > FUSED_DEEP_MOE_PROFILE_ACTIVE_STAGE_CAPACITY ||
            layoutStageCount > FUSED_DEEP_MOE_PROFILE_RESERVED_STAGE_CAPACITY || layoutStageCount != stageCount ||
            stageLayout->stageCount != stageCount ||
            layoutActiveStageCapacity != FUSED_DEEP_MOE_PROFILE_ACTIVE_STAGE_CAPACITY ||
            layoutActiveStageCapacity > FUSED_DEEP_MOE_PROFILE_RESERVED_STAGE_CAPACITY ||
            layoutGroupCountCapacity == 0U ||
            layoutGroupCountCapacity > FUSED_DEEP_MOE_PROFILE_MAX_GROUP_COUNT_CAPACITY ||
            logicalCoreCount != FUSED_DEEP_MOE_PROFILE_LOGICAL_CORE_COUNT_CAPACITY ||
            layoutRecordsPerLaunch != recordsPerLaunch || recordsPerLaunch == 0U) {
            if (coreType == FUSED_DEEP_MOE_PROFILE_CORE_TYPE_AIC && coreIdx == 0) {
                uint32_t flags = UnpackProfileFlags(header->flagsPacked);
                uint32_t droppedLaunches = UnpackProfileDroppedLaunches(header->flagsPacked) + 1U;
                header->flagsPacked = PackProfileFlags(flags, droppedLaunches);
            }
            enabled = false;
            return;
        }
        uint64_t alignedLaunchBytes =
            (static_cast<uint64_t>(recordsPerLaunch) * sizeof(FusedDeepMoeProfileRecord) + 63ULL) / 64ULL * 64ULL;
        uint64_t requiredBytes =
            GetFusedDeepMoeProfileDataOffset() + (static_cast<uint64_t>(launchId) + 1ULL) * alignedLaunchBytes;
        if (requiredBytes > profileBufferBytes) {
            if (coreType == FUSED_DEEP_MOE_PROFILE_CORE_TYPE_AIC && coreIdx == 0) {
                uint32_t flags = UnpackProfileFlags(header->flagsPacked);
                uint32_t droppedLaunches = UnpackProfileDroppedLaunches(header->flagsPacked) + 1U;
                header->flagsPacked = PackProfileFlags(flags, droppedLaunches);
            }
            enabled = false;
            return;
        }
        uint64_t launchOffset =
            GetFusedDeepMoeProfileLaunchOffset(GetFusedDeepMoeProfileDataOffset(), recordsPerLaunch,
                                               static_cast<uint32_t>(sizeof(FusedDeepMoeProfileRecord)), launchId);
        records = reinterpret_cast<__gm__ FusedDeepMoeProfileRecord *>(base + launchOffset);
    }

    __aicore__ inline uint64_t Now() const
    {
        return static_cast<uint64_t>(AscendC::GetSystemCycle());
    }

    __aicore__ inline void Record(FusedDeepMoeProfileStage stage, uint64_t startCycle, uint64_t endCycle) const
    {
        Record(stage, 0U, startCycle, endCycle);
    }

    __aicore__ inline void Record(FusedDeepMoeProfileStage stage, uint32_t occurrenceId, uint64_t startCycle,
                                  uint64_t endCycle) const
    {
        if (!enabled) {
            return;
        }
        uint32_t stageId = static_cast<uint32_t>(stage);
        if (stageId >= stageCount) {
            return;
        }
        if (stageLayout == nullptr) {
            return;
        }
        uint32_t stageOccurrenceCount = GetProfileStageOccurrenceCount(*stageLayout, stageId);
        if (occurrenceId >= stageOccurrenceCount) {
            return;
        }
        uint32_t logicalCoreLinear = GetFusedDeepMoeProfileLogicalCoreLinear(coreType, coreIdx);
        if (logicalCoreLinear == UINT32_MAX || logicalCoreLinear >= logicalCoreCount) {
            return;
        }
        uint32_t stageBase = GetFusedDeepMoeProfileStageBaseOffset(*stageLayout, stageId);
        uint64_t slot = (static_cast<uint64_t>(stageBase) + static_cast<uint64_t>(occurrenceId)) *
                            static_cast<uint64_t>(logicalCoreCount) +
                        static_cast<uint64_t>(logicalCoreLinear);
        if (slot >= recordsPerLaunch) {
            return;
        }
        records[slot].coreType = coreType;
        records[slot].coreIdx = coreIdx;
        records[slot].stageId = stageId;
        records[slot].occurrenceId = occurrenceId;
        records[slot].launchId = launchId;
        records[slot].startCycle = startCycle;
        records[slot].endCycle = endCycle;
        records[slot].reserved0 = 0;
    }
};

}  // namespace Cam

#endif  // FUSED_DEEP_MOE_A5_PROFILE_H
