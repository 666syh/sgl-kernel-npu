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
    uint32_t recordsPerLaunch{0};
    uint32_t launchId{0};
    uint64_t profileBufferBytes{0};

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
        recordsPerLaunch = FUSED_DEEP_MOE_PROFILE_RECORDS_PER_LAUNCH;
        auto *base = reinterpret_cast<__gm__ uint8_t *>(profileGM);
        auto *header = reinterpret_cast<__gm__ FusedDeepMoeProfileHeader *>(base);
        if (header == nullptr || header->magic != FUSED_DEEP_MOE_PROFILE_MAGIC ||
            header->version != FUSED_DEEP_MOE_PROFILE_VERSION || header->cycleToUs == 0) {
            enabled = false;
            return;
        }
        uint64_t launchCountCapacity = UnpackProfileLaunchCapacity(header->launchCountsPacked);
        uint32_t layoutStageCount = UnpackProfileStageCount(header->layoutPacked0);
        uint32_t layoutLogicalCoreCount = UnpackProfileLogicalCoreCountCapacity(header->layoutPacked1);
        uint32_t layoutRecordsPerLaunch = UnpackProfileRecordsPerLaunch(header->layoutPacked1);
        if (launchId >= launchCountCapacity || layoutStageCount != stageCount ||
            layoutRecordsPerLaunch != recordsPerLaunch ||
            layoutLogicalCoreCount != FUSED_DEEP_MOE_PROFILE_LOGICAL_CORE_COUNT_CAPACITY) {
            if (coreType == FUSED_DEEP_MOE_PROFILE_CORE_TYPE_AIC && coreIdx == 0) {
                header->droppedLaunches += 1;
            }
            enabled = false;
            return;
        }
        uint64_t requiredBytes = sizeof(FusedDeepMoeProfileHeader) + (static_cast<uint64_t>(launchId) + 1ULL) *
                                                                         static_cast<uint64_t>(recordsPerLaunch) *
                                                                         sizeof(FusedDeepMoeProfileRecord);
        if (requiredBytes > profileBufferBytes) {
            if (coreType == FUSED_DEEP_MOE_PROFILE_CORE_TYPE_AIC && coreIdx == 0) {
                header->droppedLaunches += 1;
            }
            enabled = false;
            return;
        }
        uint64_t launchOffset = sizeof(FusedDeepMoeProfileHeader) + static_cast<uint64_t>(launchId) *
                                                                        static_cast<uint64_t>(recordsPerLaunch) *
                                                                        sizeof(FusedDeepMoeProfileRecord);
        records = reinterpret_cast<__gm__ FusedDeepMoeProfileRecord *>(base + launchOffset);
    }

    __aicore__ inline uint64_t Now() const
    {
        return static_cast<uint64_t>(AscendC::GetSystemCycle());
    }

    __aicore__ inline void Record(FusedDeepMoeProfileStage stage, uint64_t startCycle, uint64_t endCycle) const
    {
        if (!enabled) {
            return;
        }
        uint32_t stageId = static_cast<uint32_t>(stage);
        if (stageId >= stageCount) {
            return;
        }
        uint32_t logicalCoreLinear = GetFusedDeepMoeProfileLogicalCoreLinear(coreType, coreIdx);
        if (logicalCoreLinear == UINT32_MAX ||
            logicalCoreLinear >= FUSED_DEEP_MOE_PROFILE_LOGICAL_CORE_COUNT_CAPACITY) {
            return;
        }
        uint32_t slot = logicalCoreLinear * stageCount + stageId;
        if (slot >= recordsPerLaunch) {
            return;
        }
        records[slot].coreType = coreType;
        records[slot].coreIdx = coreIdx;
        records[slot].stageId = stageId;
        records[slot].launchId = launchId;
        records[slot].startCycle = startCycle;
        records[slot].endCycle = endCycle;
        records[slot].reserved0 = 0;
        records[slot].reserved1 = 0;
    }
};

}  // namespace Cam

#endif  // FUSED_DEEP_MOE_A5_PROFILE_H
