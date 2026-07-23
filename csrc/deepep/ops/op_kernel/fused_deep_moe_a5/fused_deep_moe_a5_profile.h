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
    __gm__ FusedDeepMoeProfileHeader *header{nullptr};
    __gm__ FusedDeepMoeProfileRecord *records{nullptr};
    bool enabled{false};
    uint32_t coreIdx{0};
    uint32_t coreType{0};
    uint32_t coreCount{0};
    uint32_t stageCount{0};

    __aicore__ inline void Init(GM_ADDR profileGM, bool enable, uint32_t coreCount_, uint32_t coreType_)
    {
        enabled = enable;
        if (!enabled) {
            return;
        }
        coreIdx = AscendC::GetBlockIdx();
        coreType = coreType_;
        coreCount = coreCount_;
        stageCount = static_cast<uint32_t>(FusedDeepMoeProfileStage::Count);
        auto *base = reinterpret_cast<__gm__ uint8_t *>(profileGM);
        header = reinterpret_cast<__gm__ FusedDeepMoeProfileHeader *>(base);
        records = reinterpret_cast<__gm__ FusedDeepMoeProfileRecord *>(base + sizeof(FusedDeepMoeProfileHeader));
        if (coreIdx == 0) {
            header->magic = FUSED_DEEP_MOE_PROFILE_MAGIC;
            header->version = FUSED_DEEP_MOE_PROFILE_VERSION;
            header->coreCount = coreCount;
            header->stageCount = stageCount;
            header->recordCount = coreCount * stageCount;
            header->cycleToUs = FUSED_DEEP_MOE_PROFILE_CYCLE_TO_US;
            header->reserved0 = 0;
            header->reserved1 = 0;
        }
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
        uint32_t slot = coreIdx * stageCount + stageId;
        records[slot].coreType = coreType;
        records[slot].coreIdx = coreIdx;
        records[slot].stageId = stageId;
        records[slot].startCycle = startCycle;
        records[slot].endCycle = endCycle;
        records[slot].reserved0 = 0;
        records[slot].reserved1 = 0;
        records[slot].reserved2 = 0;
    }
};

}  // namespace Cam

#endif  // FUSED_DEEP_MOE_A5_PROFILE_H
