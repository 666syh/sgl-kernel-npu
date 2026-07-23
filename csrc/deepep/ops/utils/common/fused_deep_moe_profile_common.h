/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */
#ifndef OPS_UTILS_COMMON_FUSED_DEEP_MOE_PROFILE_COMMON_H
#define OPS_UTILS_COMMON_FUSED_DEEP_MOE_PROFILE_COMMON_H

#include <cstdint>

namespace Cam {

constexpr uint64_t FUSED_DEEP_MOE_PROFILE_MAGIC = 0x46444D5035413031ULL;  // FDMP5A01
constexpr uint64_t FUSED_DEEP_MOE_PROFILE_VERSION = 1;
constexpr uint64_t FUSED_DEEP_MOE_PROFILE_CYCLE_TO_US = 50;

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
    uint64_t coreCount;
    uint64_t stageCount;
    uint64_t recordCount;
    uint64_t cycleToUs;
    uint64_t reserved0;
    uint64_t reserved1;
};

struct FusedDeepMoeProfileRecord {
    uint64_t coreType;
    uint64_t coreIdx;
    uint64_t stageId;
    uint64_t startCycle;
    uint64_t endCycle;
    uint64_t reserved0;
    uint64_t reserved1;
    uint64_t reserved2;
};

static_assert(sizeof(FusedDeepMoeProfileHeader) == 64, "Unexpected fused profile header size");
static_assert(sizeof(FusedDeepMoeProfileRecord) == 64, "Unexpected fused profile record size");

}  // namespace Cam

#endif  // OPS_UTILS_COMMON_FUSED_DEEP_MOE_PROFILE_COMMON_H
