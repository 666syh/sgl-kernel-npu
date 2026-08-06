/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */
#ifndef DEEPEP_PROFILING_ADAPTERS_FUSED_DEEP_MOE_A5_PROFILE_PAYLOAD_H
#define DEEPEP_PROFILING_ADAPTERS_FUSED_DEEP_MOE_A5_PROFILE_PAYLOAD_H

#include "profiling/common/profile_protocol_common.h"

namespace Cam {

struct DispatchSendPrivatePayloadV1 {
    uint64_t header;
    uint64_t validTokenCount;
    uint64_t perTokenCommBytes;
};

static_assert(sizeof(DispatchSendPrivatePayloadV1) <= sizeof(ProfilePrivatePayloadRaw),
              "DispatchSend payload must fit into raw payload slots");

inline constexpr DispatchSendPrivatePayloadV1 MakeDispatchSendPrivatePayloadV1(uint8_t validTag, uint8_t formatId,
                                                                               uint64_t validTokenCount,
                                                                               uint64_t perTokenCommBytes)
{
    return DispatchSendPrivatePayloadV1{PackProfilePrivate0(validTag, formatId), validTokenCount, perTokenCommBytes};
}

inline constexpr ProfilePrivatePayloadRaw ToProfilePrivatePayloadRaw(const DispatchSendPrivatePayloadV1 &payload)
{
    return MakeProfilePrivatePayloadRaw(payload.header, payload.validTokenCount, payload.perTokenCommBytes);
}

inline constexpr DispatchSendPrivatePayloadV1 AsDispatchSendPrivatePayloadV1(const ProfilePrivatePayloadRaw &payload)
{
    return DispatchSendPrivatePayloadV1{payload.private0, payload.private1, payload.private2};
}

inline constexpr DispatchSendPrivatePayloadV1 AsDispatchSendPrivatePayloadV1(const ProfileRecord &record)
{
    return AsDispatchSendPrivatePayloadV1(record.payload);
}

}  // namespace Cam

#endif  // DEEPEP_PROFILING_ADAPTERS_FUSED_DEEP_MOE_A5_PROFILE_PAYLOAD_H
