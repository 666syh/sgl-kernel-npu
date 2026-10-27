#ifndef CAM_MOE_LAYERED_HOST_VALIDATION_H
#define CAM_MOE_LAYERED_HOST_VALIDATION_H

#include <cstdint>
#include <limits>

#include "../op_kernel/cam_moe_layered_plan.h"

namespace CamMoeLayeredHost {

inline bool MulFits(uint64_t a, uint64_t b, uint64_t &result)
{
    if (a != 0U && b > std::numeric_limits<uint64_t>::max() / a) return false;
    result = a * b;
    return true;
}

inline bool AddFits(uint64_t a, uint64_t b, uint64_t &result)
{
    if (a > std::numeric_limits<uint64_t>::max() - b) return false;
    result = a + b;
    return true;
}

inline bool Align32Fits(uint64_t value, uint64_t &result)
{
    uint64_t padded = 0U;
    if (!AddFits(value, 31U, padded)) return false;
    result = (padded / 32U) * 32U;
    return true;
}

inline bool AlignCacheLineFits(uint64_t value, uint64_t &result)
{
    uint64_t padded = 0U;
    constexpr uint64_t alignment = CamMoeLayered::CACHE_LINE_ISOLATION_BYTES;
    if (!AddFits(value, alignment - 1U, padded)) return false;
    result = (padded / alignment) * alignment;
    return true;
}

inline bool AlignCombineFlagFits(uint64_t value, uint64_t &result)
{
    uint64_t padded = 0U;
    constexpr uint64_t stride = CamMoeLayered::COMBINE_FLAG_STRIDE;
    if (!AddFits(value, stride - 1U, padded)) return false;
    result = (padded / stride) * stride;
    return true;
}

inline bool ValidWindowInputArithmetic(uint64_t expertNum, uint64_t world, uint64_t ranks, uint64_t servers,
                                       uint64_t rounds, uint64_t slots, uint64_t topk, uint64_t recordStride,
                                       uint64_t partialStride)
{
    if (world == 0U || ranks == 0U || servers == 0U || rounds == 0U || slots == 0U || recordStride == 0U ||
        partialStride == 0U || expertNum == 0U || expertNum % world != 0U || expertNum % servers != 0U)
        return false;

    uint64_t topology = 0U, expertsPerServer = expertNum / servers;
    if (!MulFits(servers, ranks, topology) || topology != world || expertsPerServer % ranks != 0U) return false;
    const uint64_t localExperts = expertsPerServer / ranks;

    uint64_t countRow = 0U, countPayload = 0U, basePayload = 0U;
    uint64_t controlStride = 0U, baseAckStride = 0U, laneCount = 0U;
    if (!MulFits(expertsPerServer, sizeof(int32_t), countRow) || !AddFits(sizeof(uint32_t), countRow, countRow) ||
        !MulFits(rounds, countRow, countPayload) ||
        !AddFits(sizeof(CamMoeLayered::CountControlMeta), countPayload, countPayload) ||
        !AlignCacheLineFits(countPayload, controlStride) || !MulFits(rounds, expertsPerServer, basePayload) ||
        !MulFits(basePayload, sizeof(int32_t), basePayload) ||
        !AddFits(sizeof(CamMoeLayered::BaseAckControlMeta), basePayload, basePayload) ||
        !AlignCacheLineFits(basePayload, baseAckStride) || !MulFits(servers, world, laneCount) ||
        !MulFits(controlStride, laneCount, countPayload) || !MulFits(baseAckStride, laneCount, basePayload))
        return false;

    uint64_t proxyStride = 0U, proxyStageBytes = 0U;
    uint64_t baseAckSliceStride = 0U, baseAckStageProxyStride = 0U, baseAckStageBytes = 0U;
    uint64_t baseAckFanoutOwnerStride = 0U, baseAckFanoutBytes = 0U;
    uint64_t dispatchRecordArea = 0U, dispatchBatchStride = 0U, dispatchBytes = 0U;
    uint64_t dispatchPacketMetaBytes = 0U, dispatchTokenOffset = 0U, dispatchMemberOffset = 0U;
    uint64_t fanoutMemberOffset = 0U, fanoutBatchBytes = 0U;
    uint64_t dispatchMemberMetaBytes = 0U;
    uint64_t dispatchFanoutLanes = 0U;
    uint64_t dispatchFanoutBytes = 0U, pipelineFlagBytes = 0U;
    uint64_t partialBytes = 0U, returnPartialBytes = 0U;
    uint64_t stateLaneStride = 0U, stateBytes = 0U;
    if (!MulFits(servers, rounds, proxyStride) || !MulFits(proxyStride, localExperts, proxyStride) ||
        !MulFits(proxyStride, sizeof(int32_t), proxyStride) || !AlignCacheLineFits(proxyStride, proxyStride) ||
        !MulFits(proxyStride, ranks, proxyStageBytes) || !MulFits(rounds, localExperts, baseAckSliceStride) ||
        !MulFits(baseAckSliceStride, sizeof(int32_t), baseAckSliceStride) ||
        !AlignCacheLineFits(baseAckSliceStride, baseAckSliceStride) ||
        !MulFits(baseAckSliceStride, servers, baseAckStageProxyStride) ||
        !AddFits(baseAckStageProxyStride, CamMoeLayered::CACHE_LINE_ISOLATION_BYTES, baseAckStageProxyStride) ||
        !AlignCacheLineFits(baseAckStageProxyStride, baseAckStageProxyStride) ||
        !MulFits(baseAckStageProxyStride, ranks, baseAckStageBytes) ||
        !MulFits(baseAckSliceStride, servers, baseAckFanoutOwnerStride) ||
        !AddFits(baseAckFanoutOwnerStride, CamMoeLayered::CACHE_LINE_ISOLATION_BYTES, baseAckFanoutOwnerStride) ||
        !AlignCacheLineFits(baseAckFanoutOwnerStride, baseAckFanoutOwnerStride) ||
        !MulFits(baseAckFanoutOwnerStride, ranks, baseAckFanoutBytes) ||
        !MulFits(slots, sizeof(CamMoeLayered::DispatchPacketMeta), dispatchPacketMetaBytes) ||
        !AddFits(sizeof(CamMoeLayered::DispatchBatchMeta), dispatchPacketMetaBytes, dispatchTokenOffset) ||
        !Align32Fits(dispatchTokenOffset, dispatchTokenOffset) || !MulFits(slots, recordStride, dispatchRecordArea) ||
        !AddFits(dispatchTokenOffset, dispatchRecordArea, dispatchMemberOffset) ||
        !Align32Fits(dispatchMemberOffset, dispatchMemberOffset) || !MulFits(slots, topk, dispatchMemberMetaBytes) ||
        !MulFits(dispatchMemberMetaBytes, sizeof(CamMoeLayered::DispatchMemberMeta), dispatchMemberMetaBytes) ||
        !AddFits(dispatchMemberOffset, dispatchMemberMetaBytes, dispatchBatchStride) ||
        !Align32Fits(dispatchBatchStride, dispatchBatchStride) || !MulFits(servers, 2U, dispatchBytes) ||
        !MulFits(dispatchBytes, dispatchBatchStride, dispatchBytes) || !MulFits(servers, ranks, dispatchFanoutLanes) ||
        !AddFits(sizeof(CamMoeLayered::DispatchBatchMeta), dispatchPacketMetaBytes, fanoutMemberOffset) ||
        !Align32Fits(fanoutMemberOffset, fanoutMemberOffset) ||
        !AddFits(fanoutMemberOffset, dispatchMemberMetaBytes, fanoutBatchBytes) ||
        !Align32Fits(fanoutBatchBytes, fanoutBatchBytes) ||
        !MulFits(fanoutBatchBytes, dispatchFanoutLanes, dispatchFanoutBytes) ||
        !AddFits(CamMoeLayered::DispatchMaxChunks(static_cast<uint32_t>(slots), static_cast<uint32_t>(recordStride)),
                 2U, pipelineFlagBytes) ||
        !MulFits(pipelineFlagBytes, dispatchFanoutLanes, pipelineFlagBytes) ||
        !MulFits(pipelineFlagBytes, sizeof(uint64_t), pipelineFlagBytes) || !MulFits(ranks, ranks, partialBytes) ||
        !MulFits(partialBytes, servers, partialBytes) || !MulFits(partialBytes, servers, partialBytes) ||
        !MulFits(partialBytes, slots, partialBytes) || !MulFits(partialBytes, partialStride, partialBytes) ||
        !MulFits(ranks, servers, returnPartialBytes) || !MulFits(returnPartialBytes, servers, returnPartialBytes) ||
        !MulFits(returnPartialBytes, slots, returnPartialBytes) ||
        !MulFits(returnPartialBytes, partialStride, returnPartialBytes) || !MulFits(ranks, ranks, stateLaneStride) ||
        !AddFits(
            stateLaneStride,
            16ULL + CamMoeLayered::DispatchMaxChunks(static_cast<uint32_t>(slots), static_cast<uint32_t>(recordStride)),
            stateLaneStride) ||
        !MulFits(servers, servers, stateBytes) || !MulFits(stateBytes, rounds, stateBytes) ||
        !MulFits(stateBytes, stateLaneStride, stateBytes) || !MulFits(stateBytes, sizeof(uint64_t), stateBytes))
        return false;

    return true;
}

inline bool ValidWindowLayout(const CamMoeLayered::WindowLayout &w, uint64_t world, uint64_t servers, uint64_t ranks,
                              uint64_t rounds)
{
    uint64_t laneCount = 0U, controlBytes = 0U, baseAckBytes = 0U;
    uint64_t controlFlagBytes = 0U, baseAckFlagBytes = 0U;
    if (!MulFits(servers, world, laneCount) || !MulFits(w.controlStride, laneCount, controlBytes) ||
        !MulFits(w.baseAckStride, laneCount, baseAckBytes) ||
        !MulFits(laneCount, CamMoeLayered::CONTROL_FLAG_STRIDE, controlFlagBytes) ||
        !MulFits(laneCount, CamMoeLayered::CONTROL_FLAG_STRIDE, baseAckFlagBytes))
        return false;

    uint64_t proxyBytes = 0U, proxyStageBytes = 0U, proxyFlags = 0U, stateBytes = 0U;
    uint64_t baseAckStageBytes = 0U, baseAckFanoutBytes = 0U, baseAckFanoutFlags = 0U;
    uint64_t baseAckStageProxyStride = 0U, baseAckFanoutOwnerStride = 0U;
    uint64_t baseAckOwnerRowsOffset = 0U, baseAckOwnerRowsEnd = 0U;
    uint64_t baseAckEnd = 0U, controlFlagEnd = 0U, baseAckFlagEnd = 0U;
    uint64_t proxyStageEnd = 0U, proxyAggregateEnd = 0U, proxyStageFlagEnd = 0U;
    uint64_t baseAckStageEnd = 0U, baseAckFanoutEnd = 0U, baseAckFanoutFlagEnd = 0U;
    uint64_t dispatchPayloadEnd = 0U, dispatchFanoutEnd = 0U;
    uint64_t pipelineFlagEnd = 0U;
    uint64_t expectedDispatchBytes = 0U, expectedDispatchArenaBytes = 0U, expectedFanoutBytes = 0U;
    uint64_t expectedPipelineFlagBytes = 0U;
    uint64_t partialEnd = 0U, returnPartialEnd = 0U, stateEnd = 0U;
    uint64_t serverRoundLanes = 0U, stateStrideBytes = 0U;
    if (!MulFits(w.proxyAggregateStride, ranks, proxyBytes) ||
        !MulFits(w.proxyAggregateStride, ranks, proxyStageBytes) || !MulFits(ranks, sizeof(uint64_t), proxyFlags) ||
        !MulFits(w.baseAckStageProxyStride, ranks, baseAckStageBytes) ||
        !MulFits(w.baseAckFanoutOwnerStride, ranks, baseAckFanoutBytes) ||
        !MulFits(ranks, sizeof(uint64_t), baseAckFanoutFlags) ||
        !MulFits(w.baseAckStageSliceStride, servers, baseAckOwnerRowsOffset) ||
        !AddFits(baseAckOwnerRowsOffset, CamMoeLayered::CACHE_LINE_ISOLATION_BYTES, baseAckOwnerRowsEnd) ||
        !AlignCacheLineFits(baseAckOwnerRowsEnd, baseAckStageProxyStride) ||
        !MulFits(w.baseAckStageSliceStride, servers, baseAckFanoutOwnerStride) ||
        !AddFits(baseAckFanoutOwnerStride, CamMoeLayered::CACHE_LINE_ISOLATION_BYTES, baseAckFanoutOwnerStride) ||
        !AlignCacheLineFits(baseAckFanoutOwnerStride, baseAckFanoutOwnerStride) ||
        !AddFits(w.proxyAggregateStageOffset, proxyStageBytes, proxyStageEnd) ||
        !AddFits(w.proxyAggregateFlagOffset, proxyFlags, proxyStageFlagEnd) ||
        !AddFits(w.baseAckStageOffset, baseAckStageBytes, baseAckStageEnd) ||
        !AddFits(w.baseAckFanoutOffset, baseAckFanoutBytes, baseAckFanoutEnd) ||
        !AddFits(w.baseAckFanoutFlagOffset, baseAckFanoutFlags, baseAckFanoutFlagEnd) ||
        !MulFits(servers, servers, serverRoundLanes) || !MulFits(serverRoundLanes, rounds, serverRoundLanes) ||
        !MulFits(servers, w.dispatchBatchStride, expectedDispatchBytes) ||
        !MulFits(expectedDispatchBytes, 2U, expectedDispatchArenaBytes) ||
        !MulFits(servers, ranks, expectedFanoutBytes) ||
        !MulFits(expectedFanoutBytes, w.dispatchFanoutBatchStride, expectedFanoutBytes) ||
        !MulFits(servers, ranks, expectedPipelineFlagBytes) ||
        !MulFits(expectedPipelineFlagBytes, static_cast<uint64_t>(w.dispatchMaxChunks) + 2U,
                 expectedPipelineFlagBytes) ||
        !MulFits(expectedPipelineFlagBytes, sizeof(uint64_t), expectedPipelineFlagBytes) ||
        !MulFits(w.stateLaneStride, sizeof(uint64_t), stateStrideBytes) ||
        !MulFits(serverRoundLanes, stateStrideBytes, stateBytes) ||
        !AddFits(w.baseAckOffset, w.baseAckBytes, baseAckEnd) ||
        !AddFits(w.controlFlagOffset, controlFlagBytes, controlFlagEnd) ||
        !AddFits(w.baseAckFlagOffset, baseAckFlagBytes, baseAckFlagEnd) ||
        !AddFits(w.proxyAggregateOffset, proxyBytes, proxyAggregateEnd) ||
        !AddFits(w.dispatchPayloadOffset, w.dispatchPayloadCapacity, dispatchPayloadEnd) ||
        !AddFits(w.dispatchFanoutOffset, w.dispatchFanoutBytes, dispatchFanoutEnd) ||
        !AddFits(w.dispatchPipelineFlagOffset, w.dispatchPipelineFlagBytes, pipelineFlagEnd) ||
        !AddFits(w.partialOffset, w.localPartialBytes, partialEnd) ||
        !AddFits(w.returnPartialOffset, w.returnPartialBytes, returnPartialEnd) ||
        !AddFits(w.stateOffset, stateBytes, stateEnd))
        return false;

    return w.controlStride != 0U && w.baseAckStride != 0U && w.dispatchBatchStride != 0U &&
           w.dispatchBatchStride >= sizeof(CamMoeLayered::DispatchBatchMeta) &&
           w.dispatchBatchStride <= std::numeric_limits<uint32_t>::max() && w.requiredBytes != 0U &&
           (w.controlStride % CamMoeLayered::CACHE_LINE_ISOLATION_BYTES) == 0U &&
           (w.baseAckStride % CamMoeLayered::CACHE_LINE_ISOLATION_BYTES) == 0U &&
           (w.controlBytes % CamMoeLayered::CACHE_LINE_ISOLATION_BYTES) == 0U &&
           (w.baseAckOffset % CamMoeLayered::CACHE_LINE_ISOLATION_BYTES) == 0U &&
           (w.baseAckBytes % CamMoeLayered::CACHE_LINE_ISOLATION_BYTES) == 0U &&
           (w.controlFlagOffset % CamMoeLayered::CACHE_LINE_ISOLATION_BYTES) == 0U &&
           (w.baseAckFlagOffset % CamMoeLayered::CACHE_LINE_ISOLATION_BYTES) == 0U &&
           (w.proxyAggregateStageOffset % CamMoeLayered::CACHE_LINE_ISOLATION_BYTES) == 0U &&
           (w.proxyAggregateOffset % 32U) == 0U &&
           (w.proxyAggregateStride % CamMoeLayered::CACHE_LINE_ISOLATION_BYTES) == 0U &&
           (w.proxyAggregateFlagOffset % 32U) == 0U &&
           (w.baseAckStageOffset % CamMoeLayered::CACHE_LINE_ISOLATION_BYTES) == 0U &&
           (w.baseAckStageSliceStride % CamMoeLayered::CACHE_LINE_ISOLATION_BYTES) == 0U &&
           (w.baseAckStageProxyStride % CamMoeLayered::CACHE_LINE_ISOLATION_BYTES) == 0U &&
           (w.baseAckFanoutOffset % CamMoeLayered::CACHE_LINE_ISOLATION_BYTES) == 0U &&
           (w.baseAckFanoutOwnerStride % CamMoeLayered::CACHE_LINE_ISOLATION_BYTES) == 0U &&
           (w.baseAckFanoutFlagOffset % 32U) == 0U && (w.dispatchPayloadOffset % 32U) == 0U &&
           (w.dispatchBatchStride % 32U) == 0U && (w.dispatchPayloadCapacity % 32U) == 0U &&
           (w.dispatchFanoutBatchStride % 32U) == 0U && (w.dispatchFanoutBytes % 32U) == 0U &&
           (w.dispatchFanoutOffset % 32U) == 0U && (w.dispatchPipelineFlagOffset % 32U) == 0U &&
           (w.partialOffset % 32U) == 0U && (w.localPartialBytes % 32U) == 0U && (w.returnPartialOffset % 32U) == 0U &&
           (w.returnPartialBytes % 32U) == 0U && (w.stateOffset % 32U) == 0U && (w.requiredBytes % 32U) == 0U &&
           w.dispatchPayloadCapacity == expectedDispatchArenaBytes && w.dispatchFanoutBytes == expectedFanoutBytes &&
           w.controlBytes == controlBytes && w.dispatchPipelineFlagBytes == expectedPipelineFlagBytes &&
           w.stateLaneStride == 16U + ranks * ranks + w.dispatchMaxChunks && w.baseAckOffset >= w.controlBytes &&
           w.baseAckBytes == baseAckBytes && w.controlFlagOffset >= baseAckEnd &&
           w.baseAckFlagOffset >= controlFlagEnd && w.proxyAggregateStageOffset >= baseAckFlagEnd &&
           w.proxyAggregateStageBytes == proxyStageBytes && w.proxyAggregateOffset >= proxyStageEnd &&
           w.proxyAggregateBytes == proxyBytes && w.proxyAggregateFlagOffset >= proxyAggregateEnd &&
           w.baseAckStageOffset >= proxyStageFlagEnd && w.baseAckStageOwnerRowsOffset == baseAckOwnerRowsOffset &&
           w.baseAckStageProxyStride == baseAckStageProxyStride && w.baseAckStageBytes == baseAckStageBytes &&
           w.baseAckFanoutOffset >= baseAckStageEnd && w.baseAckFanoutOwnerStride == baseAckFanoutOwnerStride &&
           w.baseAckFanoutBytes == baseAckFanoutBytes && w.baseAckFanoutFlagOffset >= baseAckFanoutEnd &&
           w.dispatchPayloadOffset >= baseAckFanoutFlagEnd &&
#ifdef __DAV_C310__
           w.dispatchPayloadOffset >= Moe::A5WindowLayout::kDataOffset &&
#endif
           w.dispatchFanoutOffset >= dispatchPayloadEnd && w.dispatchPipelineFlagOffset >= dispatchFanoutEnd &&
           w.partialOffset >= pipelineFlagEnd && w.returnPartialOffset >= partialEnd &&
           w.stateOffset >= returnPartialEnd && w.requiredBytes >= stateEnd;
}

inline bool ValidWindowCapacity(const CamMoeLayered::WindowLayout &w, uint64_t configuredBytes)
{
    return configuredBytes != 0U && w.requiredBytes != 0U && w.requiredBytes <= configuredBytes;
}

inline bool ValidCombineWindowArithmetic(uint64_t baseOffset, uint64_t serverNum, uint64_t ranksPerServer,
                                         uint64_t localExperts, uint64_t ownerCapacity, uint64_t maxBs, uint64_t hidden,
                                         uint64_t elementBytes, uint64_t &requiredBytes)
{
    uint64_t ownerStride = 0U, ownerBytes = 0U, expertStride = 0U, expertBytes = 0U;
    uint64_t partialStride = 0U, partialBytes = 0U, cursor = baseOffset;
    uint64_t flags = 0U;
    if (!MulFits(hidden, elementBytes, ownerStride) || !Align32Fits(ownerStride, ownerStride) ||
        !MulFits(ownerCapacity, ownerStride, ownerBytes) || !AddFits(cursor, ownerBytes, cursor) ||
        !Align32Fits(cursor, cursor) || !AddFits(localExperts, 1U, expertStride) ||
        !MulFits(expertStride, sizeof(uint32_t), expertStride) || !Align32Fits(expertStride, expertStride) ||
        expertStride > std::numeric_limits<uint32_t>::max() || !MulFits(ranksPerServer, expertStride, expertBytes) ||
        !AddFits(cursor, expertBytes, cursor) || !Align32Fits(cursor, cursor) ||
        !MulFits(hidden, sizeof(float), partialStride) || !Align32Fits(partialStride, partialStride) ||
        !MulFits(serverNum, maxBs, partialBytes) || !MulFits(partialBytes, partialStride, partialBytes) ||
        partialBytes > std::numeric_limits<uint32_t>::max() || !AddFits(cursor, partialBytes, cursor) ||
        !Align32Fits(cursor, cursor) || !AddFits(cursor, partialBytes, cursor) || !Align32Fits(cursor, cursor)) {
        return false;
    }
    const uint64_t flagCounts[8] = {ranksPerServer, ranksPerServer, serverNum, serverNum,
                                    serverNum,      serverNum,      serverNum, serverNum};
    for (uint64_t count : flagCounts) {
        if (!AlignCombineFlagFits(cursor, cursor) || !MulFits(count, CamMoeLayered::COMBINE_FLAG_STRIDE, flags) ||
            !AddFits(cursor, flags, cursor)) {
            return false;
        }
    }
    if (!AlignCombineFlagFits(cursor, cursor)) return false;
    requiredBytes = cursor;
    return requiredBytes <= std::numeric_limits<uint32_t>::max();
}

}  // namespace CamMoeLayeredHost

#endif
