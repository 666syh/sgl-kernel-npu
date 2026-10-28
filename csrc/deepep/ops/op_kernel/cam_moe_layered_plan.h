#ifndef CAM_MOE_LAYERED_PLAN_H
#define CAM_MOE_LAYERED_PLAN_H

#include <cstddef>
#include <cstdint>
#include "window_layout.h"

namespace CamMoeLayered {

#ifdef __CCE_AICORE__
#define CAM_MOE_LAYERED_INLINE __aicore__ inline
#else
#define CAM_MOE_LAYERED_INLINE inline
#endif

constexpr uint32_t PLAN_MAGIC = 0x4C4D4F45U;  // "LMOE"
constexpr uint32_t PLAN_VERSION = 25U;
constexpr uint32_t DISPATCH_CHUNK_TARGET_BYTES = 128U * 1024U;
constexpr int32_t INVALID_SLOT = -1;
constexpr uint32_t ALIGN_BYTES = 32U;
// Keep independently published records on distinct DCCI lines. 128B is a
// conservative isolation granularity for the supported Ascend devices.
constexpr uint32_t CACHE_LINE_ISOLATION_BYTES = 128U;
constexpr uint32_t CONTROL_FLAG_STRIDE = CACHE_LINE_ISOLATION_BYTES;
constexpr uint32_t ASSIST_INFO_STRIDE = 9U;

enum class Phase : uint32_t {
    CONTROL_COUNT_READY = 1,
    CONTROL_BASE_READY = 2,
    DISPATCH_DATA_READY = 4,
    SERVER_FANOUT_DONE = 5,
    PARTIAL_READY = 6,
    BUFFER_REUSABLE = 7,
    DISPATCH_FANOUT_READY = 9,
    DISPATCH_METADATA_READY = 11,
    DISPATCH_PROXY_REUSABLE = 12,
    COMBINE_OWNER_READY = 16,
    COMBINE_OWNER_CONSUMED = 17,
    COMBINE_SOURCE_READY = 18,
    COMBINE_SOURCE_CONSUMED = 19,
    COMBINE_GLOBAL_READY = 20,
    DISPATCH_CHUNK_READY = 21,
    DISPATCH_REUSABLE = 22,
    DISPATCH_REVERSE_DONE = 23,
    DISPATCH_PIPELINE_ERROR = 24,
    ERROR_OVERFLOW = 8,
    ERROR_PROTOCOL = 10,
};

CAM_MOE_LAYERED_INLINE constexpr uint64_t ComposeFlag(uint64_t epoch, Phase phase)
{
    return (epoch << 8U) | static_cast<uint64_t>(phase);
}

CAM_MOE_LAYERED_INLINE constexpr uint32_t AlignUp(uint32_t value, uint32_t align = ALIGN_BYTES)
{
    return (value + align - 1U) / align * align;
}

// All plan offsets are int32 elements. The plan is persistent GM metadata;
// control and packet buffers in HCCL windows are deliberately not reused for it.
struct PlanHeader {
    // 固定魔数，用于判断 layered_plan 是否由本协议生成。
    uint32_t magic;

    // Plan ABI 版本号。Dispatch/Combine 必须拒绝不支持的版本。
    uint32_t version;

    // epoch 低 32 位。与 epochHi 组合成当前执行批次的 64-bit generation。
    uint32_t epochLo;

    // epoch 高 32 位。用于区分不同执行批次的 flag 和 record，避免复用冲突。
    uint32_t epochHi;

    // 当前 rank 负责处理的 token 数量，即输入 topk_idx 的 BS。
    uint32_t bs;

    // Uniform per-rank capacity and this rank's effective round count.
    uint32_t maxBs;
    uint32_t validRounds;

    // 每个 token 的 hidden size。
    uint32_t hiddenSize;

    // 每个 token 的 top-k 数量，即输入 topk_idx 的 K。
    uint32_t topk;

    // 全局 MoE expert 数量。
    uint32_t expertNum;

    // EP 通信域中的总 rank 数。
    uint32_t worldSize;

    // server 数量，满足 worldSize = serverNum * ranksPerServer。
    uint32_t serverNum;

    // 每个 server 包含的 rank 数。
    uint32_t ranksPerServer;

    // 当前 rank 在 EP 通信域中的全局 rank id。
    uint32_t rankId;

    // 当前 rank 在所属 server 内的 rank id。
    // 用于确定跨 server RDMA 的同号 landing rank。
    uint32_t rankInServer;

    // 统一通信容量对应的物理搬运轮数，即 ceil(maxBs / slotsPerServer)。
    uint32_t physicalRounds;

    // 每一轮可容纳的 packet 槽位数。
    // round 内的 token 使用 slot 标识，不允许超过该容量。
    uint32_t slotsPerServer;

    // DispatchRecord 的固定字节跨度。
    // 用于计算 dispatch batch 中每个 dense packet 的地址。
    uint32_t recordStrideBytes;

    // packetCount section 的 int32 元素偏移。
    // 逻辑布局为 [physicalRounds, serverNum]。
    uint32_t packetCountOffset;

    // packetSlot section 的 int32 元素偏移。
    // 物理布局为 [maxBs, serverNum]，无效映射为 INVALID_SLOT。
    uint32_t packetSlotOffset;

    // expertCount section 的 int32 元素偏移。
    // 紧凑布局为 [round][targetServer][expertInServer]。
    uint32_t expertCountOffset;

    // localExpertOffset section 的 int32 元素偏移。
    // 保存当前 rank 每个 local expert 在 compact recv_x 中的起始行。
    uint32_t localExpertOffset;

    // localExpertCount section 的 int32 元素偏移。
    // 保存当前 rank 每个 local expert 的实际接收行数。
    uint32_t localExpertCountOffset;

    // source packet descriptor section 的 int32 元素偏移。
    uint32_t packetDescOffset;

    // memberDesc section 的 int32 元素偏移。
    uint32_t memberDescOffset;

    // overflow 标志的 int32 元素偏移。
    // 非零表示 packet 容量、expert 路由或其他规划约束溢出。
    uint32_t overflowOffset;

    // error 状态的 int32 元素偏移。
    // 保存 ERROR_OVERFLOW、ERROR_PROTOCOL 等错误码。
    uint32_t errorOffset;

    // layered_plan 的总元素数量，单位为 int32。
    // Host 分配 plan tensor 时使用该值进行容量校验。
    uint32_t totalElements;

    // 每个 [round][targetServer] batch 的 member 总数。
    uint32_t dispatchMemberCountOffset;

    // Notify 的 BaseAck 回程写入的全局最大精确 compact 行数。
    // Combine 以此计算所有 rank 一致的轮数。
    uint32_t globalCombineRows;

    // 预留字段，保持 ABI 扩展空间并确保 header 对齐。
    uint32_t reserved[2];
};

struct SourcePacketDesc {
    int32_t sourceToken;      // 保存原始token Id
    int32_t memberBegin;      // 该token的member是发送给目标server的第几个member
    int32_t memberCount;      // 该token的member数量
    int32_t dispatchOrdinal;  // 该token是发送给目标server的第几个packet
};

struct ReversePacketDesc {
    int32_t sourceRank;
    int32_t sourceToken;
};

struct MemberDesc {
    int32_t expertId;
    int32_t expertRow;
    int32_t topkSlot;
};

// Proxy-local reverse index, retained across later dispatches and window reuse.
struct ReverseMember {
    int32_t expertId;
    int32_t expertRow;
    int32_t topkSlot;
    int32_t weightBits;
};
static_assert(sizeof(ReverseMember) == 4U * sizeof(int32_t), "ReverseMember ABI changed");

// CountControl carries the source rank's return-base and packet/expert counts.
// The fixed 32B control block keeps the following arrays aligned for RDMA.
struct CountControlMeta {
    uint32_t sourceReturnBase;
    uint32_t padding[7];
};

// BaseAck returns row bases and, in the same payload, the maximum exact
// compact-row count for the target server.  The fixed 32B header keeps the
// row-base matrix aligned while avoiding a separate Combine control exchange.
struct BaseAckControlMeta {
    uint32_t serverMaxRows;
    uint32_t ingressOffsetLo;
    uint32_t ingressOffsetHi;
    uint32_t payloadBytesLo;
    uint32_t payloadBytesHi;
    uint32_t reserved[3];
};

struct DispatchBatchMeta {
    uint32_t packetCount;
    uint32_t memberCount;
    uint32_t packetMetaOffset;
    uint32_t tokenRecordOffset;
    uint32_t memberMetaOffset;
    uint32_t reserved;
    // Used by the owner-fanout header; zero in the network ingress header.
    uint32_t ingressOffsetLo;
    uint32_t ingressOffsetHi;
};

struct DispatchPacketMeta {
    uint32_t sourceToken;
    uint32_t memberBegin;
    uint32_t memberCount;
};

struct DispatchMemberMeta {
    int32_t expertId;
    int32_t expertRow;
    int32_t topkSlot;
};

// Full source-token routing is carried once per server packet.  A member
// references its selected item with DispatchMemberMeta::topkSlot.
struct DispatchTopkRouteRecord {
    int32_t expertId;
    int32_t weightBits;
};

// Persistent per-received-row combine metadata.  Keep this layout stable:
// Dispatch writes it and Combine consumes it after the dispatch record slots
// have been reused by later rounds.
struct CombineRoute {
    int32_t epoch;
    int32_t sourceRank;
    int32_t sourceToken;
    int32_t sourceServer;
    int32_t sourceRankInServer;
    int32_t round;
    int32_t slot;
    int32_t topkSlot;
    int32_t weightBits;
};

struct PartialRecordHeader {
    uint64_t epoch;
    uint32_t sourceRank;
    uint32_t sourceToken;
    uint32_t round;
    uint32_t sourceSlot;
    uint32_t dstServer;
    uint32_t reserved;
};

static_assert(sizeof(PlanHeader) == 32U * sizeof(uint32_t), "LayeredPlan header ABI changed");
static_assert(sizeof(SourcePacketDesc) == 4U * sizeof(int32_t), "SourcePacketDesc ABI changed");
static_assert(sizeof(ReversePacketDesc) == 2U * sizeof(int32_t), "ReversePacketDesc ABI changed");
static_assert(sizeof(MemberDesc) == 3U * sizeof(int32_t), "MemberDesc ABI changed");
static_assert(sizeof(CountControlMeta) == ALIGN_BYTES, "CountControl meta ABI changed");
static_assert(sizeof(BaseAckControlMeta) == ALIGN_BYTES, "BaseAck control meta ABI changed");
static_assert(sizeof(DispatchBatchMeta) == ALIGN_BYTES, "Dispatch batch meta must be 32B");
static_assert(sizeof(DispatchPacketMeta) == 3U * sizeof(uint32_t), "Dispatch packet meta ABI changed");
static_assert(offsetof(DispatchBatchMeta, ingressOffsetLo) == 6U * sizeof(uint32_t),
              "Dispatch ingress low word ABI changed");
static_assert(offsetof(DispatchBatchMeta, ingressOffsetHi) == 7U * sizeof(uint32_t),
              "Dispatch ingress high word ABI changed");
static_assert(sizeof(DispatchMemberMeta) == 3U * sizeof(int32_t), "Dispatch member meta ABI changed");
static_assert(sizeof(DispatchTopkRouteRecord) == 2U * sizeof(int32_t), "Dispatch top-k ABI changed");
static_assert(sizeof(CombineRoute) == ASSIST_INFO_STRIDE * sizeof(int32_t), "Combine route ABI changed");

CAM_MOE_LAYERED_INLINE constexpr uint32_t HeaderElements()
{
    return sizeof(PlanHeader) / sizeof(int32_t);
}
CAM_MOE_LAYERED_INLINE constexpr uint32_t PacketElements()
{
    return sizeof(SourcePacketDesc) / sizeof(int32_t);
}
CAM_MOE_LAYERED_INLINE constexpr uint32_t ReversePacketElements()
{
    return sizeof(ReversePacketDesc) / sizeof(int32_t);
}
CAM_MOE_LAYERED_INLINE constexpr uint32_t MemberElements()
{
    return sizeof(MemberDesc) / sizeof(int32_t);
}
CAM_MOE_LAYERED_INLINE constexpr uint32_t ReverseMemberElements()
{
    return sizeof(ReverseMember) / sizeof(int32_t);
}
CAM_MOE_LAYERED_INLINE constexpr uint32_t LocalExpertNum(uint32_t expertNum, uint32_t worldSize)
{
    return worldSize == 0 ? 0 : expertNum / worldSize;
}
CAM_MOE_LAYERED_INLINE constexpr uint32_t ServerExpertNum(uint32_t expertNum, uint32_t worldSize,
                                                          uint32_t ranksPerServer)
{
    return LocalExpertNum(expertNum, worldSize) * ranksPerServer;
}
CAM_MOE_LAYERED_INLINE uint32_t DispatchChunkPackets(uint32_t maxBs, uint32_t recordStride)
{
    const uint32_t byBytes = recordStride == 0U ? 1U : DISPATCH_CHUNK_TARGET_BYTES / recordStride;
    const uint32_t packets = byBytes == 0U ? 1U : byBytes;
    return packets < maxBs ? packets : maxBs;
}
CAM_MOE_LAYERED_INLINE uint32_t DispatchMaxChunks(uint32_t maxBs, uint32_t recordStride)
{
    const uint32_t packets = DispatchChunkPackets(maxBs, recordStride);
    return packets == 0U ? 0U : (maxBs - 1U) / packets + 1U;
}
CAM_MOE_LAYERED_INLINE constexpr uint32_t DispatchTopkRouteOffset(uint32_t h, uint32_t elementBytes)
{
    return AlignUp(h * elementBytes);
}
CAM_MOE_LAYERED_INLINE constexpr uint32_t DispatchRecordStride(uint32_t h, uint32_t k, uint32_t elementBytes)
{
    return AlignUp(DispatchTopkRouteOffset(h, elementBytes) + k * sizeof(DispatchTopkRouteRecord));
}
CAM_MOE_LAYERED_INLINE constexpr uint32_t DispatchRecordRouteEnd(uint32_t h, uint32_t k, uint32_t elementBytes)
{
    return DispatchTopkRouteOffset(h, elementBytes) + k * sizeof(DispatchTopkRouteRecord);
}
static_assert(DispatchRecordRouteEnd(33U, 2U, 2U) <= DispatchRecordStride(33U, 2U, 2U),
              "16-bit token routes must fit in their record");
static_assert(DispatchRecordStride(33U, 2U, 2U) == 128U, "h=33, k=2 must reserve the aligned route tail");
CAM_MOE_LAYERED_INLINE constexpr uint32_t DispatchPacketMetaOffsetInBatch()
{
    return sizeof(DispatchBatchMeta);
}
CAM_MOE_LAYERED_INLINE constexpr uint32_t DispatchIngressOffsetLoWord()
{
    return offsetof(DispatchBatchMeta, ingressOffsetLo) / sizeof(uint32_t);
}
CAM_MOE_LAYERED_INLINE constexpr uint32_t DispatchIngressOffsetHiWord()
{
    return offsetof(DispatchBatchMeta, ingressOffsetHi) / sizeof(uint32_t);
}
CAM_MOE_LAYERED_INLINE constexpr uint32_t DispatchTokenRecordOffsetInBatch(uint32_t slots)
{
    return AlignUp(sizeof(DispatchBatchMeta) + slots * sizeof(DispatchPacketMeta));
}
CAM_MOE_LAYERED_INLINE constexpr uint32_t DispatchMemberMetaOffsetInBatch(uint32_t h, uint32_t k, uint32_t elementBytes,
                                                                          uint32_t slots)
{
    return AlignUp(DispatchTokenRecordOffsetInBatch(slots) +
                   static_cast<uint64_t>(slots) * DispatchRecordStride(h, k, elementBytes));
}
CAM_MOE_LAYERED_INLINE constexpr uint32_t DispatchIngressBatchStride(uint32_t h, uint32_t k, uint32_t elementBytes,
                                                                     uint32_t slots)
{
    return AlignUp(DispatchMemberMetaOffsetInBatch(h, k, elementBytes, slots) +
                   static_cast<uint64_t>(slots) * k * sizeof(DispatchMemberMeta));
}
CAM_MOE_LAYERED_INLINE uint32_t PartialRecordStride(uint32_t h, uint32_t elementBytes)
{
    (void)elementBytes;
    return AlignUp(sizeof(PartialRecordHeader) + h * sizeof(float));
}
// Every rank exposes one outbound batch per target server and one compact
// inbound stream containing the batches from its same-rank source proxies.
struct WindowLayout {
    uint64_t controlStride;
    uint64_t baseAckStride;
    uint64_t controlBytes;
    uint64_t baseAckOffset;
    uint64_t baseAckBytes;
    uint64_t controlFlagOffset;
    uint64_t baseAckFlagOffset;
    uint64_t proxyAggregateStageOffset;
    uint64_t proxyAggregateStageBytes;
    uint64_t proxyAggregateOffset;
    uint64_t proxyAggregateStride;
    uint64_t proxyAggregateBytes;
    uint64_t proxyAggregateFlagOffset;
    uint64_t baseAckStageOffset;
    uint64_t baseAckStageSliceStride;
    uint64_t baseAckStageOwnerRowsOffset;
    uint64_t baseAckStageProxyStride;
    uint64_t baseAckStageBytes;
    uint64_t baseAckFanoutOffset;
    uint64_t baseAckFanoutOwnerStride;
    uint64_t baseAckFanoutBytes;
    uint64_t baseAckFanoutFlagOffset;
    // Per-invocation compact payload arena. Cross-server outbound staging
    // occupies a prefix; inbound ingress follows the actual outbound bytes.
    // Server-local fanout metadata lives in the separate fixed region below.
    uint64_t dispatchPayloadOffset;
    uint64_t dispatchPayloadCapacity;
    uint64_t dispatchPacketMetaOffset;
    uint64_t dispatchTokenRecordOffset;
    uint64_t dispatchMemberMetaOffset;
    uint64_t dispatchPacketMetaStride;
    uint64_t dispatchMemberMetaStride;
    uint64_t dispatchFanoutOffset;
    uint64_t dispatchFanoutBytes;
    uint64_t dispatchFanoutBatchStride;
    uint64_t dispatchFanoutMemberMetaOffset;
    uint64_t dispatchPipelineFlagOffset;
    uint64_t dispatchPipelineFlagBytes;
    uint32_t dispatchMaxChunks;
    uint64_t partialOffset;
    uint64_t localPartialBytes;
    uint64_t returnPartialOffset;
    uint64_t returnPartialBytes;
    uint64_t dispatchBatchStride;
    uint64_t stateOffset;
    uint64_t stateLaneStride;
    uint64_t requiredBytes;
};

CAM_MOE_LAYERED_INLINE constexpr uint64_t AllRoundCountPayloadBytes(uint32_t rounds, uint32_t expertsPerServer)
{
    return sizeof(CountControlMeta) +
           static_cast<uint64_t>(rounds) * (sizeof(uint32_t) + expertsPerServer * sizeof(int32_t));
}

CAM_MOE_LAYERED_INLINE constexpr uint64_t AllRoundBasePayloadBytes(uint32_t rounds, uint32_t expertsPerServer)
{
    return sizeof(BaseAckControlMeta) + static_cast<uint64_t>(rounds) * expertsPerServer * sizeof(int32_t);
}

CAM_MOE_LAYERED_INLINE constexpr uint64_t CountControlDataOffset()
{
    return sizeof(CountControlMeta);
}

CAM_MOE_LAYERED_INLINE constexpr uint64_t BaseAckOwnerExpertOffset(uint32_t ownerInServer, uint32_t localExpert,
                                                                   uint32_t localExperts)
{
    return sizeof(BaseAckControlMeta) +
           (static_cast<uint64_t>(ownerInServer) * localExperts + localExpert) * sizeof(int32_t);
}

CAM_MOE_LAYERED_INLINE constexpr uint64_t BaseAckStageOffset(const WindowLayout &l, uint32_t proxyInServer,
                                                             uint32_t sourceServer)
{
    return l.baseAckStageOffset + proxyInServer * l.baseAckStageProxyStride + sourceServer * l.baseAckStageSliceStride;
}

CAM_MOE_LAYERED_INLINE constexpr uint64_t BaseAckStageOwnerRowsOffset(const WindowLayout &l, uint32_t proxyInServer)
{
    return l.baseAckStageOffset + proxyInServer * l.baseAckStageProxyStride + l.baseAckStageOwnerRowsOffset;
}

CAM_MOE_LAYERED_INLINE constexpr uint64_t BaseAckFanoutOwnerOffset(const WindowLayout &l, uint32_t ownerInServer)
{
    return l.baseAckFanoutOffset + ownerInServer * l.baseAckFanoutOwnerStride;
}

CAM_MOE_LAYERED_INLINE constexpr uint64_t BaseAckFanoutSliceOffset(const WindowLayout &l, uint32_t ownerInServer,
                                                                   uint32_t sourceServer)
{
    return BaseAckFanoutOwnerOffset(l, ownerInServer) + sourceServer * l.baseAckStageSliceStride;
}

CAM_MOE_LAYERED_INLINE constexpr uint64_t BaseAckFanoutOwnerRowsOffset(const WindowLayout &l, uint32_t ownerInServer)
{
    return BaseAckFanoutOwnerOffset(l, ownerInServer) + l.baseAckStageOwnerRowsOffset;
}

CAM_MOE_LAYERED_INLINE constexpr uint64_t BaseAckFanoutFlagOffset(const WindowLayout &l, uint32_t ownerInServer)
{
    return l.baseAckFanoutFlagOffset + ownerInServer * sizeof(uint64_t);
}

CAM_MOE_LAYERED_INLINE constexpr uint64_t AlignUp64(uint64_t value, uint64_t align = ALIGN_BYTES)
{
    return (value + align - 1U) / align * align;
}

CAM_MOE_LAYERED_INLINE uint64_t DispatchBatchPayloadBytes(uint32_t packetCount, uint32_t memberCount,
                                                          uint32_t recordStride)
{
    if (packetCount == 0U && memberCount == 0U) {
        return 0U;
    }
    /*
     * [
     *  DispatchBatchMeta                           // 固定头
     *  packetCount × sizeof(DispatchPacketMeta)    // packet 元数据数组
     *  packetCount × recordStride                  // token 记录
     *  memberCount × sizeof(DispatchMemberMeta) )  // member 元数据数组
     * ]
     */
    const uint64_t packetMetaOffset = sizeof(DispatchBatchMeta);
    const uint64_t tokenRecordOffset =
        AlignUp64(packetMetaOffset + static_cast<uint64_t>(packetCount) * sizeof(DispatchPacketMeta));
    const uint64_t memberMetaOffset = AlignUp64(tokenRecordOffset + static_cast<uint64_t>(packetCount) * recordStride);
    return AlignUp64(memberMetaOffset + static_cast<uint64_t>(memberCount) * sizeof(DispatchMemberMeta));
}

// All address calculations are byte-based. Source and target server are both
// encoded so each sender/receiver pair owns a distinct batch lane.
CAM_MOE_LAYERED_INLINE constexpr uint64_t ControlLaneOffset(const WindowLayout &l, uint32_t worldSize,
                                                            uint32_t sourceRank, uint32_t targetServer)
{
    return (targetServer * worldSize + sourceRank) * l.controlStride;
}
CAM_MOE_LAYERED_INLINE constexpr uint64_t BaseAckLaneOffset(const WindowLayout &l, uint32_t worldSize,
                                                            uint32_t sourceRank, uint32_t targetServer)
{
    return l.baseAckOffset + (targetServer * worldSize + sourceRank) * l.baseAckStride;
}
CAM_MOE_LAYERED_INLINE constexpr uint64_t ControlFlagOffset(const WindowLayout &l, uint32_t worldSize,
                                                            uint32_t sourceRank, uint32_t targetServer)
{
    return l.controlFlagOffset + (targetServer * worldSize + sourceRank) * CONTROL_FLAG_STRIDE;
}
CAM_MOE_LAYERED_INLINE constexpr uint64_t BaseAckFlagOffset(const WindowLayout &l, uint32_t worldSize,
                                                            uint32_t sourceRank, uint32_t targetServer)
{
    return l.baseAckFlagOffset + (targetServer * worldSize + sourceRank) * CONTROL_FLAG_STRIDE;
}
CAM_MOE_LAYERED_INLINE constexpr uint64_t ProxyAggregateOffset(const WindowLayout &l, uint32_t proxyRankInServer)
{
    return l.proxyAggregateOffset + proxyRankInServer * l.proxyAggregateStride;
}
CAM_MOE_LAYERED_INLINE constexpr uint64_t ProxyAggregateStageOffset(const WindowLayout &l, uint32_t ownerRankInServer)
{
    return l.proxyAggregateStageOffset + ownerRankInServer * l.proxyAggregateStride;
}
CAM_MOE_LAYERED_INLINE constexpr uint64_t ProxyAggregateFlagOffset(const WindowLayout &l, uint32_t proxyRankInServer)
{
    return l.proxyAggregateFlagOffset + proxyRankInServer * sizeof(uint64_t);
}
CAM_MOE_LAYERED_INLINE constexpr uint64_t ProxyOwnerCountOffset(uint32_t sourceServer, uint32_t localExpert,
                                                                uint32_t localExperts)
{
    return (static_cast<uint64_t>(sourceServer) * localExperts + localExpert) * sizeof(int32_t);
}
CAM_MOE_LAYERED_INLINE constexpr uint64_t DispatchPayloadAddress(const WindowLayout &l, uint64_t compactOffset)
{
    return l.dispatchPayloadOffset + compactOffset;
}
CAM_MOE_LAYERED_INLINE constexpr uint64_t DispatchMetadataFanoutOffset(const WindowLayout &l, uint32_t ranksPerServer,
                                                                       uint32_t sourceServer, uint32_t proxyInServer)
{
    return l.dispatchFanoutOffset +
           (static_cast<uint64_t>(sourceServer) * ranksPerServer + proxyInServer) * l.dispatchFanoutBatchStride;
}
CAM_MOE_LAYERED_INLINE constexpr uint64_t DispatchOwnerPipelineFlagOffset(const WindowLayout &l,
                                                                          uint32_t ranksPerServer,
                                                                          uint32_t sourceServer, uint32_t proxyInServer,
                                                                          uint32_t slot)
{
    return l.dispatchPipelineFlagOffset + ((static_cast<uint64_t>(sourceServer) * ranksPerServer + proxyInServer) *
                                               (static_cast<uint64_t>(l.dispatchMaxChunks) + 2U) +
                                           slot) *
                                              sizeof(uint64_t);
}
CAM_MOE_LAYERED_INLINE constexpr uint64_t DispatchSourcePipelineFlagOffset(const WindowLayout &l, uint32_t serverNum,
                                                                           uint32_t ranksPerServer,
                                                                           uint32_t sourceServer, uint32_t targetServer,
                                                                           uint32_t slot)
{
    return l.stateOffset + ((static_cast<uint64_t>(targetServer) * serverNum + sourceServer) * l.stateLaneStride + 13U +
                            static_cast<uint64_t>(ranksPerServer) * ranksPerServer + slot) *
                               sizeof(uint64_t);
}
CAM_MOE_LAYERED_INLINE constexpr uint64_t PhaseFlagOffset(const WindowLayout &l, uint32_t serverNum,
                                                          uint32_t sourceServer, uint32_t targetServer, Phase phase)
{
    return l.stateOffset +
           ((targetServer * serverNum + sourceServer) * l.stateLaneStride + static_cast<uint32_t>(phase)) *
               sizeof(uint64_t);
}
CAM_MOE_LAYERED_INLINE WindowLayout MakeWindowLayout(uint32_t expertNum, uint32_t worldSize, uint32_t ranksPerServer,
                                                     uint32_t serverNum, uint32_t rounds, uint32_t slots,
                                                     uint32_t recordStride, uint32_t partialStride, uint32_t topk,
                                                     bool hybridDeployment = false)
{
    WindowLayout l{};
    const uint32_t expertsPerServer = expertNum / serverNum;
    l.controlStride = AlignUp64(AllRoundCountPayloadBytes(rounds, expertsPerServer), CACHE_LINE_ISOLATION_BYTES);
    l.baseAckStride = AlignUp64(AllRoundBasePayloadBytes(rounds, expertsPerServer), CACHE_LINE_ISOLATION_BYTES);
    // RDMA control region: controls are addressed by source rank, not by
    // source server. This is
    // Required when every source rank sends to its same-rank proxy in the
    // target server. The source-rank dimension keeps proxy lanes distinct.
    l.controlBytes = l.controlStride * serverNum * worldSize;
    l.baseAckOffset = AlignUp64(l.controlBytes, CACHE_LINE_ISOLATION_BYTES);
    l.baseAckBytes = l.baseAckStride * serverNum * worldSize;
    l.controlFlagOffset = AlignUp64(l.baseAckOffset + l.baseAckBytes, CACHE_LINE_ISOLATION_BYTES);
    const uint64_t controlFlagBytes = static_cast<uint64_t>(serverNum) * worldSize * CONTROL_FLAG_STRIDE;
    l.baseAckFlagOffset = AlignUp64(l.controlFlagOffset + controlFlagBytes, CACHE_LINE_ISOLATION_BYTES);
    // Server-local DataCopy staging starts after all RDMA payloads and flags.
    // It is deliberately not overlaid with the RDMA control region.
    const uint32_t localExperts = ranksPerServer == 0U ? 0U : expertsPerServer / ranksPerServer;
    // Staging and destination lanes intentionally use the same owner-local
    // source-preserving layout, so one proxy can copy a complete owner matrix.
    l.proxyAggregateStride = AlignUp64(static_cast<uint64_t>(serverNum) * rounds * localExperts * sizeof(int32_t),
                                       CACHE_LINE_ISOLATION_BYTES);
    const uint64_t baseAckFlagBytes = static_cast<uint64_t>(serverNum) * worldSize * CONTROL_FLAG_STRIDE;
    l.proxyAggregateStageOffset = AlignUp64(l.baseAckFlagOffset + baseAckFlagBytes);
    l.proxyAggregateStageBytes = l.proxyAggregateStride * ranksPerServer;
    l.proxyAggregateOffset = AlignUp64(l.proxyAggregateStageOffset + l.proxyAggregateStageBytes);
    l.proxyAggregateBytes = l.proxyAggregateStride * ranksPerServer;
    l.proxyAggregateFlagOffset = AlignUp64(l.proxyAggregateOffset + l.proxyAggregateBytes);
    const uint64_t baseAckSliceBytes =
        AlignUp64(static_cast<uint64_t>(rounds) * localExperts * sizeof(int32_t), CACHE_LINE_ISOLATION_BYTES);
    l.baseAckStageOffset =
        AlignUp64(l.proxyAggregateFlagOffset + static_cast<uint64_t>(ranksPerServer) * sizeof(uint64_t),
                  CACHE_LINE_ISOLATION_BYTES);
    l.baseAckStageSliceStride = baseAckSliceBytes;
    l.baseAckStageOwnerRowsOffset = l.baseAckStageSliceStride * serverNum;
    l.baseAckStageProxyStride =
        AlignUp64(l.baseAckStageOwnerRowsOffset + CACHE_LINE_ISOLATION_BYTES, CACHE_LINE_ISOLATION_BYTES);
    l.baseAckStageBytes = l.baseAckStageProxyStride * ranksPerServer;
    l.baseAckFanoutOffset = AlignUp64(l.baseAckStageOffset + l.baseAckStageBytes, CACHE_LINE_ISOLATION_BYTES);
    l.baseAckFanoutOwnerStride = l.baseAckStageProxyStride;
    l.baseAckFanoutBytes = l.baseAckFanoutOwnerStride * ranksPerServer;
    l.baseAckFanoutFlagOffset = AlignUp64(l.baseAckFanoutOffset + l.baseAckFanoutBytes);
    const uint64_t metadataEnd =
        AlignUp64(l.baseAckFanoutFlagOffset + static_cast<uint64_t>(ranksPerServer) * sizeof(uint64_t));
#ifdef __DAV_C310__
    l.dispatchPayloadOffset = hybridDeployment ? Moe::A5WindowLayout::kDataOffset : metadataEnd;
#else
    (void)hybridDeployment;
    l.dispatchPayloadOffset = metadataEnd;
#endif
    // Reserve one worst-case arena for compact per-call outbound staging and
    // inbound ingress. Notify computes their actual, non-overlapping ranges.
    l.dispatchPacketMetaOffset = sizeof(DispatchBatchMeta);
    l.dispatchTokenRecordOffset =
        AlignUp64(l.dispatchPacketMetaOffset + static_cast<uint64_t>(slots) * sizeof(DispatchPacketMeta));
    l.dispatchMemberMetaOffset = AlignUp64(l.dispatchTokenRecordOffset + static_cast<uint64_t>(slots) * recordStride);
    l.dispatchPacketMetaStride = sizeof(DispatchPacketMeta);
    l.dispatchMemberMetaStride = sizeof(DispatchMemberMeta);
    const uint64_t maxMembers = static_cast<uint64_t>(slots) * topk;
    l.dispatchBatchStride = AlignUp64(l.dispatchMemberMetaOffset + maxMembers * sizeof(DispatchMemberMeta));
    l.dispatchPayloadCapacity = static_cast<uint64_t>(2U) * serverNum * l.dispatchBatchStride;
    l.dispatchFanoutMemberMetaOffset =
        AlignUp64(sizeof(DispatchBatchMeta) + static_cast<uint64_t>(slots) * sizeof(DispatchPacketMeta));
    l.dispatchFanoutBatchStride = AlignUp64(l.dispatchFanoutMemberMetaOffset + maxMembers * sizeof(DispatchMemberMeta));
    l.dispatchFanoutOffset = AlignUp64(l.dispatchPayloadOffset + l.dispatchPayloadCapacity);
    l.dispatchFanoutBytes = static_cast<uint64_t>(serverNum) * ranksPerServer * l.dispatchFanoutBatchStride;
    l.dispatchMaxChunks = DispatchMaxChunks(slots, recordStride);
    l.dispatchPipelineFlagOffset = AlignUp64(l.dispatchFanoutOffset + l.dispatchFanoutBytes);
    l.dispatchPipelineFlagBytes = static_cast<uint64_t>(serverNum) * ranksPerServer *
                                  (static_cast<uint64_t>(l.dispatchMaxChunks) + 2U) * sizeof(uint64_t);
    l.partialOffset = AlignUp64(l.dispatchPipelineFlagOffset + l.dispatchPipelineFlagBytes);
    l.localPartialBytes =
        static_cast<uint64_t>(ranksPerServer) * ranksPerServer * serverNum * serverNum * slots * partialStride;
    l.returnPartialOffset = AlignUp64(l.partialOffset + l.localPartialBytes);
    l.returnPartialBytes = static_cast<uint64_t>(ranksPerServer) * serverNum * serverNum * slots * partialStride;
    l.stateOffset = AlignUp64(l.returnPartialOffset + l.returnPartialBytes);
    l.stateLaneStride = 16U + static_cast<uint64_t>(ranksPerServer) * ranksPerServer + l.dispatchMaxChunks;
    // Flags are deliberately separate from payload. A receiver never treats
    // a record as readable until its phase flag reaches the matching epoch.
    l.requiredBytes = AlignUp64(l.stateOffset + static_cast<uint64_t>(serverNum) * serverNum * rounds *
                                                    l.stateLaneStride * sizeof(uint64_t));
    return l;
}

// Combine runs after Dispatch has released its Window lanes. Owner rows,
// outbound partials and inbound partials occupy distinct ranges even when
// WindowIn and WindowOut alias on A5.
CAM_MOE_LAYERED_INLINE constexpr uint32_t CombineOwnerRowUbBytes(uint32_t hidden, uint32_t elementBytes)
{
    return AlignUp(hidden * elementBytes);
}
CAM_MOE_LAYERED_INLINE constexpr uint32_t CombinePartialUbBytes(uint32_t hidden)
{
    return AlignUp(hidden * sizeof(float));
}
CAM_MOE_LAYERED_INLINE constexpr uint64_t CombineExpertBasesUbBytes(uint32_t localExperts)
{
    return AlignUp64((static_cast<uint64_t>(localExperts) + 1U) * sizeof(uint32_t));
}
static_assert(CombineOwnerRowUbBytes(33U, 2U) == 96U && CombinePartialUbBytes(33U) == 160U &&
                  CombineOwnerRowUbBytes(513U, 2U) == 1056U && CombinePartialUbBytes(513U) == 2080U,
              "Combine UB buffers must cover the complete hidden dimension");
static_assert(CombineExpertBasesUbBytes(0U) == 32U && CombineExpertBasesUbBytes(8U) == 64U,
              "Combine expert-base UB must be 32-byte aligned");

struct CombineWindowLayout {
    uint64_t ownerStageOffset;
    uint64_t ownerStageStride;
    uint64_t expertBaseOffset;
    uint64_t expertBaseStride;
    uint64_t expertBaseBytes;
    uint64_t proxyPartialOffset;
    uint64_t sourcePartialOffset;
    uint64_t partialStride;
    uint64_t ownerReadyOffset;
    uint64_t ownerConsumedOffset;
    uint64_t serverReadyOffset;
    uint64_t sourceReadyOffset;
    uint64_t sourceConsumedOffset;
    uint64_t serverFlagStageOffset;
    uint64_t sourceFlagStageOffset;
    uint64_t consumedFlagStageOffset;
    uint64_t requiredBytes;
};

// Short flag writes from different AIVs must not share a cache line.
constexpr uint64_t COMBINE_FLAG_STRIDE = 64U;

CAM_MOE_LAYERED_INLINE CombineWindowLayout MakeCombineWindowLayout(const WindowLayout &base, uint32_t serverNum,
                                                                   uint32_t ranksPerServer, uint32_t localExperts,
                                                                   uint32_t ownerCapacity, uint32_t maxBs,
                                                                   uint32_t hidden, uint32_t elementBytes)
{
    CombineWindowLayout c{};
    c.ownerStageStride = AlignUp64(static_cast<uint64_t>(hidden) * elementBytes);
    c.ownerStageOffset = base.partialOffset;
    c.expertBaseOffset = AlignUp64(c.ownerStageOffset + static_cast<uint64_t>(ownerCapacity) * c.ownerStageStride);
    c.expertBaseStride = AlignUp64((static_cast<uint64_t>(localExperts) + 1U) * sizeof(uint32_t));
    c.expertBaseBytes = static_cast<uint64_t>(ranksPerServer) * c.expertBaseStride;
    c.partialStride = AlignUp64(static_cast<uint64_t>(hidden) * sizeof(float));
    c.proxyPartialOffset = AlignUp64(c.expertBaseOffset + c.expertBaseBytes);
    const uint64_t partialBytes = static_cast<uint64_t>(serverNum) * maxBs * c.partialStride;
    c.sourcePartialOffset = AlignUp64(c.proxyPartialOffset + partialBytes);
    c.ownerReadyOffset = AlignUp64(c.sourcePartialOffset + partialBytes, COMBINE_FLAG_STRIDE);
    c.ownerConsumedOffset = AlignUp64(c.ownerReadyOffset + ranksPerServer * COMBINE_FLAG_STRIDE, COMBINE_FLAG_STRIDE);
    c.serverReadyOffset = AlignUp64(c.ownerConsumedOffset + ranksPerServer * COMBINE_FLAG_STRIDE, COMBINE_FLAG_STRIDE);
    c.sourceReadyOffset = AlignUp64(c.serverReadyOffset + serverNum * COMBINE_FLAG_STRIDE, COMBINE_FLAG_STRIDE);
    c.sourceConsumedOffset = AlignUp64(c.sourceReadyOffset + serverNum * COMBINE_FLAG_STRIDE, COMBINE_FLAG_STRIDE);
    c.serverFlagStageOffset = AlignUp64(c.sourceConsumedOffset + serverNum * COMBINE_FLAG_STRIDE, COMBINE_FLAG_STRIDE);
    c.sourceFlagStageOffset = AlignUp64(c.serverFlagStageOffset + serverNum * COMBINE_FLAG_STRIDE, COMBINE_FLAG_STRIDE);
    c.consumedFlagStageOffset =
        AlignUp64(c.sourceFlagStageOffset + serverNum * COMBINE_FLAG_STRIDE, COMBINE_FLAG_STRIDE);
    c.requiredBytes = AlignUp64(c.consumedFlagStageOffset + serverNum * COMBINE_FLAG_STRIDE, COMBINE_FLAG_STRIDE);
    return c;
}

struct PlanLayout {
    uint32_t packetCount;
    uint32_t dispatchMemberCount;
    uint32_t packetSlot;
    uint32_t expertCount;
    uint32_t localExpertOffset;
    uint32_t localExpertCount;
    uint32_t packetDesc;
    uint32_t memberDesc;
    uint32_t sourceReturnBase;
    uint32_t outboundBase;
    uint32_t outboundBytes;
    uint32_t outboundTotalBytes;
    uint32_t sendIngressBase;
    uint32_t sendIngressBytes;
    uint32_t recvIngressBase;
    uint32_t recvIngressBytes;
    uint32_t reverseCount;
    uint32_t reverseReturnBase;
    uint32_t reversePacket;
    uint32_t reverseMember;
    uint32_t overflow;
    uint32_t error;
    uint32_t total;
};

CAM_MOE_LAYERED_INLINE PlanLayout MakePlanLayout(uint32_t bs, uint32_t maxBs, uint32_t k, uint32_t expertNum,
                                                 uint32_t worldSize, uint32_t serverNum, uint32_t ranksPerServer,
                                                 uint32_t rounds, uint32_t slots)
{
    PlanLayout l{};
    l.packetCount = HeaderElements();
    l.dispatchMemberCount = l.packetCount + rounds * serverNum;
    l.packetSlot = l.dispatchMemberCount + rounds * serverNum;
    // Counts are Notify-local input to CountControl. Final member rows are
    // written directly into MemberDesc once the returned BaseAck arrives.
    // token determines its physical round, so slots only need [BS, server]
    // storage rather than a redundant [round, BS, server] matrix.
    const uint32_t expertsPerServer = serverNum == 0U ? 0U : expertNum / serverNum;
    l.expertCount = l.packetSlot + maxBs * serverNum;
    l.localExpertOffset = l.expertCount + rounds * serverNum * expertsPerServer;
    l.localExpertCount = l.localExpertOffset + LocalExpertNum(expertNum, worldSize);
    l.packetDesc = l.localExpertCount + LocalExpertNum(expertNum, worldSize);
    l.memberDesc = l.packetDesc + rounds * serverNum * slots * PacketElements();
    l.sourceReturnBase = l.memberDesc + rounds * serverNum * slots * k * MemberElements();
    l.outboundBase = l.sourceReturnBase + serverNum;
    l.outboundBytes = l.outboundBase + 2U * serverNum;
    l.outboundTotalBytes = l.outboundBytes + 2U * serverNum;
    l.sendIngressBase = l.outboundTotalBytes + 2U;
    l.sendIngressBytes = l.sendIngressBase + 2U * serverNum;
    l.recvIngressBase = l.sendIngressBytes + 2U * serverNum;
    l.recvIngressBytes = l.recvIngressBase + 2U * serverNum;
    l.reverseCount = l.recvIngressBytes + 2U * serverNum;
    l.reverseReturnBase = l.reverseCount + serverNum;
    l.reversePacket = l.reverseReturnBase + serverNum;
    l.reverseMember = l.reversePacket + worldSize * maxBs * ReversePacketElements();
    l.overflow = l.reverseMember + worldSize * maxBs * k * ReverseMemberElements();
    l.error = l.overflow + 1U;
    l.total = l.error + 1U;
    return l;
}

}  // namespace CamMoeLayered

#undef CAM_MOE_LAYERED_INLINE

#endif
