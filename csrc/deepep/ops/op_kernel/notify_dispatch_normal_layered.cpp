#include "kernel_operator.h"
#include "notify_dispatch_normal_layered_tiling.h"
#include "cam_moe_layered_plan.h"
#include "cam_moe_layered_comm.h"

using namespace AscendC;

namespace {
using namespace CamMoeLayered;

#ifdef __DAV_C310__
constexpr uint32_t kHcommUbBytes = HCOMM_UB_BUF_SIZE;
#else
constexpr uint32_t kHcommUbBytes = 0U;
#endif

constexpr uint32_t kInitTileBytes = 60U * 1024U;
constexpr uint32_t kInitTileElements = kInitTileBytes / sizeof(int32_t);
constexpr uint32_t kReduceTileBytes = 60U * 1024U;
constexpr uint32_t kReduceTileElements = kReduceTileBytes / sizeof(int32_t);
constexpr uint32_t kAggregateCopyTileBytes = 60U * 1024U;
constexpr uint32_t kSharedWorkUbBytes = 2U * kReduceTileBytes;
constexpr uint32_t kDataCopyThreshold = 128U;
constexpr uint32_t kRoundMaxScratchStride = ALIGN_BYTES / sizeof(int32_t);

static_assert(kInitTileBytes <= kReduceTileBytes);
static_assert(kAggregateCopyTileBytes <= kReduceTileBytes);
static_assert(kReduceTileBytes % 32U == 0U);

class NotifyDispatchNormalLayered
{
public:
    __aicore__ inline NotifyDispatchNormalLayered() {}

    __aicore__ inline void Init(GM_ADDR roceContext, GM_ADDR topkIdx, GM_ADDR layeredPlan, GM_ADDR recvTokensPerExpert,
                                GM_ADDR expertTotalCount, GM_ADDR workspace, const CamMoeLayeredInfo &info,
                                LocalTensor<uint64_t> &doorbell, LocalTensor<uint32_t> &head,
                                LocalTensor<int32_t> &initUb, LocalTensor<int32_t> &reduceSrcUb,
                                LocalTensor<int32_t> &reduceAccUb, LocalTensor<uint8_t> &aggregateCopyUb,
                                LocalTensor<uint8_t> &hcommUb)
    {
        topk_ = (__gm__ int32_t *)topkIdx;
        plan_ = (__gm__ int32_t *)layeredPlan;
        recv_ = (__gm__ int32_t *)recvTokensPerExpert;
        total_ = (__gm__ int32_t *)expertTotalCount;
        workspace_ = (__gm__ int32_t *)workspace;
        info_ = info;
        doorbell_ = &doorbell;
        head_ = &head;
        initUb_ = &initUb;
        reduceSrcUb_ = &reduceSrcUb;
        reduceAccUb_ = &reduceAccUb;
        aggregateCopyUb_ = &aggregateCopyUb;
        localExperts_ = info_.moeExpertNum / info_.epWorldSize;
        expertsPerServer_ = info_.moeExpertNum / info_.serverNum;
        sourceServer_ = info_.epRankId / info_.ranksPerServer;
        comm_.Init(info_.epRankId, roceContext, hcommUb);
        layout_ = CamMoeLayered::MakeWindowLayout(info_.moeExpertNum, info_.epWorldSize, info_.ranksPerServer,
                                                  info_.serverNum, info_.physicalRounds, info_.slotsPerServer,
                                                  info_.recordStride, info_.partialStride, info_.k);
        countSlice_ = info_.physicalRounds * info_.serverNum * expertsPerServer_;
        packetSlice_ = info_.physicalRounds * info_.serverNum;
        countSlicePadded_ = AlignWorkspaceElements(countSlice_);
        packetSlicePadded_ = AlignWorkspaceElements(packetSlice_);
        memberCountSlicePadded_ = packetSlicePadded_;
    }

    __aicore__ inline void Process()
    {
        const uint64_t registeredWindowBytes = comm_.RegisteredWindowBytes();
        if (registeredWindowBytes != 0U && layout_.requiredBytes > registeredWindowBytes) {
            if (GetBlockIdx() == 0U) {
                plan_[info_.errorOffset] = static_cast<int32_t>(Phase::ERROR_OVERFLOW);
                total_[0] = 0;
            }
            return;
        }
        Cleanup();
        InitPlanHeader();  // 初始化 plan 信息和状态（AIV 0）
        SyncAll<true>();
        BuildLocalRoutePlanParallel();  // 根据 topk_idx 构建本地路由关系（按token分核）
        SyncAll<true>();
        ReduceLocalExpertCounts();  // 合并多 AIV 计数，并修正全局 ordinal（按token分核，但需要依赖其他AIV计算的数据）
        SyncAll<true>();
        PublishAllRoundCountControls();  // server间发送给同号卡（按server分核）
        AggregateServerCounts();         // 本地rank重排并转发给server内其他rank
        ApplyProxyAggregatesToOwner();   // 等待数据到达，本地按照专家重排，记录recv_x
        SyncAll<true>();
        FanoutBaseAckSlicesToProxies();  // 将计算好的行号发送给本地proxy
        WaitAllBaseAckSlices();          // 等待所有的本地rank收齐
        SyncAll<true>();
        AssembleBaseAcksAtProxies();   // inbuffer搬到outbuffer
        PublishBaseAcksFromProxies();  // 直接rdma发送，不需要重排
        ApplyAllReturnedRowBases();    // 等待数据收齐，并回填每个member的行号
    }

private:
    __aicore__ inline uint64_t CountSlice(uint32_t aiv) const
    {
        return (uint64_t)aiv * countSlicePadded_;
    }

    __aicore__ inline uint64_t PacketSlice(uint32_t aiv) const
    {
        return (uint64_t)info_.aivNum * countSlicePadded_ + (uint64_t)aiv * packetSlicePadded_;
    }

    __aicore__ inline uint64_t MemberCountSlice(uint32_t aiv) const
    {
        return (uint64_t)info_.aivNum * countSlicePadded_ + (uint64_t)info_.aivNum * packetSlicePadded_ +
               (uint64_t)aiv * memberCountSlicePadded_;
    }

    __aicore__ inline uint64_t RoundMaxScratch(uint32_t aiv) const
    {
        const uint32_t aivNum = info_.aivNum == 0U ? 1U : info_.aivNum;
        const uint64_t counters =
            static_cast<uint64_t>(aivNum) * (countSlicePadded_ + packetSlicePadded_ + memberCountSlicePadded_);
        return counters + static_cast<uint64_t>(aiv) * kRoundMaxScratchStride;
    }

    __aicore__ inline uint32_t AlignWorkspaceElements(uint32_t elements) const
    {
        constexpr uint32_t elementsPerBlock = 32U / sizeof(int32_t);
        return (elements + elementsPerBlock - 1U) / elementsPerBlock * elementsPerBlock;
    }

    __aicore__ inline void ZeroWorkspaceRange(__gm__ int32_t *dst, uint32_t elements)
    {
        if (elements == 0U) {
            return;
        }
        Duplicate(*initUb_, 0, kInitTileElements);
        for (uint32_t offset = 0; offset < elements; offset += kInitTileElements) {
            const uint32_t count = min(kInitTileElements, elements - offset);
            GlobalTensor<int32_t> dstTensor;
            dstTensor.SetGlobalBuffer(dst + offset);
            DataCopy(dstTensor, *initUb_, count);
        }
    }

    __aicore__ inline void FillPlanRange(uint32_t begin, uint32_t end, int32_t value)
    {
        if (begin >= end) {
            return;
        }
        const uint32_t length = end - begin;
        if (length < kDataCopyThreshold) {
            for (uint32_t i = begin; i < end; ++i) {
                plan_[i] = value;
            }
            return;
        }

        // DataCopy operates on aligned GM/UB blocks. Handle the two partial
        // blocks with scalar stores and copy only the aligned middle.
        constexpr uint32_t elementsPerBlock = 32U / sizeof(int32_t);
        const uint32_t alignedBegin = (begin + elementsPerBlock - 1U) / elementsPerBlock * elementsPerBlock;
        const uint32_t alignedEnd = end / elementsPerBlock * elementsPerBlock;
        const uint32_t scalarBeginEnd = min(alignedBegin, end);
        for (uint32_t i = begin; i < scalarBeginEnd; ++i) {
            plan_[i] = value;
        }

        if (alignedBegin < alignedEnd) {
            for (uint32_t offset = alignedBegin; offset < alignedEnd;) {
                const uint32_t count = min(kInitTileElements, alignedEnd - offset);
                Duplicate(*initUb_, value, count);
                GlobalTensor<int32_t> tileDst;
                tileDst.SetGlobalBuffer(plan_ + offset);
                DataCopy(tileDst, *initUb_, count);
                offset += count;
            }
        }

        for (uint32_t i = alignedEnd; i < end; ++i) {
            plan_[i] = value;
        }
    }

    __aicore__ inline void Cleanup()
    {
        const uint32_t aiv = GetBlockIdx();
        const uint32_t aivNum = info_.aivNum == 0U ? 1U : info_.aivNum;
        const uint64_t packetSlotElements = static_cast<uint64_t>(info_.maxBs) * info_.serverNum;
        const uint32_t packetSlotBegin = info_.packetSlotOffset;
        const uint32_t packetSlotEnd = packetSlotBegin + static_cast<uint32_t>(packetSlotElements);
        const uint32_t dataBegin = HeaderElements();
        // overflow/error/state are single-writer fields initialized by
        // InitPlanHeader(). Keep them out of the parallel data clearing range.
        const uint32_t dataEnd = info_.overflowOffset;
        const uint32_t dataSize = dataEnd - dataBegin;
        const uint32_t begin = dataBegin + static_cast<uint32_t>(static_cast<uint64_t>(dataSize) * aiv / aivNum);
        const uint32_t end = dataBegin + static_cast<uint32_t>(static_cast<uint64_t>(dataSize) * (aiv + 1U) / aivNum);

        // Clear all ordinary plan metadata except packetSlot, whose sentinel
        // value is INVALID_SLOT rather than zero.
        if (begin < packetSlotBegin) {
            FillPlanRange(begin, min(end, packetSlotBegin), 0);
        }
        if (end > packetSlotEnd) {
            const uint32_t zeroBegin = begin > packetSlotEnd ? begin : packetSlotEnd;
            FillPlanRange(zeroBegin, end, 0);
        }

        // Initialize each AIV's private reduction slices before route building.
        ZeroWorkspaceRange(workspace_ + CountSlice(aiv), countSlicePadded_);
        ZeroWorkspaceRange(workspace_ + PacketSlice(aiv), packetSlicePadded_);
        ZeroWorkspaceRange(workspace_ + MemberCountSlice(aiv), memberCountSlicePadded_);

        // packetSlot is initialized independently because zero is a valid slot.
        const uint32_t packetBegin = static_cast<uint32_t>(packetSlotElements * aiv / aivNum);
        const uint32_t packetEnd = static_cast<uint32_t>(packetSlotElements * (aiv + 1U) / aivNum);
        FillPlanRange(packetSlotBegin + packetBegin, packetSlotBegin + packetEnd, INVALID_SLOT);
    }

    __aicore__ inline void ReduceWorkspaceRange(uint32_t dstOffset, uint32_t srcOffset, uint32_t elements,
                                                uint64_t sourceBase, uint32_t sourceStride)
    {
        if (elements == 0U) {
            return;
        }
        const uint32_t aivNum = info_.aivNum == 0U ? 1U : info_.aivNum;
        if (elements < kDataCopyThreshold) {
            for (uint32_t i = 0; i < elements; ++i) {
                int32_t value = 0;
                for (uint32_t p = 0; p < aivNum; ++p) {
                    value += workspace_[sourceBase + static_cast<uint64_t>(p) * sourceStride + srcOffset + i];
                }
                plan_[dstOffset + i] = value;
            }
            return;
        }

        for (uint32_t offset = 0; offset < elements; offset += kReduceTileElements) {
            const uint32_t count = min(kReduceTileElements, elements - offset);
            Duplicate(*reduceAccUb_, 0, count);
            for (uint32_t p = 0; p < aivNum; ++p) {
                GlobalTensor<int32_t> srcTensor;
                srcTensor.SetGlobalBuffer(workspace_ + sourceBase + static_cast<uint64_t>(p) * sourceStride +
                                          srcOffset + offset);
                DataCopyExtParams copyParams{1U, count * sizeof(int32_t), 0U, 0U, 0U};
                DataCopyPadExtParams<int32_t> padParams{false, 0U, 0U, 0};
                DataCopyPad(*reduceSrcUb_, srcTensor, copyParams, padParams);
                SyncFunc<HardEvent::MTE2_V>();
                Add(*reduceAccUb_, *reduceAccUb_, *reduceSrcUb_, count);
                SyncFunc<HardEvent::V_MTE2>();
            }
            SyncFunc<HardEvent::V_MTE3>();
            GlobalTensor<int32_t> dstTensor;
            dstTensor.SetGlobalBuffer(plan_ + dstOffset + offset);
            DataCopyExtParams copyParams{1U, count * sizeof(int32_t), 0U, 0U, 0U};
            DataCopyPadExtParams<int32_t> padParams{false, 0U, 0U, 0};
            DataCopyPad(dstTensor, *reduceAccUb_, copyParams, padParams);
            SyncFunc<HardEvent::MTE3_V>();
        }
    }

    __aicore__ inline void InitPlanHeader()
    {
        if (GetBlockIdx() != 0U) {
            return;
        }

        auto *h = (__gm__ PlanHeader *)plan_;
        h->magic = PLAN_MAGIC;
        h->version = PLAN_VERSION;
        h->epochLo = (uint32_t)info_.epoch;
        h->epochHi = (uint32_t)(info_.epoch >> 32U);
        h->bs = info_.bs;
        h->maxBs = info_.maxBs;
        h->validRounds = info_.validRounds;
        h->hiddenSize = info_.h;
        h->topk = info_.k;
        h->expertNum = info_.moeExpertNum;
        h->worldSize = info_.epWorldSize;
        h->serverNum = info_.serverNum;
        h->ranksPerServer = info_.ranksPerServer;
        h->rankId = info_.epRankId;
        h->rankInServer = info_.rankInServer;
        h->physicalRounds = info_.physicalRounds;
        h->slotsPerServer = info_.slotsPerServer;
        h->recordStrideBytes = info_.recordStride;
        h->packetCountOffset = info_.packetCountOffset;
        h->dispatchMemberCountOffset = info_.dispatchMemberCountOffset;
        h->packetSlotOffset = info_.packetSlotOffset;
        h->expertCountOffset = info_.expertCountOffset;
        h->localExpertOffset = info_.localExpertOffset;
        h->localExpertCountOffset = info_.localExpertCountOffset;
        h->packetDescOffset = info_.packetDescOffset;
        h->memberDescOffset = info_.memberDescOffset;
        h->overflowOffset = info_.overflowOffset;
        h->errorOffset = info_.errorOffset;
        h->totalElements = info_.planElements;
        h->globalCombineRows = 0U;
        h->reserved[0] = 0U;
        h->reserved[1] = 0U;
        plan_[info_.errorOffset] = 0;
        total_[0] = 0;
    }

    __aicore__ inline void BuildLocalRoutePlanParallel()
    {
        const uint32_t aiv = GetBlockIdx();
        const uint32_t n = info_.aivNum == 0U ? 1U : info_.aivNum;
        const uint32_t begin = info_.bs * aiv / n;
        const uint32_t end = info_.bs * (aiv + 1U) / n;

        auto *counts = workspace_ + CountSlice(aiv);
        auto *packets = workspace_ + PacketSlice(aiv);
        auto *memberCounts = workspace_ + MemberCountSlice(aiv);
        // 遍历 token
        for (uint32_t token = begin; token < end; ++token) {
            const uint32_t slot = token;
            // 读取 top-k expert
            for (uint32_t topkSlot = 0; topkSlot < info_.k; ++topkSlot) {
                const int32_t expert = topk_[(uint64_t)token * info_.k + topkSlot];
                if (expert == INVALID_SLOT) {
                    continue;
                }
                const uint32_t target = (uint32_t)expert / expertsPerServer_;
                // 每个 token/server 只占一个 packet。
                const uint32_t packet = target * info_.slotsPerServer + slot;
                auto *packetDesc = (__gm__ SourcePacketDesc *)(plan_ + info_.packetDescOffset) + packet;
                if (packetDesc->memberCount == 0) {
                    packetDesc->sourceToken = static_cast<int32_t>(token);
                    packetDesc->memberBegin = static_cast<int32_t>(memberCounts[target]);
                    // Each AIV assigns a stable local position.  Reduction
                    // below adds the preceding AIV packet counts.
                    packetDesc->dispatchOrdinal = static_cast<int32_t>(packets[target]++);
                }
                // member = token 的有效 top-k
                const uint32_t member = static_cast<uint32_t>(packetDesc->memberCount++);
                memberCounts[target]++;
                const uint32_t mb = info_.memberDescOffset + (packet * info_.k + member) * MemberElements();
                // 发往目标 server、目标 expert 的局部序号。
                const uint32_t expertInServer = static_cast<uint32_t>(expert) - target * expertsPerServer_;
                const uint32_t ordinal = counts[target * expertsPerServer_ + expertInServer]++;
                plan_[mb] = expert;
                plan_[mb + 1U] = static_cast<int32_t>(ordinal);  // 暂存本地 expert ordinal，稍后改为最终行号
                plan_[mb + 2U] = (int32_t)topkSlot;
                // [bs][serverNum]
                plan_[info_.packetSlotOffset + token * info_.serverNum + target] = (int32_t)slot;
            }
        }
    }

    __aicore__ inline void ReduceLocalExpertCounts()
    {
        const uint32_t aiv = GetBlockIdx();
        const uint32_t n = info_.aivNum == 0U ? 1U : info_.aivNum;
        const uint32_t packetElements = info_.serverNum;
        const uint32_t expertElements = packetElements * expertsPerServer_;
        const uint32_t packetBegin = static_cast<uint32_t>(static_cast<uint64_t>(packetElements) * aiv / n);
        const uint32_t packetEnd = static_cast<uint32_t>(static_cast<uint64_t>(packetElements) * (aiv + 1U) / n);
        const uint32_t expertBegin = static_cast<uint32_t>(static_cast<uint64_t>(expertElements) * aiv / n);
        const uint32_t expertEnd = static_cast<uint32_t>(static_cast<uint64_t>(expertElements) * (aiv + 1U) / n);
        const uint32_t memberBegin = static_cast<uint32_t>(static_cast<uint64_t>(packetElements) * aiv / n);
        const uint32_t memberEnd = static_cast<uint32_t>(static_cast<uint64_t>(packetElements) * (aiv + 1U) / n);
        // 每个目标 server 的 packet 数
        ReduceWorkspaceRange(info_.packetCountOffset + packetBegin, packetBegin, packetEnd - packetBegin,
                             static_cast<uint64_t>(n) * countSlicePadded_, packetSlicePadded_);
        ReduceWorkspaceRange(
            info_.dispatchMemberCountOffset + memberBegin, memberBegin, memberEnd - memberBegin,
            static_cast<uint64_t>(n) * countSlicePadded_ + static_cast<uint64_t>(n) * packetSlicePadded_,
            memberCountSlicePadded_);
        // 每个目标 server/expert 的 member 数
        ReduceWorkspaceRange(info_.expertCountOffset + expertBegin, expertBegin, expertEnd - expertBegin, 0U,
                             countSlicePadded_);
        const uint32_t begin = info_.bs * aiv / n;
        const uint32_t end = info_.bs * (aiv + 1U) / n;
        // 将每个AIV的 ordinal 归约成全局的 ordinal
        for (uint32_t token = begin; token < end; ++token) {
            for (uint32_t target = 0; target < info_.serverNum; ++target) {
                const uint32_t packet = target * info_.slotsPerServer + token;
                auto *packetDesc = (__gm__ SourcePacketDesc *)(plan_ + info_.packetDescOffset) + packet;
                if (packetDesc->memberCount > 0) {
                    uint32_t packetPrefix = 0;
                    for (uint32_t p = 0; p < aiv; ++p) {
                        packetPrefix += workspace_[PacketSlice(p) + target];
                    }
                    packetDesc->dispatchOrdinal += static_cast<int32_t>(packetPrefix);
                    uint32_t memberPrefix = 0U;
                    for (uint32_t p = 0; p < aiv; ++p) {
                        memberPrefix += workspace_[MemberCountSlice(p) + target];
                    }
                    packetDesc->memberBegin += static_cast<int32_t>(memberPrefix);
                }
                for (uint32_t m = 0; m < static_cast<uint32_t>(packetDesc->memberCount); ++m) {
                    const uint32_t mb = info_.memberDescOffset + (packet * info_.k + m) * MemberElements();
                    const uint32_t e = (uint32_t)plan_[mb];
                    uint32_t prefix = 0;
                    for (uint32_t p = 0; p < aiv; ++p) {
                        prefix +=
                            workspace_[CountSlice(p) + target * expertsPerServer_ + (e - target * expertsPerServer_)];
                    }
                    plan_[mb + 1U] += static_cast<int32_t>(prefix);
                }
            }
        }
    }
    __aicore__ inline bool Wait(__gm__ uint64_t *f, uint64_t expected)
    {
        while (true) {
            cacheReadInvalidate((__gm__ uint8_t *)f, sizeof(uint64_t));
            if (*f == expected) {
                return true;
            }
            GetSystemCycle();
        }
    }

    __aicore__ inline void CopyWindowFlag(__gm__ uint64_t *dst, uint64_t value)
    {
        // WindowIn(peer) is a mapped peer-window address.  Publish the flag
        // through MTE3 instead of issuing a scalar store to that address.
        doorbell_->SetValue(0, value);
        GlobalTensor<uint64_t> dstTensor;
        dstTensor.SetGlobalBuffer(dst);
        DataCopyPad(dstTensor, *doorbell_, DataCopyExtParams{1U, sizeof(uint64_t), 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE3_S>();
    }

    __aicore__ inline void CopyWindowBytes(__gm__ uint8_t *src, __gm__ uint8_t *dst, uint64_t bytes)
    {
        for (uint64_t offset = 0; offset < bytes;) {
            const uint32_t count =
                static_cast<uint32_t>(min(static_cast<uint64_t>(kAggregateCopyTileBytes), bytes - offset));
            GlobalTensor<uint8_t> srcTensor;
            srcTensor.SetGlobalBuffer(src + offset);
            const DataCopyExtParams copyParams{1U, count, 0U, 0U, 0U};
            const DataCopyPadExtParams<uint8_t> padParams{false, 0U, 0U, 0U};
            DataCopyPad(*aggregateCopyUb_, srcTensor, copyParams, padParams);
            SyncFunc<HardEvent::MTE2_V>();
            GlobalTensor<uint8_t> dstTensor;
            dstTensor.SetGlobalBuffer(dst + offset);
            DataCopyPad(dstTensor, *aggregateCopyUb_, copyParams);
            SyncFunc<HardEvent::MTE3_S>();
            offset += count;
        }
    }
    __aicore__ inline void PublishAllRoundCountControls()
    {
        constexpr uint64_t active = 0ULL;
        // 当前 rank 的发送窗口
        auto *out = (__gm__ uint8_t *)comm_.WindowOut(info_.epRankId);
        const uint32_t aiv = GetBlockIdx();
        const uint32_t aivNum = info_.aivNum == 0U ? 1U : info_.aivNum;
        // Each target server is owned by one AIV. If there are fewer AIVs
        // than servers, ownership is reused in a deterministic stripe.
        for (uint32_t target = aiv; target < info_.serverNum; target += aivNum) {
            // [targetServer][sourceRank]
            const uint64_t off = active + ControlLaneOffset(layout_, info_.epWorldSize, info_.epRankId, target, 0);
            auto *payload = out + off;
            // 当前 rank 针对当前 targetServer 构造的完整 CountControl 消息地址
            auto *meta = (__gm__ CountControlMeta *)payload;
            for (uint32_t i = 0; i < 7U; ++i) {
                meta->padding[i] = 0U;
            }
            uint32_t returnBase = 0U;
            for (uint32_t preceding = 0U; preceding < target; ++preceding) {
                returnBase += static_cast<uint32_t>(plan_[info_.packetCountOffset + preceding]);
            }
            meta->sourceReturnBase = returnBase;
            const auto p = MakePlanLayout(info_.bs, info_.maxBs, info_.k, info_.moeExpertNum, info_.epWorldSize,
                                          info_.serverNum, info_.ranksPerServer, 1U, info_.maxBs);
            plan_[p.sourceReturnBase + target] = static_cast<int32_t>(returnBase);
            auto *entry = payload + AllRoundCountRoundOffset(0U, expertsPerServer_);
            ((__gm__ uint32_t *)entry)[0] = static_cast<uint32_t>(plan_[info_.packetCountOffset + target]);
            auto *c = (__gm__ int32_t *)(entry + sizeof(uint32_t));
            for (uint32_t e = 0; e < expertsPerServer_; ++e) {
                c[e] = plan_[info_.expertCountOffset + target * expertsPerServer_ + e];
            }
            cacheWriteThrough(payload, layout_.controlStride);
            const uint64_t flagOff = active + ControlFlagOffset(layout_, info_.epWorldSize, info_.epRankId, target);
            auto *flag = (__gm__ uint64_t *)(out + flagOff);
            // 写flag区
            *flag = ComposeFlag(info_.epoch, Phase::CONTROL_COUNT_READY);
            cacheWriteThrough((__gm__ uint8_t *)flag, sizeof(uint64_t));
            // 发送给同号卡
            const uint32_t proxyRank = target * info_.ranksPerServer + info_.rankInServer;
            auto *in = (__gm__ uint8_t *)comm_.WindowIn(proxyRank);
            // 目的位置
            auto *dst = in + off;
            auto *remoteFlag = (__gm__ uint64_t *)(in + flagOff);
            if (proxyRank == info_.epRankId) {
                CopyWindowBytes(payload, dst, layout_.controlStride);
                CopyWindowFlag(remoteFlag, *flag);
            } else {
                comm_.PostRdmaSend(payload, dst, proxyRank, layout_.controlStride, *doorbell_, *head_);
                comm_.PostReadyFlag((GM_ADDR)flag, (GM_ADDR)remoteFlag, proxyRank, *doorbell_, *head_);
            }
        }
    }

    __aicore__ inline void AggregateServerCounts()
    {
        constexpr uint64_t active = 0ULL;
        auto *in = (__gm__ uint8_t *)comm_.WindowIn(info_.epRankId);
        auto *out = (__gm__ uint8_t *)comm_.WindowOut(info_.epRankId);
        const uint32_t aiv = GetBlockIdx();
        const uint32_t aivNum = info_.aivNum == 0U ? 1U : info_.aivNum;
        const uint32_t proxyInServer = info_.rankInServer;
        const uint32_t ownerBegin = static_cast<uint32_t>(static_cast<uint64_t>(info_.ranksPerServer) * aiv / aivNum);
        const uint32_t ownerEnd =
            static_cast<uint32_t>(static_cast<uint64_t>(info_.ranksPerServer) * (aiv + 1U) / aivNum);
        // 等待所有的server间flag到达
        for (uint32_t sourceServer = aiv; sourceServer < info_.serverNum; sourceServer += aivNum) {
            const uint32_t source = proxyInServer + sourceServer * info_.ranksPerServer;
            // 遍历源rank的flag
            const uint64_t flag = active + ControlFlagOffset(layout_, info_.epWorldSize, source, sourceServer_);
            Wait((__gm__ uint64_t *)(in + flag), ComposeFlag(info_.epoch, Phase::CONTROL_COUNT_READY));
        }
        SyncAll<true>();

        // Preserve sourceServer. Owner-side BaseAck construction needs each
        // source rank's count rather than only a proxy-wide aggregate.
        const uint32_t aggregateElements = info_.serverNum * expertsPerServer_;
        const uint32_t begin = static_cast<uint32_t>(static_cast<uint64_t>(aggregateElements) * aiv / aivNum);
        const uint32_t end = static_cast<uint32_t>(static_cast<uint64_t>(aggregateElements) * (aiv + 1U) / aivNum);
        /*
         * 当前 proxy 对每个 (sourceServer, expertInServer) 按照本地 rank 维度重排，
           即将 [expertsPerServer] 拆分成 [ownerInServer][localExpert]
                [ownerInServer 0]
                    [sourceServer][localExpert]
                [ownerInServer 1]
                    [sourceServer][round][localExpert]
                [ownerInServer 2]
                    [sourceServer][round][localExpert]
         */
        // 所有专家count数据分核
        for (uint32_t index = begin; index < end; ++index) {
            const uint32_t sourceServer = index / expertsPerServer_;
            const uint32_t expert = index % expertsPerServer_;
            const uint32_t source = proxyInServer + sourceServer * info_.ranksPerServer;
            auto *payload = in + active + ControlLaneOffset(layout_, info_.epWorldSize, source, sourceServer_, 0);
            auto *p = (__gm__ int32_t *)(payload + AllRoundCountRoundOffset(0U, expertsPerServer_) + sizeof(uint32_t));
            const int32_t count = p[expert];
            const uint32_t ownerInServer = expert / localExperts_;
            const uint32_t localExpert = expert % localExperts_;
            auto *aggregate = out + active + ProxyAggregateStageOffset(layout_, proxyInServer, ownerInServer);
            *((__gm__ int32_t *)(aggregate + ProxyOwnerCountOffset(sourceServer, 0U, localExpert, 1U, localExperts_))) =
                count;
        }
        cacheWriteThrough(out + active + ProxyAggregateStageOffset(layout_, proxyInServer, 0U),
                          layout_.proxyAggregateStageStride);
        SyncAll<true>();

        // 转发给每个rank（owner），每个rank对应一个AIV来处理
        for (uint32_t ownerInServer = ownerBegin; ownerInServer < ownerEnd; ++ownerInServer) {
            const uint32_t owner = sourceServer_ * info_.ranksPerServer + ownerInServer;
            auto *ownerWindow = (__gm__ uint8_t *)comm_.WindowIn(owner);
            const uint32_t blockBytes = 32U;
            const uint32_t blockCount = static_cast<uint32_t>(layout_.proxyAggregateStride / blockBytes);
            auto *aggregate = out + active + ProxyAggregateStageOffset(layout_, proxyInServer, ownerInServer);
            for (uint32_t block = 0; block < blockCount;) {
                const uint32_t blockNum = min(kAggregateCopyTileBytes / blockBytes, blockCount - block);
                const uint32_t byteOffset = block * blockBytes;
                GlobalTensor<uint8_t> srcTensor;
                srcTensor.SetGlobalBuffer(aggregate + byteOffset);
                const DataCopyExtParams copyParams{1U, blockNum * blockBytes, 0U, 0U, 0U};
                const DataCopyPadExtParams<uint8_t> padParams{false, 0U, 0U, 0U};
                DataCopyPad(*aggregateCopyUb_, srcTensor, copyParams, padParams);
                SyncFunc<HardEvent::MTE2_V>();
                GlobalTensor<uint8_t> dstTensor;
                // 对端rank窗口中，当前rank的位置
                dstTensor.SetGlobalBuffer(ownerWindow + active + ProxyAggregateOffset(layout_, proxyInServer) +
                                          byteOffset);
                DataCopyPad(dstTensor, *aggregateCopyUb_, copyParams);
                SyncFunc<HardEvent::MTE3_S>();
                block += blockNum;
            }
        }
        // payload写完写flag通知对端rank
        for (uint32_t ownerInServer = ownerBegin; ownerInServer < ownerEnd; ++ownerInServer) {
            const uint32_t owner = sourceServer_ * info_.ranksPerServer + ownerInServer;
            auto *ownerWindow = (__gm__ uint8_t *)comm_.WindowIn(owner);
            auto *flag = (__gm__ uint64_t *)(ownerWindow + active + ProxyAggregateFlagOffset(layout_, proxyInServer));
            CopyWindowFlag(flag, ComposeFlag(info_.epoch, Phase::SERVER_FANOUT_DONE));
        }
    }

    __aicore__ inline void ApplyProxyAggregatesToOwner()
    {
        constexpr uint64_t active = 0ULL;
        auto *own = (__gm__ uint8_t *)comm_.WindowIn(info_.epRankId);
        auto *out = (__gm__ uint8_t *)comm_.WindowOut(info_.epRankId);
        const uint32_t aiv = GetBlockIdx();
        const uint32_t aivNum = info_.aivNum == 0U ? 1U : info_.aivNum;
        // 等待所有 proxy 发来的 aggregate
        for (uint32_t proxy = aiv; proxy < info_.ranksPerServer; proxy += aivNum) {
            auto *flag = (__gm__ uint64_t *)(own + active + ProxyAggregateFlagOffset(layout_, proxy));
            Wait(flag, ComposeFlag(info_.epoch, Phase::SERVER_FANOUT_DONE));
        }
        SyncAll<true>();

        // count[proxy][sourceServer][localExpert] and one base per route.
        for (uint32_t local = aiv; local < localExperts_; local += aivNum) {
            int32_t running = 0;
            // running is this expert's prefix over source ranks in stable rank order.
            // source rank 收到 BaseAck 后将本地 expert ordinal 原位转换成最终行号。
            for (uint32_t sourceServer = 0; sourceServer < info_.serverNum; ++sourceServer) {
                for (uint32_t proxy = 0; proxy < info_.ranksPerServer; ++proxy) {
                    auto *stage = out + active + BaseAckStageOffset(layout_, proxy, sourceServer);
                    *((__gm__ int32_t *)(stage + static_cast<uint64_t>(local) * sizeof(int32_t))) = running;
                    auto *counts = own + active + ProxyAggregateOffset(layout_, proxy);
                    running += *(
                        (__gm__ int32_t *)(counts + ProxyOwnerCountOffset(sourceServer, 0U, local, 1U, localExperts_)));
                }
            }
            plan_[info_.localExpertCountOffset + local] = running;
        }
        SyncAll<true>();

        if (aiv == 0U) {
            uint32_t ownerRows = 0U;
            for (uint32_t local = 0U; local < localExperts_; ++local) {
                ownerRows += static_cast<uint32_t>(plan_[info_.localExpertCountOffset + local]);
            }
            for (uint32_t proxy = 0U; proxy < info_.ranksPerServer; ++proxy) {
                *(__gm__ uint32_t *)(out + active + BaseAckStageOwnerRowsOffset(layout_, proxy)) = ownerRows;
            }
        }
        SyncAll<true>();

        cacheWriteThrough(out + active + BaseAckStageOffset(layout_, 0U, 0U), layout_.baseAckStageBytes);

        // Keep plan counts exact for row bounds.  The exported count and
        // recv_x offsets are padded independently for the public alignment
        // contract; padding rows retain invalid CombineRoute metadata.
        for (uint32_t local = aiv; local < localExperts_; local += aivNum) {
            int32_t offset = 0;
            for (uint32_t i = 0; i < local; ++i) {
                const uint32_t count = static_cast<uint32_t>(plan_[info_.localExpertCountOffset + i]);
                const uint32_t alignment = info_.expertAlignment == 0U ? 1U : info_.expertAlignment;
                offset += static_cast<int32_t>((count + alignment - 1U) / alignment * alignment);
            }
            plan_[info_.localExpertOffset + local] = offset;
            const uint32_t count = static_cast<uint32_t>(plan_[info_.localExpertCountOffset + local]);
            const uint32_t alignment = info_.expertAlignment == 0U ? 1U : info_.expertAlignment;
            recv_[local] = static_cast<int32_t>((count + alignment - 1U) / alignment * alignment);
        }
    }

    __aicore__ inline void FanoutBaseAckSlicesToProxies()
    {
        constexpr uint64_t active = 0ULL;
        auto *out = (__gm__ uint8_t *)comm_.WindowOut(info_.epRankId);
        const uint32_t aiv = GetBlockIdx();
        const uint32_t aivNum = info_.aivNum == 0U ? 1U : info_.aivNum;
        const uint32_t proxyBegin = static_cast<uint32_t>(static_cast<uint64_t>(info_.ranksPerServer) * aiv / aivNum);
        const uint32_t proxyEnd =
            static_cast<uint32_t>(static_cast<uint64_t>(info_.ranksPerServer) * (aiv + 1U) / aivNum);
        // 当前out的内存布局已经是
        // BaseAckStage[proxy][sourceServer][localExpert][round]
        // 所以直接拷贝 [sourceServer][localExpert][round] 区域
        for (uint32_t proxyInServer = proxyBegin; proxyInServer < proxyEnd; ++proxyInServer) {
            const uint32_t proxy = sourceServer_ * info_.ranksPerServer + proxyInServer;
            auto *proxyWindow = (__gm__ uint8_t *)comm_.WindowIn(proxy);
            CopyWindowBytes(out + active + BaseAckStageOffset(layout_, proxyInServer, 0U),
                            proxyWindow + active + BaseAckFanoutOwnerOffset(layout_, info_.rankInServer),
                            layout_.baseAckStageProxyStride);
            CopyWindowFlag(
                (__gm__ uint64_t *)(proxyWindow + active + BaseAckFanoutFlagOffset(layout_, info_.rankInServer)),
                ComposeFlag(info_.epoch, Phase::SERVER_FANOUT_DONE));
        }
    }

    __aicore__ inline void WaitAllBaseAckSlices()
    {
        constexpr uint64_t active = 0ULL;
        auto *own = (__gm__ uint8_t *)comm_.WindowIn(info_.epRankId);
        const uint32_t aiv = GetBlockIdx();
        const uint32_t aivNum = info_.aivNum == 0U ? 1U : info_.aivNum;
        for (uint32_t owner = aiv; owner < info_.ranksPerServer; owner += aivNum) {
            Wait((__gm__ uint64_t *)(own + active + BaseAckFanoutFlagOffset(layout_, owner)),
                 ComposeFlag(info_.epoch, Phase::SERVER_FANOUT_DONE));
        }
    }

    __aicore__ inline void AssembleBaseAcksAtProxies()
    {
        constexpr uint64_t active = 0ULL;
        auto *in = (__gm__ uint8_t *)comm_.WindowIn(info_.epRankId);
        auto *out = (__gm__ uint8_t *)comm_.WindowOut(info_.epRankId);
        const uint32_t aiv = GetBlockIdx();
        const uint32_t aivNum = info_.aivNum == 0U ? 1U : info_.aivNum;
        for (uint32_t sourceServer = aiv; sourceServer < info_.serverNum; sourceServer += aivNum) {
            const uint32_t sourceRank = sourceServer * info_.ranksPerServer + info_.rankInServer;
            // ack[ownerInServer][localExpert][round]
            auto *ack = out + active + BaseAckLaneOffset(layout_, info_.epWorldSize, sourceRank, sourceServer_, 0);
            uint32_t serverMaxRows = 0U;
            for (uint32_t owner = 0U; owner < info_.ranksPerServer; ++owner) {
                auto *ownerRows = (__gm__ uint32_t *)(in + active + BaseAckFanoutOwnerRowsOffset(layout_, owner));
                serverMaxRows = serverMaxRows > *ownerRows ? serverMaxRows : *ownerRows;
            }
            auto *control = (__gm__ BaseAckControlMeta *)ack;
            control->serverMaxRows = serverMaxRows;
            uint64_t ingressBase = 0U;
            for (uint32_t preceding = 0U; preceding < sourceServer; ++preceding) {
                const uint32_t rank = preceding * info_.ranksPerServer + info_.rankInServer;
                auto *lane = in + ControlLaneOffset(layout_, info_.epWorldSize, rank, sourceServer_, 0U);
                cacheReadInvalidate(lane, layout_.controlStride);
                auto *counts = lane + AllRoundCountRoundOffset(0U, expertsPerServer_);
                const uint32_t packetCount = *(__gm__ uint32_t *)counts;
                auto *expertCounts = (__gm__ int32_t *)(counts + sizeof(uint32_t));
                uint32_t memberCount = 0U;
                for (uint32_t expert = 0U; expert < expertsPerServer_; ++expert) {
                    memberCount += static_cast<uint32_t>(expertCounts[expert]);
                }
                ingressBase += AlignUp64(DispatchBatchPayloadBytes(packetCount, memberCount, info_.recordStride));
            }
            auto *lane = in + ControlLaneOffset(layout_, info_.epWorldSize, sourceRank, sourceServer_, 0U);
            cacheReadInvalidate(lane, layout_.controlStride);
            auto *counts = lane + AllRoundCountRoundOffset(0U, expertsPerServer_);
            const uint32_t packetCount = *(__gm__ uint32_t *)counts;
            auto *expertCounts = (__gm__ int32_t *)(counts + sizeof(uint32_t));
            uint32_t memberCount = 0U;
            for (uint32_t expert = 0U; expert < expertsPerServer_; ++expert) {
                memberCount += static_cast<uint32_t>(expertCounts[expert]);
            }
            const uint64_t payloadBytes = DispatchBatchPayloadBytes(packetCount, memberCount, info_.recordStride);
            const auto p = MakePlanLayout(info_.bs, info_.maxBs, info_.k, info_.moeExpertNum, info_.epWorldSize,
                                          info_.serverNum, info_.ranksPerServer, 1U, info_.maxBs);
            plan_[p.recvIngressBase + 2U * sourceServer] = static_cast<int32_t>(ingressBase);
            plan_[p.recvIngressBase + 2U * sourceServer + 1U] = static_cast<int32_t>(ingressBase >> 32U);
            plan_[p.recvIngressBytes + 2U * sourceServer] = static_cast<int32_t>(payloadBytes);
            plan_[p.recvIngressBytes + 2U * sourceServer + 1U] = static_cast<int32_t>(payloadBytes >> 32U);
            control->ingressOffsetLo = static_cast<uint32_t>(ingressBase);
            control->ingressOffsetHi = static_cast<uint32_t>(ingressBase >> 32U);
            control->payloadBytesLo = static_cast<uint32_t>(payloadBytes);
            control->payloadBytesHi = static_cast<uint32_t>(payloadBytes >> 32U);
            for (uint32_t i = 0U; i < 3U; ++i) {
                control->reserved[i] = 0U;
            }
            for (uint32_t owner = 0; owner < info_.ranksPerServer; ++owner) {
                // [owner][sourceServer][localExpert][round]
                auto *slice = in + active + BaseAckFanoutSliceOffset(layout_, owner, sourceServer);
                const uint64_t dstOffset = static_cast<uint64_t>(owner) * localExperts_ * sizeof(int32_t);
                CopyWindowBytes(slice, ack + sizeof(BaseAckControlMeta) + dstOffset,
                                static_cast<uint64_t>(localExperts_) * sizeof(int32_t));
            }
            cacheWriteThrough(ack, layout_.baseAckStride);
        }
    }

    __aicore__ inline void PublishBaseAcksFromProxies()
    {
        constexpr uint64_t active = 0ULL;
        auto *out = (__gm__ uint8_t *)comm_.WindowOut(info_.epRankId);
        const uint32_t aiv = GetBlockIdx();
        const uint32_t aivNum = info_.aivNum == 0U ? 1U : info_.aivNum;
        for (uint32_t sourceServer = aiv; sourceServer < info_.serverNum; sourceServer += aivNum) {
            const uint32_t source = sourceServer * info_.ranksPerServer + info_.rankInServer;
            // ack[ownerInServer][localExpert][round]
            auto *ack = out + active + BaseAckLaneOffset(layout_, info_.epWorldSize, source, sourceServer_, 0);
            auto *sin = (__gm__ uint8_t *)comm_.WindowIn(source);
            // [ownerInServer][localExpert][round]
            auto *dst = sin + active + BaseAckLaneOffset(layout_, info_.epWorldSize, source, sourceServer_, 0);
            if (source == info_.epRankId) {
                CopyWindowBytes(ack, dst, layout_.baseAckStride);
            } else {
                comm_.PostRdmaSend(ack, dst, source, layout_.baseAckStride, *doorbell_, *head_);
            }
            auto *flag = (__gm__ uint64_t *)(out + active +
                                             BaseAckFlagOffset(layout_, info_.epWorldSize, source, sourceServer_));
            *flag = ComposeFlag(info_.epoch, Phase::CONTROL_BASE_READY);
            // PostReadyFlag reads this local WindowOut word through RDMA. Make
            // the scalar store visible before submitting that second WQE.
            cacheWriteThrough((__gm__ uint8_t *)flag, sizeof(uint64_t));
            auto *remoteFlag =
                (__gm__ uint64_t *)(sin + active +
                                    BaseAckFlagOffset(layout_, info_.epWorldSize, source, sourceServer_));
            if (source == info_.epRankId) {
                CopyWindowFlag(remoteFlag, *flag);
            } else {
                comm_.PostReadyFlag((GM_ADDR)flag, (GM_ADDR)remoteFlag, source, *doorbell_, *head_);
            }
        }
    }
    __aicore__ inline void ApplyAllReturnedRowBases()
    {
        const uint32_t aiv = GetBlockIdx();
        const uint32_t aivNum = info_.aivNum == 0U ? 1U : info_.aivNum;
        constexpr uint64_t active = 0ULL;
        auto *self = (__gm__ uint8_t *)comm_.WindowIn(info_.epRankId);
        if (aiv == 0U) {
            int32_t total = 0;
            for (uint32_t i = 0; i < localExperts_; ++i) {
                total += recv_[i];
            }
            total_[0] = total;
        }

        // All BaseAck lanes must be ready before packet work is distributed
        // independently of target server.  Each flag is polled by one AIV.
        // 等待数据收齐
        uint32_t localMaxRows = 0U;
        for (uint32_t target = aiv; target < info_.serverNum; target += aivNum) {
            auto *f = (__gm__ uint64_t *)(self + active +
                                          BaseAckFlagOffset(layout_, info_.epWorldSize, info_.epRankId, target));
            Wait(f, ComposeFlag(info_.epoch, Phase::CONTROL_BASE_READY));
            auto *ack = self + active + BaseAckLaneOffset(layout_, info_.epWorldSize, info_.epRankId, target, 0);
            // The ready flag orders the remote payload write. Invalidate the
            // complete lane because the row-base matrix is reused across
            // epochs and is consumed below after the global max reduction.
            cacheReadInvalidate(ack, layout_.baseAckStride);
            auto *control = (__gm__ BaseAckControlMeta *)ack;
            const auto p = MakePlanLayout(info_.bs, info_.maxBs, info_.k, info_.moeExpertNum, info_.epWorldSize,
                                          info_.serverNum, info_.ranksPerServer, 1U, info_.maxBs);
            plan_[p.sendIngressBase + 2U * target] = static_cast<int32_t>(control->ingressOffsetLo);
            plan_[p.sendIngressBase + 2U * target + 1U] = static_cast<int32_t>(control->ingressOffsetHi);
            plan_[p.sendIngressBytes + 2U * target] = static_cast<int32_t>(control->payloadBytesLo);
            plan_[p.sendIngressBytes + 2U * target + 1U] = static_cast<int32_t>(control->payloadBytesHi);
            const uint64_t ingress =
                (static_cast<uint64_t>(control->ingressOffsetHi) << 32U) | control->ingressOffsetLo;
            const uint64_t bytes = (static_cast<uint64_t>(control->payloadBytesHi) << 32U) | control->payloadBytesLo;
            if (ingress > layout_.dispatchIngressBytes || bytes > layout_.dispatchIngressBytes - ingress ||
                bytes > 0xffffffffU) {
                plan_[info_.errorOffset] = static_cast<int32_t>(Phase::ERROR_OVERFLOW);
            }
            localMaxRows = localMaxRows > control->serverMaxRows ? localMaxRows : control->serverMaxRows;
        }
        workspace_[RoundMaxScratch(aiv)] = static_cast<int32_t>(localMaxRows);
        cacheWriteThrough((__gm__ uint8_t *)(workspace_ + RoundMaxScratch(aiv)), sizeof(int32_t));
        SyncAll<true>();

        if (aiv == 0U) {
            uint32_t globalMaxRows = 0U;
            for (uint32_t worker = 0U; worker < aivNum; ++worker) {
                auto *value = workspace_ + RoundMaxScratch(worker);
                cacheReadInvalidate((__gm__ uint8_t *)value, sizeof(int32_t));
                globalMaxRows =
                    globalMaxRows > static_cast<uint32_t>(*value) ? globalMaxRows : static_cast<uint32_t>(*value);
            }
            auto *header = (__gm__ PlanHeader *)plan_;
            header->globalCombineRows = globalMaxRows;
            cacheWriteThrough((__gm__ uint8_t *)&header->globalCombineRows, sizeof(uint32_t));
        }

        // SourcePacketDesc is laid out as [targetServer][token]. Splitting
        // this contiguous space gives every packet and its MemberDesc range a
        // unique AIV owner, including when aivNum exceeds serverNum.
        const uint32_t packetElements = info_.serverNum * info_.slotsPerServer;
        const uint32_t begin = static_cast<uint32_t>(static_cast<uint64_t>(packetElements) * aiv / aivNum);
        const uint32_t end = static_cast<uint32_t>(static_cast<uint64_t>(packetElements) * (aiv + 1U) / aivNum);
        // 每个AIV处理一部分packet
        for (uint32_t packet = begin; packet < end; ++packet) {
            const uint32_t target = (packet / info_.slotsPerServer) % info_.serverNum;
            const uint32_t slot = packet % info_.slotsPerServer;
            const uint32_t token = slot;
            if (token >= info_.bs) {
                continue;
            }
            auto *packetDesc = (__gm__ SourcePacketDesc *)(plan_ + info_.packetDescOffset) + packet;
            const uint32_t memberCount = static_cast<uint32_t>(packetDesc->memberCount);
            const auto *ack = self + active + BaseAckLaneOffset(layout_, info_.epWorldSize, info_.epRankId, target, 0);
            const uint32_t expertBegin = target * expertsPerServer_;
            // 对当前 packet 中每个实际topk member，找到目标 server 回传的对应 expert row base,
            // 再加上该 member 在 source rank 内的局部序号，得到它在目标 expert 内的最终行号
            for (uint32_t m = 0; m < memberCount; ++m) {
                const uint32_t mb = info_.memberDescOffset + (packet * info_.k + m) * MemberElements();
                const uint32_t expert = static_cast<uint32_t>(plan_[mb]);
                const uint32_t expertInServer = expert - expertBegin;
                const uint32_t owner = expertInServer / localExperts_;
                const uint32_t local = expertInServer % localExperts_;
                auto *base =
                    (__gm__ int32_t *)(ack + AllRoundBaseOwnerExpertRoundOffset(0U, owner, local, 1U, localExperts_));
                plan_[mb + 1U] += *base;
            }
        }
    }
    __gm__ int32_t *topk_{nullptr};
    __gm__ int32_t *plan_{nullptr};
    __gm__ int32_t *recv_{nullptr};
    __gm__ int32_t *total_{nullptr};
    __gm__ int32_t *workspace_{nullptr};
    CamMoeLayeredInfo info_{};
    WindowLayout layout_{};
    CommAdapter comm_{};
    LocalTensor<uint64_t> *doorbell_{nullptr};
    LocalTensor<uint32_t> *head_{nullptr};
    LocalTensor<int32_t> *initUb_{nullptr};
    LocalTensor<int32_t> *reduceSrcUb_{nullptr};
    LocalTensor<int32_t> *reduceAccUb_{nullptr};
    LocalTensor<uint8_t> *aggregateCopyUb_{nullptr};
    uint32_t localExperts_{0};
    uint32_t expertsPerServer_{0};
    uint32_t sourceServer_{0};
    uint32_t countSlice_{0};
    uint32_t packetSlice_{0};
    uint32_t countSlicePadded_{0};
    uint32_t packetSlicePadded_{0};
    uint32_t memberCountSlicePadded_{0};
};
}  // namespace

extern "C" __global__ __aicore__ void notify_dispatch_normal_layered(GM_ADDR roceContext, GM_ADDR topkIdx,
                                                                     GM_ADDR layeredPlan, GM_ADDR recvTokensPerExpert,
                                                                     GM_ADDR expertTotalCount, GM_ADDR workspace,
                                                                     GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(NotifyDispatchNormalLayeredTilingData);
    GET_TILING_DATA_WITH_STRUCT(NotifyDispatchNormalLayeredTilingData, data, tiling);
    TPipe pipe;
#ifdef __DAV_C310__
    TBuf hcommUb;
    pipe.InitBuffer(hcommUb, kHcommUbBytes);
#endif
    TBuf rdmaUb;
    pipe.InitBuffer(rdmaUb, 32U);
    TBuf sharedWorkUb;
    pipe.InitBuffer(sharedWorkUb, kSharedWorkUbBytes);
    auto doorbell = rdmaUb.GetWithOffset<uint64_t>(1U, 0U);
    auto head = rdmaUb.GetWithOffset<uint32_t>(1U, 16U);
    // Init and aggregate copy reuse the first half after their preceding
    // MTE operations have completed. Reduction owns both halves concurrently.
    auto initTensor = sharedWorkUb.GetWithOffset<int32_t>(kInitTileElements, 0U);
    auto reduceSrcTensor = sharedWorkUb.GetWithOffset<int32_t>(kReduceTileElements, 0U);
    auto reduceAccTensor = sharedWorkUb.GetWithOffset<int32_t>(kReduceTileElements, kReduceTileBytes);
    auto aggregateCopyTensor = sharedWorkUb.GetWithOffset<uint8_t>(kAggregateCopyTileBytes, 0U);
#ifdef __DAV_C310__
    auto hcommTensor = hcommUb.Get<uint8_t>();
#else
    auto hcommTensor = aggregateCopyTensor;
#endif
    NotifyDispatchNormalLayered op;
    op.Init(roceContext, topkIdx, layeredPlan, recvTokensPerExpert, expertTotalCount, workspace, data.layered, doorbell,
            head, initTensor, reduceSrcTensor, reduceAccTensor, aggregateCopyTensor, hcommTensor);
    op.Process();
}
