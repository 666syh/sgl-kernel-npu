#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

#include "error_log.h"
#include "mc2_tiling_utils.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/hccl/hccl_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "cam_moe_layered_host_validation.h"
#include "../op_kernel/cam_moe_dispatch_normal_layered_tiling.h"
#include "../op_kernel/cam_moe_layered_plan.h"

using namespace ge;

namespace {
constexpr uint32_t ROCE_CONTEXT_INDEX = 0U;
constexpr uint32_t X_INDEX = 1U;
constexpr uint32_t WEIGHTS_INDEX = 2U;
constexpr uint32_t TOPK_INDEX = 3U;
constexpr uint32_t PLAN_INDEX = 4U;
constexpr uint32_t RECV_X_INDEX = 0U;
constexpr uint32_t ASSIST_INDEX = 1U;
constexpr uint32_t RECV_TOPK_INDEX = 2U;
constexpr uint32_t RECV_TOPK_WEIGHTS_INDEX = 3U;
constexpr uint32_t ATTR_EP_GROUP = 0U;
constexpr uint32_t ATTR_EP_WORLD = 1U;
constexpr uint32_t ATTR_EP_RANK = 2U;
constexpr uint32_t ATTR_TP_GROUP = 3U;
constexpr uint32_t ATTR_EXPERTS = 6U;
constexpr uint32_t ATTR_RANKS_PER_SERVER = 7U;
constexpr uint32_t ATTR_SLOTS = 8U;
constexpr uint32_t ATTR_IPC_TILE = 9U;
constexpr uint32_t ATTR_MAX_BS = 10U;
constexpr uint32_t MAX_SERVERS = 32U;
constexpr uint32_t MAX_TOPK = 16U;
constexpr uint32_t MAX_HIDDEN = 8192U;
constexpr uint64_t WORKSPACE_BYTES = 16ULL * 1024ULL * 1024ULL;
#ifdef __DAV_C310__
constexpr uint64_t RDMA_UB_BYTES = 32ULL;
constexpr uint64_t HCOMM_UB_BYTES = 512ULL;
#else
constexpr uint64_t RDMA_UB_BYTES = 32ULL;
constexpr uint64_t HCOMM_UB_BYTES = 0ULL;
#endif
constexpr uint64_t PEER_COPY_UB_BYTES = 16ULL * 1024ULL;

int64_t Attr(const gert::TilingContext *context, uint32_t index)
{
    auto *attrs = context->GetAttrs();
    auto *value = attrs == nullptr ? nullptr : attrs->GetAttrPointer<int64_t>(static_cast<int>(index));
    return value == nullptr ? -1 : *value;
}

bool Shape2D(const gert::StorageShape *shape, int64_t &d0, int64_t &d1)
{
    if (shape == nullptr || shape->GetStorageShape().GetDimNum() != 2U) return false;
    d0 = shape->GetStorageShape().GetDim(0);
    d1 = shape->GetStorageShape().GetDim(1);
    return d0 >= 0 && d1 > 0;
}

void ConfigureMc2(CamMoeDispatchNormalLayeredTilingData *data, const std::string &epGroup, const std::string &tpGroup)
{
    AscendC::Mc2CcTilingConfig config(epGroup, 8U, "AlltoAll=level0:fullmesh;level1:pairwise");
    config.SetCommEngine(mc2tiling::AIV_ENGINE);
    config.GetTiling(data->normal.mc2InitTiling);
    config.GetTiling(data->normal.mc2CcTiling1);
    config.SetGroupName(tpGroup);
    config.SetOpType(6U);
    config.SetAlgConfig("AllGather=level0:ring");
    config.GetTiling(data->normal.mc2CcTiling2);
}
}  // namespace

namespace optiling {
static ge::graphStatus CamMoeDispatchNormalLayeredTiling(gert::TilingContext *context)
{
    const char *node = context->GetNodeName();
    auto *data = context->GetTilingData<CamMoeDispatchNormalLayeredTilingData>();
    auto *roceContextDesc = context->GetInputDesc(ROCE_CONTEXT_INDEX);
    auto *xDesc = context->GetInputDesc(X_INDEX);
    auto *weightsDesc = context->GetInputDesc(WEIGHTS_INDEX);
    auto *topkDesc = context->GetInputDesc(TOPK_INDEX);
    auto *planDesc = context->GetInputDesc(PLAN_INDEX);
    auto *assistDesc = context->GetOutputDesc(ASSIST_INDEX);
    auto *recvTopkDesc = context->GetOutputDesc(RECV_TOPK_INDEX);
    auto *recvTopkWeightsDesc = context->GetOutputDesc(RECV_TOPK_WEIGHTS_INDEX);
    int64_t bs = 0, hidden = 0, weightBs = 0, topk = 0, recvRows = 0, recvHidden = 0;
    int64_t assistRows = 0, assistWidth = 0, recvTopkRows = 0, recvTopkWidth = 0, recvWeightRows = 0,
            recvWeightWidth = 0, topkBs = 0, topkWidth = 0;
    OP_TILING_CHECK(
        data == nullptr || roceContextDesc == nullptr || xDesc == nullptr || weightsDesc == nullptr ||
            topkDesc == nullptr || planDesc == nullptr || assistDesc == nullptr || recvTopkDesc == nullptr ||
            recvTopkWeightsDesc == nullptr || (xDesc->GetDataType() != DT_BF16 && xDesc->GetDataType() != DT_FLOAT16) ||
            roceContextDesc->GetDataType() != DT_INT32 || weightsDesc->GetDataType() != DT_FLOAT ||
            topkDesc->GetDataType() != DT_INT32 || planDesc->GetDataType() != DT_INT32 ||
            assistDesc->GetDataType() != DT_INT32 || recvTopkDesc->GetDataType() != DT_INT32 ||
            recvTopkWeightsDesc->GetDataType() != DT_FLOAT || !Shape2D(context->GetInputShape(X_INDEX), bs, hidden) ||
            !Shape2D(context->GetInputShape(WEIGHTS_INDEX), weightBs, topk) ||
            !Shape2D(context->GetInputShape(TOPK_INDEX), topkBs, topkWidth) || weightBs != bs || topkBs != bs ||
            topkWidth != topk || !Shape2D(context->GetOutputShape(RECV_X_INDEX), recvRows, recvHidden) ||
            recvHidden != hidden || !Shape2D(context->GetOutputShape(ASSIST_INDEX), assistRows, assistWidth) ||
            assistRows != recvRows || assistWidth != static_cast<int64_t>(CamMoeLayered::ASSIST_INFO_STRIDE) ||
            !Shape2D(context->GetOutputShape(RECV_TOPK_INDEX), recvTopkRows, recvTopkWidth) ||
            !Shape2D(context->GetOutputShape(RECV_TOPK_WEIGHTS_INDEX), recvWeightRows, recvWeightWidth) ||
            recvTopkRows != recvRows || recvWeightRows != recvRows || recvTopkWidth != topk || recvWeightWidth != topk,
        OP_LOGE(node, "invalid normal-layered dispatch tensor shapes or types"), return GRAPH_FAILED);
    const auto *planShape = context->GetInputShape(PLAN_INDEX);
    OP_TILING_CHECK(planShape == nullptr || planShape->GetStorageShape().GetDimNum() != 1U ||
                        planShape->GetStorageShape().GetDim(0) < static_cast<int64_t>(CamMoeLayered::HeaderElements()),
                    OP_LOGE(node, "layered_plan must contain PlanHeader"), return GRAPH_FAILED);

    const int64_t world = Attr(context, ATTR_EP_WORLD);
    const int64_t rank = Attr(context, ATTR_EP_RANK);
    const int64_t experts = Attr(context, ATTR_EXPERTS);
    const int64_t ranksPerServer = Attr(context, ATTR_RANKS_PER_SERVER);
    const int64_t slots = Attr(context, ATTR_SLOTS);
    const int64_t ipcTile = Attr(context, ATTR_IPC_TILE);
    const int64_t maxBs = Attr(context, ATTR_MAX_BS);
    OP_TILING_CHECK(world <= 0 || rank < 0 || rank >= world || experts <= 0 || ranksPerServer <= 0 ||
                        world % ranksPerServer != 0 || experts % world != 0 ||
                        experts % (world / ranksPerServer) != 0 || slots <= 0 || maxBs <= 0 || slots != maxBs ||
                        bs <= 0 || bs > maxBs || ipcTile <= 0 || topk <= 0 || topk > MAX_TOPK || hidden > MAX_HIDDEN ||
                        recvRows <= 0 || static_cast<uint64_t>(world / ranksPerServer) > MAX_SERVERS ||
                        static_cast<uint64_t>(world) > std::numeric_limits<uint32_t>::max() ||
                        static_cast<uint64_t>(rank) > std::numeric_limits<uint32_t>::max() ||
                        static_cast<uint64_t>(experts) > std::numeric_limits<uint32_t>::max() ||
                        static_cast<uint64_t>(ranksPerServer) > std::numeric_limits<uint32_t>::max() ||
                        static_cast<uint64_t>(slots) > std::numeric_limits<uint32_t>::max() ||
                        static_cast<uint64_t>(ipcTile) > std::numeric_limits<uint32_t>::max() ||
                        static_cast<uint64_t>(maxBs) > std::numeric_limits<uint32_t>::max() ||
                        static_cast<uint64_t>(hidden) > std::numeric_limits<uint32_t>::max(),
                    OP_LOGE(node, "invalid normal-layered dispatch topology or attrs"), return GRAPH_FAILED);

    auto &l = data->layered;
    l.epWorldSize = static_cast<uint32_t>(world);
    l.epRankId = static_cast<uint32_t>(rank);
    l.ranksPerServer = static_cast<uint32_t>(ranksPerServer);
    l.serverNum = static_cast<uint32_t>(world / ranksPerServer);
    l.rankInServer = static_cast<uint32_t>(rank % ranksPerServer);
    l.moeExpertNum = static_cast<uint32_t>(experts);
    l.bs = static_cast<uint32_t>(bs);
    l.realMaxBs = static_cast<uint32_t>(maxBs);
    l.maxBs = static_cast<uint32_t>(maxBs);
    l.k = static_cast<uint32_t>(topk);
    l.h = static_cast<uint32_t>(hidden);
    l.elementBytes = 2U;
    l.isHybridDeployment = Mc2TilingUtils::IsHybridDeployment() ? 1U : 0U;
    l.slotsPerServer = static_cast<uint32_t>(slots);
    l.ipcTilePackets = static_cast<uint32_t>(ipcTile);
    l.validRounds = 1U;
    l.physicalRounds = 1U;
    l.recordStride = CamMoeLayered::DispatchRecordStride(l.h, l.k, l.elementBytes);
    l.dispatchChunkPackets = CamMoeLayered::DispatchChunkPackets(l.maxBs, l.recordStride);
    l.dispatchMaxChunks = CamMoeLayered::DispatchMaxChunks(l.maxBs, l.recordStride);
    OP_TILING_CHECK(l.dispatchChunkPackets == 0U || l.dispatchMaxChunks >= 0x1fffffffU,
                    OP_LOGE(node, "normal-layered dispatch chunk count exceeds scheduler state"), return GRAPH_FAILED);
    l.partialStride = CamMoeLayered::PartialRecordStride(l.h, l.elementBytes);
    OP_TILING_CHECK(!CamMoeLayeredHost::ValidWindowInputArithmetic(l.moeExpertNum, l.epWorldSize, l.ranksPerServer,
                                                                   l.serverNum, l.physicalRounds, l.slotsPerServer, l.k,
                                                                   l.recordStride, l.partialStride),
                    OP_LOGE(node, "normal-layered dispatch window arithmetic overflows"), return GRAPH_FAILED);
    const uint64_t controlStride = CamMoeLayered::AlignUp64(
        CamMoeLayered::AllRoundCountPayloadBytes(l.physicalRounds, l.moeExpertNum / l.serverNum),
        CamMoeLayered::CACHE_LINE_ISOLATION_BYTES);
    OP_TILING_CHECK(controlStride > std::numeric_limits<uint32_t>::max(),
                    OP_LOGE(node, "normal-layered dispatch control stride overflow"), return GRAPH_FAILED);
    l.controlStride = static_cast<uint32_t>(controlStride);
    const auto window = CamMoeLayered::MakeWindowLayout(l.moeExpertNum, l.epWorldSize, l.ranksPerServer, l.serverNum,
                                                        l.physicalRounds, l.slotsPerServer, l.recordStride,
                                                        l.partialStride, l.k, l.isHybridDeployment != 0U);
    l.totalWinSize = Mc2TilingUtils::GetMaxWindowSize();
    const uint64_t maxProxyRank =
        (static_cast<uint64_t>(l.serverNum) - 1U) * l.ranksPerServer + (l.ranksPerServer - 1U);
    OP_TILING_CHECK(l.validRounds > l.physicalRounds || maxProxyRank >= l.epWorldSize ||
                        !CamMoeLayeredHost::ValidWindowLayout(window, l.epWorldSize, l.serverNum, l.ranksPerServer,
                                                              l.physicalRounds) ||
                        !CamMoeLayeredHost::ValidWindowCapacity(window, l.totalWinSize),
                    OP_LOGE(node, "normal-layered dispatch window layout or capacity is invalid"), return GRAPH_FAILED);
    const auto planLayout =
        CamMoeLayered::MakePlanLayout(l.bs, l.maxBs, l.k, l.moeExpertNum, l.epWorldSize, l.serverNum, l.ranksPerServer,
                                      l.physicalRounds, l.slotsPerServer);
    uint64_t planPacketCount = CamMoeLayered::HeaderElements(), planDispatchMemberCount = 0U, planPacketSlot = 0U;
    uint64_t planExpertCount = 0U, planLocalExpertOffset = 0U, planLocalExpertCount = 0U;
    uint64_t planPacketDesc = 0U, planMemberDesc = 0U, planSourceReturnBase = 0U, planReverseCount = 0U;
    uint64_t planReverseReturnBase = 0U, planReversePacket = 0U, planReverseMember = 0U, planOverflow = 0U;
    uint64_t planTotal = 0U, packetSlots = 0U, reversePacketSlots = 0U, routeCounts = 0U;
    uint64_t packetDescElements = 0U, memberDescElements = 0U, reversePacketElements = 0U;
    uint64_t reverseMemberElements = 0U, planIngressElements = 0U;
    const uint64_t servers = l.serverNum;
    const uint64_t expertsPerServer = l.moeExpertNum / l.serverNum;
    const uint64_t localExperts = l.moeExpertNum / l.epWorldSize;
    const uint64_t maxUint32 = std::numeric_limits<uint32_t>::max();
    const bool planArithmetic =
        CamMoeLayeredHost::AddFits(planPacketCount, servers, planDispatchMemberCount) &&
        CamMoeLayeredHost::AddFits(planDispatchMemberCount, servers, planPacketSlot) &&
        CamMoeLayeredHost::MulFits(l.maxBs, servers, packetSlots) &&
        CamMoeLayeredHost::AddFits(planPacketSlot, packetSlots, planExpertCount) &&
        CamMoeLayeredHost::MulFits(servers, expertsPerServer, routeCounts) &&
        CamMoeLayeredHost::AddFits(planExpertCount, routeCounts, planLocalExpertOffset) &&
        CamMoeLayeredHost::AddFits(planLocalExpertOffset, localExperts, planLocalExpertCount) &&
        CamMoeLayeredHost::AddFits(planLocalExpertCount, localExperts, planPacketDesc) &&
        CamMoeLayeredHost::MulFits(servers, l.maxBs, packetDescElements) &&
        CamMoeLayeredHost::MulFits(packetDescElements, CamMoeLayered::PacketElements(), packetDescElements) &&
        CamMoeLayeredHost::AddFits(planPacketDesc, packetDescElements, planMemberDesc) &&
        CamMoeLayeredHost::MulFits(servers, l.maxBs, memberDescElements) &&
        CamMoeLayeredHost::MulFits(memberDescElements, l.k, memberDescElements) &&
        CamMoeLayeredHost::MulFits(memberDescElements, CamMoeLayered::MemberElements(), memberDescElements) &&
        CamMoeLayeredHost::AddFits(planMemberDesc, memberDescElements, planSourceReturnBase) &&
        CamMoeLayeredHost::MulFits(servers, 13U, planIngressElements) &&
        CamMoeLayeredHost::AddFits(planIngressElements, 2U, planIngressElements) &&
        CamMoeLayeredHost::AddFits(planSourceReturnBase, planIngressElements, planReverseCount) &&
        CamMoeLayeredHost::AddFits(planReverseCount, servers, planReverseReturnBase) &&
        CamMoeLayeredHost::AddFits(planReverseReturnBase, servers, planReversePacket) &&
        CamMoeLayeredHost::MulFits(l.maxBs, l.epWorldSize, reversePacketSlots) &&
        CamMoeLayeredHost::MulFits(reversePacketSlots, CamMoeLayered::ReversePacketElements(), reversePacketElements) &&
        CamMoeLayeredHost::AddFits(planReversePacket, reversePacketElements, planReverseMember) &&
        CamMoeLayeredHost::MulFits(reversePacketSlots, l.k, reverseMemberElements) &&
        CamMoeLayeredHost::MulFits(reverseMemberElements, CamMoeLayered::ReverseMemberElements(),
                                   reverseMemberElements) &&
        CamMoeLayeredHost::AddFits(planReverseMember, reverseMemberElements, planOverflow) &&
        CamMoeLayeredHost::AddFits(planOverflow, 2U, planTotal) && planTotal <= maxUint32 &&
        planPacketCount == planLayout.packetCount && planDispatchMemberCount == planLayout.dispatchMemberCount &&
        planPacketSlot == planLayout.packetSlot && planExpertCount == planLayout.expertCount &&
        planLocalExpertOffset == planLayout.localExpertOffset && planLocalExpertCount == planLayout.localExpertCount &&
        planPacketDesc == planLayout.packetDesc && planMemberDesc == planLayout.memberDesc &&
        planSourceReturnBase == planLayout.sourceReturnBase &&
        planLayout.outboundBase == planSourceReturnBase + servers &&
        planLayout.outboundBytes == planSourceReturnBase + 3U * servers &&
        planLayout.outboundTotalBytes == planSourceReturnBase + 5U * servers &&
        planLayout.sendIngressBase == planSourceReturnBase + 5U * servers + 2U &&
        planLayout.sendIngressBytes == planSourceReturnBase + 7U * servers + 2U &&
        planLayout.recvIngressBase == planSourceReturnBase + 9U * servers + 2U &&
        planLayout.recvIngressBytes == planSourceReturnBase + 11U * servers + 2U &&
        planReverseCount == planLayout.reverseCount && planReverseReturnBase == planLayout.reverseReturnBase &&
        planReversePacket == planLayout.reversePacket && planReverseMember == planLayout.reverseMember &&
        planOverflow == planLayout.overflow && planTotal == planLayout.total;
    OP_TILING_CHECK(!planArithmetic, OP_LOGE(node, "normal-layered plan layout arithmetic overflows"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(planShape->GetStorageShape().GetDim(0) < static_cast<int64_t>(planLayout.total),
                    OP_LOGE(node, "layered_plan capacity is smaller than the required plan layout"),
                    return GRAPH_FAILED);
    l.planElements = planLayout.total;
    l.errorOffset = planLayout.error;
    l.activeWindowBytes = window.requiredBytes;
    l.windowRequiredBytes = window.requiredBytes;

    data->normal.camMoeDispatchNormalInfo.epWorldSize = l.epWorldSize;
    data->normal.camMoeDispatchNormalInfo.epRankId = l.epRankId;
    data->normal.camMoeDispatchNormalInfo.moeExpertNum = l.moeExpertNum;
    data->normal.camMoeDispatchNormalInfo.bs = l.bs;
    data->normal.camMoeDispatchNormalInfo.k = l.k;
    data->normal.camMoeDispatchNormalInfo.h = l.h;
    data->normal.camMoeDispatchNormalInfo.totalWinSize = l.totalWinSize;
    auto *workspace = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspace == nullptr, OP_LOGE(node, "workspace allocation unavailable"), return GRAPH_FAILED);
    workspace[0] = WORKSPACE_BYTES;
    auto *attrs = context->GetAttrs();
    auto *epGroup = attrs == nullptr ? nullptr : attrs->GetAttrPointer<char>(ATTR_EP_GROUP);
    auto *tpGroup = attrs == nullptr ? nullptr : attrs->GetAttrPointer<char>(ATTR_TP_GROUP);
    OP_TILING_CHECK(epGroup == nullptr || tpGroup == nullptr, OP_LOGE(node, "communication group missing"),
                    return GRAPH_FAILED);
    ConfigureMc2(data, std::string(epGroup), std::string(tpGroup));
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t aiv = platform.GetCoreNumAiv();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, l.totalUbSize);
    OP_TILING_CHECK(aiv == 0U || aiv > std::numeric_limits<uint32_t>::max() ||
                        l.totalUbSize < RDMA_UB_BYTES + HCOMM_UB_BYTES + PEER_COPY_UB_BYTES,
                    OP_LOGE(node, "normal-layered dispatch UB budget is invalid"), return GRAPH_FAILED);
    l.aivNum = static_cast<uint32_t>(aiv);
    uint64_t scratchElements = 0U, scratchBytes = 0U;
    OP_TILING_CHECK(
        !CamMoeLayeredHost::MulFits(l.serverNum, static_cast<uint64_t>(l.ranksPerServer) + 2U, scratchElements) ||
            !CamMoeLayeredHost::AddFits(scratchElements, 15U, scratchElements) ||
            !CamMoeLayeredHost::MulFits(scratchElements / 16U, 64U, scratchBytes) ||
            !CamMoeLayeredHost::MulFits(scratchBytes, aiv, scratchBytes) || scratchBytes == 0U,
        OP_LOGE(node, "normal-layered dispatch scheduler workspace overflows"), return GRAPH_FAILED);
    uint64_t alignedScratchBytes = 0U;
    OP_TILING_CHECK(!CamMoeLayeredHost::Align32Fits(scratchBytes, alignedScratchBytes),
                    OP_LOGE(node, "normal-layered dispatch scheduler workspace alignment overflows"),
                    return GRAPH_FAILED);
    workspace[0] = std::max<uint64_t>(WORKSPACE_BYTES, alignedScratchBytes);
    data->normal.camMoeDispatchNormalInfo.aivNum = aiv;
    context->SetBlockDim(platform.CalcTschBlockDim(aiv, 0, aiv));
    context->SetScheduleMode(1);
    context->SetTilingKey(13000U);
    return GRAPH_SUCCESS;
}

struct CamMoeDispatchNormalLayeredCompileInfo {};
static ge::graphStatus ParseCamMoeDispatchNormalLayered(gert::TilingParseContext *)
{
    return GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(CamMoeDispatchNormalLayered)
    .Tiling(CamMoeDispatchNormalLayeredTiling)
    .TilingParse<CamMoeDispatchNormalLayeredCompileInfo>(ParseCamMoeDispatchNormalLayered);
}  // namespace optiling
