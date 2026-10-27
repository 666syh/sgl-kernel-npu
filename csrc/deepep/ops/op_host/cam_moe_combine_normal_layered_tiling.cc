#include <algorithm>
#include <cstddef>
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
#include "../op_kernel/cam_moe_combine_normal_layered_tiling.h"
#include "../op_kernel/cam_moe_layered_plan.h"

using namespace ge;

namespace {
constexpr uint32_t ROCE_CONTEXT_INDEX = 0U;
constexpr uint32_t RECV_X_INDEX = 1U;
constexpr uint32_t PLAN_INDEX = 2U;
constexpr uint32_t ASSIST_INDEX = 3U;
constexpr uint32_t OUTPUT_X_INDEX = 0U;
constexpr uint32_t ATTR_EP_GROUP = 0U;
constexpr uint32_t ATTR_EP_WORLD = 1U;
constexpr uint32_t ATTR_EP_RANK = 2U;
constexpr uint32_t ATTR_TP_GROUP = 3U;
constexpr uint32_t ATTR_TP_WORLD = 4U;
constexpr uint32_t ATTR_TP_RANK = 5U;
constexpr uint32_t ATTR_EXPERTS = 6U;
constexpr uint32_t ATTR_RANKS_PER_SERVER = 10U;
constexpr uint32_t ATTR_SLOTS = 11U;
constexpr uint32_t ATTR_IPC_TILE = 12U;
constexpr uint32_t ATTR_TOPK = 13U;
constexpr uint32_t ATTR_MAX_BS = 14U;
constexpr uint32_t ATTR_OWNER_CAPACITY = 15U;
constexpr uint32_t ATTR_INVOCATION_ID = 16U;
constexpr uint32_t ATTR_WINDOW_CAPACITY = 17U;
constexpr uint32_t MAX_SERVERS = 32U;
constexpr uint32_t MAX_RANKS_PER_SERVER = 8U;
constexpr uint32_t MAX_TOPK = 16U;
constexpr uint32_t MAX_HIDDEN = 8192U;
constexpr uint64_t WORKSPACE_MIN_BYTES = 16ULL * 1024ULL * 1024ULL;
constexpr uint64_t PARTIAL_CHUNK_BYTES = 128ULL * 1024ULL;
constexpr uint64_t PROGRESS_STRIDE_BYTES = CamMoeLayered::COMBINE_FLAG_STRIDE;
#ifdef __DAV_C310__
constexpr uint64_t RDMA_UB_BYTES = 32ULL;
constexpr uint64_t HCOMM_UB_BYTES = 512ULL;
#else
constexpr uint64_t RDMA_UB_BYTES = 32ULL;
constexpr uint64_t HCOMM_UB_BYTES = 0ULL;
#endif
constexpr uint64_t PEER_COPY_UB_BYTES = 4096ULL;

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

void ConfigureMc2(CamMoeCombineNormalLayeredTilingData *data, const std::string &epGroup, const std::string &tpGroup)
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
static ge::graphStatus CamMoeCombineNormalLayeredTiling(gert::TilingContext *context)
{
    const char *node = context->GetNodeName();
    auto *data = context->GetTilingData<CamMoeCombineNormalLayeredTilingData>();
    auto *roceContextDesc = context->GetInputDesc(ROCE_CONTEXT_INDEX);
    auto *recvDesc = context->GetInputDesc(RECV_X_INDEX);
    auto *planDesc = context->GetInputDesc(PLAN_INDEX);
    auto *assistDesc = context->GetInputDesc(ASSIST_INDEX);
    int64_t rows = 0, hidden = 0, assistRows = 0, assistWidth = 0, outRows = 0, outHidden = 0;
    OP_TILING_CHECK(
        data == nullptr || roceContextDesc == nullptr || recvDesc == nullptr || planDesc == nullptr ||
            assistDesc == nullptr || (recvDesc->GetDataType() != DT_BF16 && recvDesc->GetDataType() != DT_FLOAT16) ||
            roceContextDesc->GetDataType() != DT_INT32 || planDesc->GetDataType() != DT_INT32 ||
            assistDesc->GetDataType() != DT_INT32 || !Shape2D(context->GetInputShape(RECV_X_INDEX), rows, hidden) ||
            !Shape2D(context->GetInputShape(ASSIST_INDEX), assistRows, assistWidth) || assistRows != rows ||
            assistWidth != static_cast<int64_t>(CamMoeLayered::ASSIST_INFO_STRIDE) ||
            !Shape2D(context->GetOutputShape(OUTPUT_X_INDEX), outRows, outHidden) || outHidden != hidden,
        OP_LOGE(node, "invalid normal-layered combine tensor shapes or types"), return GRAPH_FAILED);
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
    const int64_t topk = Attr(context, ATTR_TOPK);
    const int64_t maxBs = Attr(context, ATTR_MAX_BS);
    const int64_t ownerCapacity = Attr(context, ATTR_OWNER_CAPACITY);
    const int64_t invocationId = Attr(context, ATTR_INVOCATION_ID);
    const int64_t windowCapacity = Attr(context, ATTR_WINDOW_CAPACITY);
#ifdef __DAV_C310__
    OP_TILING_CHECK(windowCapacity <= 0, OP_LOGE(node, "normal-layered combine requires registered window capacity"),
                    return GRAPH_FAILED);
#endif
    uint64_t maxLocalRows = 0U;
    const bool rowBoundFits =
        world > 0 && maxBs > 0 && topk > 0 &&
        CamMoeLayeredHost::MulFits(static_cast<uint64_t>(world), static_cast<uint64_t>(maxBs), maxLocalRows) &&
        CamMoeLayeredHost::MulFits(maxLocalRows, static_cast<uint64_t>(topk), maxLocalRows) &&
        maxLocalRows <= std::numeric_limits<uint32_t>::max();
    OP_TILING_CHECK(
        world <= 0 || rank < 0 || rank >= world || experts <= 0 || ranksPerServer <= 0 || world % ranksPerServer != 0 ||
            experts % world != 0 || experts % (world / ranksPerServer) != 0 || slots <= 0 || maxBs <= 0 ||
            ownerCapacity <= 0 || static_cast<uint64_t>(ownerCapacity) > maxLocalRows || invocationId <= 0 ||
            windowCapacity < 0 || outRows > maxBs || ipcTile <= 0 || topk <= 0 || topk > MAX_TOPK ||
            hidden > MAX_HIDDEN || outRows <= 0 || static_cast<uint64_t>(world / ranksPerServer) > MAX_SERVERS ||
            static_cast<uint64_t>(ranksPerServer) > MAX_RANKS_PER_SERVER ||
            static_cast<uint64_t>(world) > std::numeric_limits<uint32_t>::max() ||
            static_cast<uint64_t>(rank) > std::numeric_limits<uint32_t>::max() ||
            static_cast<uint64_t>(experts) > std::numeric_limits<uint32_t>::max() ||
            static_cast<uint64_t>(ranksPerServer) > std::numeric_limits<uint32_t>::max() ||
            static_cast<uint64_t>(slots) > std::numeric_limits<uint32_t>::max() ||
            static_cast<uint64_t>(ipcTile) > std::numeric_limits<uint32_t>::max() ||
            static_cast<uint64_t>(maxBs) > std::numeric_limits<uint32_t>::max() ||
            static_cast<uint64_t>(rows) > std::numeric_limits<uint32_t>::max() ||
            static_cast<uint64_t>(hidden) > std::numeric_limits<uint32_t>::max() || !rowBoundFits,
        OP_LOGE(node, "invalid normal-layered combine topology or attrs"), return GRAPH_FAILED);

    auto &l = data->layered;
    l.epWorldSize = static_cast<uint32_t>(world);
    l.epoch = static_cast<uint64_t>(invocationId);
    l.epRankId = static_cast<uint32_t>(rank);
    l.ranksPerServer = static_cast<uint32_t>(ranksPerServer);
    l.serverNum = static_cast<uint32_t>(world / ranksPerServer);
    l.rankInServer = static_cast<uint32_t>(rank % ranksPerServer);
    l.moeExpertNum = static_cast<uint32_t>(experts);
    l.bs = static_cast<uint32_t>(outRows);
    l.realMaxBs = static_cast<uint32_t>(maxBs);
    l.maxBs = static_cast<uint32_t>(maxBs);
    l.k = static_cast<uint32_t>(topk);
    l.h = static_cast<uint32_t>(hidden);
    l.elementBytes = 2U;
    l.isHybridDeployment = Mc2TilingUtils::IsHybridDeployment() ? 1U : 0U;
    l.slotsPerServer = static_cast<uint32_t>(maxBs);
    l.ipcTilePackets = static_cast<uint32_t>(ipcTile);
    l.validRounds = 1U;
    l.physicalRounds = 1U;
    l.recordStride = CamMoeLayered::DispatchRecordStride(l.h, l.k, l.elementBytes);
    l.partialStride = CamMoeLayered::PartialRecordStride(l.h, l.elementBytes);
    OP_TILING_CHECK(!CamMoeLayeredHost::ValidWindowInputArithmetic(l.moeExpertNum, l.epWorldSize, l.ranksPerServer,
                                                                   l.serverNum, l.physicalRounds, l.slotsPerServer, l.k,
                                                                   l.recordStride, l.partialStride),
                    OP_LOGE(node, "normal-layered combine window arithmetic overflows"), return GRAPH_FAILED);
    const uint64_t controlStride = CamMoeLayered::AlignUp64(
        CamMoeLayered::AllRoundCountPayloadBytes(l.physicalRounds, l.moeExpertNum / l.serverNum),
        CamMoeLayered::CACHE_LINE_ISOLATION_BYTES);
    OP_TILING_CHECK(controlStride > std::numeric_limits<uint32_t>::max(),
                    OP_LOGE(node, "normal-layered combine control stride overflow"), return GRAPH_FAILED);
    l.controlStride = static_cast<uint32_t>(controlStride);
    const auto window = CamMoeLayered::MakeWindowLayout(l.moeExpertNum, l.epWorldSize, l.ranksPerServer, l.serverNum,
                                                        l.physicalRounds, l.slotsPerServer, l.recordStride,
                                                        l.partialStride, l.k, l.isHybridDeployment != 0U);
    l.totalWinSize = Mc2TilingUtils::GetMaxWindowSize();
    if (windowCapacity > 0) {
        l.totalWinSize = std::min(l.totalWinSize, static_cast<uint64_t>(windowCapacity));
    }
    const uint64_t maxProxyRank =
        (static_cast<uint64_t>(l.serverNum) - 1U) * l.ranksPerServer + (l.ranksPerServer - 1U);
    OP_TILING_CHECK(l.validRounds > l.physicalRounds || maxProxyRank >= l.epWorldSize ||
                        !CamMoeLayeredHost::ValidWindowLayout(window, l.epWorldSize, l.serverNum, l.ranksPerServer,
                                                              l.physicalRounds),
                    OP_LOGE(node, "normal-layered combine window layout or capacity is invalid"), return GRAPH_FAILED);
    const auto planLayout =
        CamMoeLayered::MakePlanLayout(l.bs, l.maxBs, l.k, l.moeExpertNum, l.epWorldSize, l.serverNum, l.ranksPerServer,
                                      l.physicalRounds, l.slotsPerServer);
    OP_TILING_CHECK(planShape->GetStorageShape().GetDim(0) < static_cast<int64_t>(planLayout.total),
                    OP_LOGE(node, "layered_plan capacity is smaller than the required plan layout"),
                    return GRAPH_FAILED);
    l.activeWindowBytes = window.requiredBytes;
    l.windowRequiredBytes = window.requiredBytes;

    OP_TILING_CHECK(ownerCapacity <= 0 || static_cast<uint64_t>(ownerCapacity) > maxLocalRows ||
                        static_cast<uint64_t>(ownerCapacity) > std::numeric_limits<uint32_t>::max(),
                    OP_LOGE(node, "normal-layered owner capacity is invalid"), return GRAPH_FAILED);
    data->ownerCapacity = static_cast<uint32_t>(ownerCapacity);
    uint64_t checkedCombineBytes = 0U;
    OP_TILING_CHECK(!CamMoeLayeredHost::ValidCombineWindowArithmetic(
                        window.partialOffset, l.serverNum, l.ranksPerServer, l.moeExpertNum / l.epWorldSize,
                        data->ownerCapacity, l.maxBs, l.h, l.elementBytes, checkedCombineBytes),
                    OP_LOGE(node, "normal-layered combine address arithmetic overflows"), return GRAPH_FAILED);
    const auto combineWindow =
        CamMoeLayered::MakeCombineWindowLayout(window, l.serverNum, l.ranksPerServer, l.moeExpertNum / l.epWorldSize,
                                               data->ownerCapacity, l.maxBs, l.h, l.elementBytes);
    OP_TILING_CHECK(
        combineWindow.requiredBytes != checkedCombineBytes || combineWindow.requiredBytes < window.partialOffset ||
            combineWindow.requiredBytes > l.totalWinSize ||
            combineWindow.ownerStageOffset % CamMoeLayered::ALIGN_BYTES != 0U ||
            combineWindow.expertBaseOffset % CamMoeLayered::ALIGN_BYTES != 0U ||
            combineWindow.expertBaseStride % CamMoeLayered::ALIGN_BYTES != 0U ||
            combineWindow.expertBaseBytes != static_cast<uint64_t>(l.ranksPerServer) * combineWindow.expertBaseStride ||
            combineWindow.expertBaseOffset + combineWindow.expertBaseBytes > combineWindow.proxyPartialOffset ||
            combineWindow.proxyPartialOffset % CamMoeLayered::ALIGN_BYTES != 0U ||
            combineWindow.sourcePartialOffset % CamMoeLayered::ALIGN_BYTES != 0U ||
            combineWindow.partialStride % CamMoeLayered::ALIGN_BYTES != 0U ||
            combineWindow.ownerReadyOffset % CamMoeLayered::COMBINE_FLAG_STRIDE != 0U ||
            combineWindow.ownerConsumedOffset % CamMoeLayered::COMBINE_FLAG_STRIDE != 0U ||
            combineWindow.serverReadyOffset % CamMoeLayered::COMBINE_FLAG_STRIDE != 0U ||
            combineWindow.sourceReadyOffset % CamMoeLayered::COMBINE_FLAG_STRIDE != 0U ||
            combineWindow.sourceConsumedOffset % CamMoeLayered::COMBINE_FLAG_STRIDE != 0U ||
            combineWindow.serverFlagStageOffset % CamMoeLayered::COMBINE_FLAG_STRIDE != 0U ||
            combineWindow.sourceFlagStageOffset % CamMoeLayered::COMBINE_FLAG_STRIDE != 0U ||
            combineWindow.consumedFlagStageOffset % CamMoeLayered::COMBINE_FLAG_STRIDE != 0U,
        OP_LOGE(node, "normal-layered combine window exceeds capacity or alignment"), return GRAPH_FAILED);
    l.activeWindowBytes = combineWindow.requiredBytes;
    l.windowRequiredBytes = combineWindow.requiredBytes;

    data->normal.camMoeCombineNormalInfo.epWorldSize = l.epWorldSize;
    data->normal.camMoeCombineNormalInfo.epRankId = l.epRankId;
    data->normal.camMoeCombineNormalInfo.moeExpertNum = l.moeExpertNum;
    data->normal.camMoeCombineNormalInfo.bs = l.bs;
    data->normal.camMoeCombineNormalInfo.k = l.k;
    data->normal.camMoeCombineNormalInfo.h = l.h;
    data->normal.camMoeCombineNormalInfo.totalWinSize = l.totalWinSize;
    auto *attrs = context->GetAttrs();
    auto *epGroup = attrs == nullptr ? nullptr : attrs->GetAttrPointer<char>(ATTR_EP_GROUP);
    auto *tpGroup = attrs == nullptr ? nullptr : attrs->GetAttrPointer<char>(ATTR_TP_GROUP);
    OP_TILING_CHECK(epGroup == nullptr || tpGroup == nullptr, OP_LOGE(node, "communication group missing"),
                    return GRAPH_FAILED);
    ConfigureMc2(data, std::string(epGroup), std::string(tpGroup));
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t aiv = platform.GetCoreNumAiv();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, l.totalUbSize);
    uint64_t expertBaseEntries = 0U, expertBaseBytes = 0U, expertBaseUbBytes = 0U;
    uint64_t copyUbBytes = 0U, combineUbBytes = 0U;
    OP_TILING_CHECK(
        !CamMoeLayeredHost::AddFits(l.moeExpertNum / l.epWorldSize, 1U, expertBaseEntries) ||
            !CamMoeLayeredHost::MulFits(expertBaseEntries, sizeof(uint32_t), expertBaseBytes) ||
            !CamMoeLayeredHost::Align32Fits(expertBaseBytes, expertBaseUbBytes) ||
            expertBaseBytes > ((1U << 21U) - 1U) ||
            !CamMoeLayeredHost::AddFits(expertBaseUbBytes, PEER_COPY_UB_BYTES, copyUbBytes) ||
            copyUbBytes > std::numeric_limits<uint32_t>::max() ||
            !CamMoeLayeredHost::AddFits(RDMA_UB_BYTES + HCOMM_UB_BYTES, copyUbBytes, combineUbBytes) ||
            !CamMoeLayeredHost::AddFits(combineUbBytes, 2U * CamMoeLayered::CombinePartialUbBytes(l.h),
                                        combineUbBytes) ||
            !CamMoeLayeredHost::AddFits(combineUbBytes, CamMoeLayered::CombineOwnerRowUbBytes(l.h, l.elementBytes),
                                        combineUbBytes) ||
            expertBaseUbBytes != CamMoeLayered::CombineExpertBasesUbBytes(l.moeExpertNum / l.epWorldSize),
        OP_LOGE(node, "normal-layered combine expert-base UB arithmetic overflows"), return GRAPH_FAILED);
    OP_TILING_CHECK(aiv > std::numeric_limits<uint32_t>::max(), OP_LOGE(node, "AIV count overflows"),
                    return GRAPH_FAILED);
    l.aivNum = static_cast<uint32_t>(aiv);
    data->normal.camMoeCombineNormalInfo.aivNum = aiv;
    const uint64_t chunkPackets =
        std::max<uint64_t>(1U, std::min<uint64_t>(l.maxBs, PARTIAL_CHUNK_BYTES / combineWindow.partialStride));
    const uint64_t maxChunks = 1U + (static_cast<uint64_t>(l.maxBs) - 1U) / chunkPackets;
    const uint64_t workerCount = aiv > l.serverNum ? aiv - l.serverNum : aiv;
    uint64_t maxChunkBytes = 0U;
    uint64_t progressBytes = 0U;
    OP_TILING_CHECK(chunkPackets > std::numeric_limits<uint32_t>::max() ||
                        maxChunks > std::numeric_limits<uint32_t>::max() ||
                        static_cast<uint64_t>(l.serverNum) * workerCount > std::numeric_limits<uint32_t>::max() ||
                        !CamMoeLayeredHost::MulFits(chunkPackets, combineWindow.partialStride, maxChunkBytes) ||
                        maxChunkBytes > std::numeric_limits<uint32_t>::max() ||
                        !CamMoeLayeredHost::MulFits(l.serverNum, workerCount, progressBytes) ||
                        !CamMoeLayeredHost::MulFits(progressBytes, PROGRESS_STRIDE_BYTES, progressBytes) ||
                        progressBytes > std::numeric_limits<size_t>::max() || progressBytes > ((1U << 21U) - 1U) ||
                        progressBytes > std::numeric_limits<uint32_t>::max() ||
                        !CamMoeLayeredHost::AddFits(combineUbBytes, progressBytes, combineUbBytes),
                    OP_LOGE(node, "normal-layered combine chunk or progress capacity overflows"), return GRAPH_FAILED);
    OP_TILING_CHECK(aiv == 0U || l.totalUbSize < combineUbBytes,
                    OP_LOGE(node, "normal-layered combine UB budget is invalid"), return GRAPH_FAILED);
    data->chunkPackets = static_cast<uint32_t>(chunkPackets);
    auto *workspace = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspace == nullptr, OP_LOGE(node, "workspace allocation unavailable"), return GRAPH_FAILED);
    workspace[0] = std::max(WORKSPACE_MIN_BYTES, progressBytes);
    context->SetBlockDim(platform.CalcTschBlockDim(aiv, 0, aiv));
    context->SetScheduleMode(1);
    context->SetTilingKey(13000U);
    return GRAPH_SUCCESS;
}

struct CamMoeCombineNormalLayeredCompileInfo {};
static ge::graphStatus ParseCamMoeCombineNormalLayered(gert::TilingParseContext *)
{
    return GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(CamMoeCombineNormalLayered)
    .Tiling(CamMoeCombineNormalLayeredTiling)
    .TilingParse<CamMoeCombineNormalLayeredCompileInfo>(ParseCamMoeCombineNormalLayered);
}  // namespace optiling
