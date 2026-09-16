#include <cstdio>
#include <cstdlib>
#include <string.h>
#include "graph/types.h"
#include "aclnn_cam_moe_dispatch_normal.h"
#include "aclnnInner_cam_moe_dispatch_normal.h"

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

namespace {

bool IsCamMoeDispatchNormalDebugEnabled()
{
    return 1;
}

void LogCamMoeDispatchNormalTensor(const char *name, const aclTensor *tensor)
{
    if (tensor == nullptr) {
        std::fprintf(stderr, "[DEEPEP_DEBUG_CAM_MOE_DISPATCH_NORMAL][op-api] %s=null\n", name);
        return;
    }

    int64_t *dims = nullptr;
    uint64_t dim_num = 0;
    aclDataType dtype = ACL_DT_UNDEFINED;
    const auto shape_status = aclGetViewShape(tensor, &dims, &dim_num);
    const auto dtype_status = aclGetDataType(tensor, &dtype);
    std::fprintf(stderr, "[DEEPEP_DEBUG_CAM_MOE_DISPATCH_NORMAL][op-api] %s=%p shape_status=%d shape=[", name,
                 static_cast<const void *>(tensor), shape_status);
    if (shape_status == ACL_SUCCESS && dims != nullptr) {
        for (uint64_t i = 0; i < dim_num; ++i) {
            std::fprintf(stderr, "%s%lld", i == 0 ? "" : ",", static_cast<long long>(dims[i]));
        }
    }
    std::fprintf(stderr, "] dtype_status=%d dtype=%d\n", dtype_status, static_cast<int>(dtype));
    delete[] dims;
}

void LogCamMoeDispatchNormalWorkspaceArguments(
    const aclTensor *x, const aclTensor *topkIdx, const aclTensor *sendOffset, const aclTensor *sendTokenIdx,
    const aclTensor *recvOffset, const aclTensor *recvCount, const aclTensor *expert_global_offset,
    const aclTensor *srcrank_in_expert_offset, const aclTensor *r_in_srcrank_offset, const char *groupEp,
    int64_t epWorldSize, int64_t epRankId, const char *groupTpOptional, int64_t tpWorldSize, int64_t tpRankId,
    int64_t moeExpertNum, int64_t quantMode, int64_t realMaxBs, int64_t globalBs, int32_t round, int32_t perRoundTokens,
    const aclTensor *recvX, const aclTensor *recvXScales, const aclTensor *assistInfoForCombine,
    const aclTensor *waitRecvCostStats)
{
    std::fprintf(stderr,
                 "[DEEPEP_DEBUG_CAM_MOE_DISPATCH_NORMAL][op-api] GetWorkspaceSize groupEp=%s epWorldSize=%lld "
                 "epRankId=%lld groupTp=%s tpWorldSize=%lld tpRankId=%lld moeExpertNum=%lld quantMode=%lld "
                 "realMaxBs=%lld globalBs=%lld round=%d perRoundTokens=%d\n",
                 groupEp == nullptr ? "<null>" : groupEp, static_cast<long long>(epWorldSize),
                 static_cast<long long>(epRankId), groupTpOptional == nullptr ? "<null>" : groupTpOptional,
                 static_cast<long long>(tpWorldSize), static_cast<long long>(tpRankId),
                 static_cast<long long>(moeExpertNum), static_cast<long long>(quantMode),
                 static_cast<long long>(realMaxBs), static_cast<long long>(globalBs), round, perRoundTokens);
    LogCamMoeDispatchNormalTensor("x", x);
    LogCamMoeDispatchNormalTensor("topkIdx", topkIdx);
    LogCamMoeDispatchNormalTensor("sendOffset", sendOffset);
    LogCamMoeDispatchNormalTensor("sendTokenIdx", sendTokenIdx);
    LogCamMoeDispatchNormalTensor("recvOffset", recvOffset);
    LogCamMoeDispatchNormalTensor("recvCount", recvCount);
    LogCamMoeDispatchNormalTensor("expert_global_offset", expert_global_offset);
    LogCamMoeDispatchNormalTensor("srcrank_in_expert_offset", srcrank_in_expert_offset);
    LogCamMoeDispatchNormalTensor("r_in_srcrank_offset", r_in_srcrank_offset);
    LogCamMoeDispatchNormalTensor("recvX", recvX);
    LogCamMoeDispatchNormalTensor("recvXScales", recvXScales);
    LogCamMoeDispatchNormalTensor("assistInfoForCombine", assistInfoForCombine);
    LogCamMoeDispatchNormalTensor("waitRecvCostStats", waitRecvCostStats);
    std::fflush(stderr);
}

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnCamMoeDispatchNormalGetWorkspaceSize(
    const aclTensor *x, const aclTensor *topkIdx, const aclTensor *sendOffset, const aclTensor *sendTokenIdx,
    const aclTensor *recvOffset, const aclTensor *recvCount, const aclTensor *expert_global_offset,
    const aclTensor *srcrank_in_expert_offset, const aclTensor *r_in_srcrank_offset, char *groupEp, int64_t epWorldSize,
    int64_t epRankId, char *groupTpOptional, int64_t tpWorldSize, int64_t tpRankId, int64_t moeExpertNum,
    int64_t quantMode, int64_t realMaxBs, int64_t globalBs, int32_t round, int32_t perRoundTokens,
    const aclTensor *recvX, const aclTensor *recvXScales, const aclTensor *assistInfoForCombine,
    const aclTensor *waitRecvCostStats, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    const bool debug_enabled = IsCamMoeDispatchNormalDebugEnabled();
    if (debug_enabled) {
        LogCamMoeDispatchNormalWorkspaceArguments(
            x, topkIdx, sendOffset, sendTokenIdx, recvOffset, recvCount, expert_global_offset, srcrank_in_expert_offset,
            r_in_srcrank_offset, groupEp, epWorldSize, epRankId, groupTpOptional, tpWorldSize, tpRankId, moeExpertNum,
            quantMode, realMaxBs, globalBs, round, perRoundTokens, recvX, recvXScales, assistInfoForCombine,
            waitRecvCostStats);
    }
    const aclnnStatus status = aclnnInnerCamMoeDispatchNormalGetWorkspaceSize(
        x, topkIdx, sendOffset, sendTokenIdx, recvOffset, recvCount, expert_global_offset, srcrank_in_expert_offset,
        r_in_srcrank_offset, groupEp, epWorldSize, epRankId, groupTpOptional, tpWorldSize, tpRankId, moeExpertNum,
        quantMode, realMaxBs, globalBs, round, perRoundTokens, recvX, recvXScales, assistInfoForCombine,
        waitRecvCostStats, workspaceSize, executor);
    if (debug_enabled) {
        std::fprintf(stderr,
                     "[DEEPEP_DEBUG_CAM_MOE_DISPATCH_NORMAL][op-api] GetWorkspaceSize status=%d workspaceSize=%llu "
                     "executor=%p\n",
                     status, workspaceSize == nullptr ? 0ULL : static_cast<unsigned long long>(*workspaceSize),
                     executor == nullptr ? nullptr : static_cast<void *>(*executor));
        std::fflush(stderr);
    }
    return status;
}

aclnnStatus aclnnCamMoeDispatchNormal(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                      aclrtStream stream)
{
    const bool debug_enabled = IsCamMoeDispatchNormalDebugEnabled();
    if (debug_enabled) {
        std::fprintf(stderr,
                     "[DEEPEP_DEBUG_CAM_MOE_DISPATCH_NORMAL][op-api] Execute workspace=%p workspaceSize=%llu "
                     "executor=%p stream=%p server=MTE\n",
                     workspace, static_cast<unsigned long long>(workspaceSize), static_cast<void *>(executor), stream);
        std::fflush(stderr);
    }
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    const aclnnStatus status = aclnnInnerCamMoeDispatchNormal(workspace, workspaceSize, executor, stream);
    if (debug_enabled) {
        std::fprintf(stderr, "[DEEPEP_DEBUG_CAM_MOE_DISPATCH_NORMAL][op-api] Execute status=%d\n", status);
        std::fflush(stderr);
    }
    return status;
}

#ifdef __cplusplus
}
#endif
