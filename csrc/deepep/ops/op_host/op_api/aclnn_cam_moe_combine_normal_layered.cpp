#include "aclnn_cam_moe_combine_normal_layered.h"
#include "aclnnInner_cam_moe_combine_normal_layered.h"

enum NnopbaseHcclServerType { NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0, NNOPBASE_HCCL_SERVER_TYPE_MTE };
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *, NnopbaseHcclServerType);

extern "C" aclnnStatus aclnnCamMoeCombineNormalLayeredGetWorkspaceSize(
    const aclTensor *roceContext, const aclTensor *recvX, const aclTensor *layeredPlan, const aclTensor *layeredAssist,
    char *epGroupName, int64_t epWorldSize, int64_t epRankId, char *tpGroupName, int64_t moeExpertNum,
    int64_t realMaxBs, int32_t round, int32_t perRoundTokens, int64_t ranksPerServer, int64_t topk, int64_t maxBs,
    int64_t ownerCapacity, int64_t invocationId, int64_t windowCapacity, const aclTensor *x, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return aclnnInnerCamMoeCombineNormalLayeredGetWorkspaceSize(
        roceContext, recvX, layeredPlan, layeredAssist, epGroupName, epWorldSize, epRankId, tpGroupName, moeExpertNum,
        realMaxBs, round, perRoundTokens, ranksPerServer, topk, maxBs, ownerCapacity, invocationId, windowCapacity, x,
        workspaceSize, executor);
}

extern "C" aclnnStatus aclnnCamMoeCombineNormalLayered(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                       aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerCamMoeCombineNormalLayered(workspace, workspaceSize, executor, stream);
}
