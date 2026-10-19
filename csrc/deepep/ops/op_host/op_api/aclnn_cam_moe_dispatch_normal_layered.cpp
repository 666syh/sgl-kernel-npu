#include "aclnn_cam_moe_dispatch_normal_layered.h"
#include "aclnnInner_cam_moe_dispatch_normal_layered.h"

enum NnopbaseHcclServerType { NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0, NNOPBASE_HCCL_SERVER_TYPE_MTE };
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *, NnopbaseHcclServerType);

extern "C" aclnnStatus aclnnCamMoeDispatchNormalLayeredGetWorkspaceSize(
    const aclTensor *roceContext, const aclTensor *x, const aclTensor *topkWeights, const aclTensor *topkIdx,
    const aclTensor *layeredPlan, char *groupEp, int64_t epWorldSize, int64_t epRankId, char *groupTp,
    int64_t tpWorldSize, int64_t tpRankId, int64_t moeExpertNum, int64_t ranksPerServer, int64_t rdmaSlotsPerPeer,
    int64_t ipcTilePackets, int64_t maxBs, const aclTensor *recvX, const aclTensor *layeredAssist,
    const aclTensor *recvTopkIdx, const aclTensor *recvTopkWeights, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerCamMoeDispatchNormalLayeredGetWorkspaceSize(
        roceContext, x, topkWeights, topkIdx, layeredPlan, groupEp, epWorldSize, epRankId, groupTp, tpWorldSize,
        tpRankId, moeExpertNum, ranksPerServer, rdmaSlotsPerPeer, ipcTilePackets, maxBs, recvX, layeredAssist,
        recvTopkIdx, recvTopkWeights, workspaceSize, executor);
}

extern "C" aclnnStatus aclnnCamMoeDispatchNormalLayered(void *workspace, uint64_t workspaceSize,
                                                        aclOpExecutor *executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerCamMoeDispatchNormalLayered(workspace, workspaceSize, executor, stream);
}
