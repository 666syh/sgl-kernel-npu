#ifndef ACLNN_CAM_MOE_COMBINE_NORMAL_LAYERED_H_
#define ACLNN_CAM_MOE_COMBINE_NORMAL_LAYERED_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnCamMoeCombineNormalLayeredGetWorkspaceSize(
    const aclTensor *roceContext, const aclTensor *recvX, const aclTensor *layeredPlan, char *epGroupName,
    int64_t epWorldSize, int64_t epRankId, char *tpGroupName, int64_t moeExpertNum, int64_t realMaxBs, int32_t round,
    int32_t perRoundTokens, int64_t ranksPerServer, int64_t topk, int64_t maxBs, int64_t ownerCapacity,
    int64_t invocationId, int64_t windowCapacity, const aclTensor *x, uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnCamMoeCombineNormalLayered(void *workspace,
                                                                                   uint64_t workspaceSize,
                                                                                   aclOpExecutor *executor,
                                                                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
