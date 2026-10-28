#ifndef ACLNN_CAM_MOE_DISPATCH_NORMAL_LAYERED_H_
#define ACLNN_CAM_MOE_DISPATCH_NORMAL_LAYERED_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnCamMoeDispatchNormalLayeredGetWorkspaceSize(
    const aclTensor *roceContext, const aclTensor *x, const aclTensor *topkWeights, const aclTensor *topkIdx,
    const aclTensor *layeredPlan, char *groupEp, int64_t epWorldSize, int64_t epRankId, char *groupTp,
    int64_t moeExpertNum, int64_t ranksPerServer, int64_t maxBs, const aclTensor *recvX, const aclTensor *layeredAssist,
    const aclTensor *recvTopkIdx, const aclTensor *recvTopkWeights, uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnCamMoeDispatchNormalLayered(void *workspace,
                                                                                    uint64_t workspaceSize,
                                                                                    aclOpExecutor *executor,
                                                                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
