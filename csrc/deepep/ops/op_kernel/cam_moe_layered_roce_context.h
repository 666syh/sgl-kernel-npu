#ifndef CAM_MOE_LAYERED_ROCE_CONTEXT_H
#define CAM_MOE_LAYERED_ROCE_CONTEXT_H

#include <cstddef>
#include <cstdint>

#ifndef CAM_MOE_LAYERED_MAX_RANK_NUM
#define CAM_MOE_LAYERED_MAX_RANK_NUM 384U
#endif

// Host and device must use the same layout. Each rank owns one or more
// handles, stored consecutively by destination rank.
struct CamMoeLayeredRoceContext {
    uint32_t rankNum;
    uint32_t channelsPerRank;
    uint64_t hcommHandle[CAM_MOE_LAYERED_MAX_RANK_NUM];
    uint64_t windowBytes;
};

static_assert(offsetof(CamMoeLayeredRoceContext, hcommHandle) == 8U, "CamMoeLayeredRoceContext ABI mismatch");
static_assert(sizeof(CamMoeLayeredRoceContext) == 16U + sizeof(uint64_t) * CAM_MOE_LAYERED_MAX_RANK_NUM,
              "CamMoeLayeredRoceContext size mismatch");

#endif
