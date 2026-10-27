#ifndef CAM_MOE_LAYERED_COMM_CONTEXT_H
#define CAM_MOE_LAYERED_COMM_CONTEXT_H

#include <cstddef>
#include <cstdint>

#ifndef CAM_MOE_LAYERED_MAX_RANK_NUM
#define CAM_MOE_LAYERED_MAX_RANK_NUM 384U
#endif

constexpr uint64_t CAM_MOE_LAYERED_A5_STATE_BYTES = 4096ULL * 1024ULL;

enum CamMoeLayeredProtocol : uint32_t {
    CAM_MOE_LAYERED_PROTOCOL_ROCE = 1U,
    CAM_MOE_LAYERED_PROTOCOL_UBOE = 2U,
};

// Host and device ABI for an EP-wide layered communication domain. Channel
// handles use the compact peer index: peer < rankId ? peer : peer - 1.
struct CamMoeLayeredCommContext {
    uint32_t protocol;
    uint64_t ipcBuffer[CAM_MOE_LAYERED_MAX_RANK_NUM];
    uint64_t hcommHandle[CAM_MOE_LAYERED_MAX_RANK_NUM - 1U];
    uint64_t windowBytes;
};

static_assert(offsetof(CamMoeLayeredCommContext, protocol) == 0U, "layered context ABI mismatch");
static_assert(offsetof(CamMoeLayeredCommContext, ipcBuffer) == 8U, "layered context ABI mismatch");
static_assert(offsetof(CamMoeLayeredCommContext, hcommHandle) == 8U + sizeof(uint64_t) * CAM_MOE_LAYERED_MAX_RANK_NUM,
              "layered context handles ABI mismatch");
static_assert(offsetof(CamMoeLayeredCommContext, windowBytes) == 16U * CAM_MOE_LAYERED_MAX_RANK_NUM,
              "layered context window size ABI mismatch");
static_assert(sizeof(CamMoeLayeredCommContext) == 16U * CAM_MOE_LAYERED_MAX_RANK_NUM + 8U,
              "layered context size mismatch");

#endif
