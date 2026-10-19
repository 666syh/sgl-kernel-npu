#ifndef CAM_MOE_LAYERED_ROCE_HOST_CONTEXT_HPP
#define CAM_MOE_LAYERED_ROCE_HOST_CONTEXT_HPP

#include <cstdint>

#include "hccl/hccl.h"

#include "ops/op_kernel/cam_moe_layered_roce_context.h"

namespace deep_ep {

struct LayeredRoceResource {
    uint32_t rankId = 0;
    uint32_t rankNum = 0;
    CamMoeLayeredRoceContext context{};
    void *deviceContext = nullptr;
    void *deviceWindow = nullptr;
    uint64_t windowBytes = 0U;
};

// Creates the A5 RoCE channel table and copies it to a device-visible HCCL
// engine context. The registered memory is the complete communicator HCCL
// buffer, which is also the allocation exported through the A5 Window API.
HcclResult BuildLayeredRoceContext(HcclComm comm, const char *contextTag, uint32_t ranksPerServer, uint64_t windowBytes,
                                   LayeredRoceResource &resource);

}  // namespace deep_ep

#endif
