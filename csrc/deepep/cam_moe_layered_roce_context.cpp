#include "cam_moe_layered_roce_context.hpp"

#ifdef __DAV_C310__
#include "hccl/hccl_rank_graph.h"
#include "hccl/hccl_res.h"

#include <vector>

namespace deep_ep {
namespace {
constexpr uint32_t kSingleChannelPerRank = 1U;
constexpr char kLayeredWindowMemTag[] = "cam_moe_layered_window";

HcclResult FindRoceLink(HcclComm comm, uint32_t srcRank, uint32_t dstRank, CommLink &result)
{
    uint32_t *layers = nullptr;
    uint32_t layerNum = 0U;
    HcclResult ret = HcclRankGraphGetLayers(comm, &layers, &layerNum);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }

    for (uint32_t layer = 0; layer < layerNum; ++layer) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0U;
        ret = HcclRankGraphGetLinks(comm, layers[layer], srcRank, dstRank, &links, &linkNum);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        for (uint32_t index = 0; index < linkNum; ++index) {
            if (links[index].linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_ROCE) {
                result = links[index];
                return HCCL_SUCCESS;
            }
        }
    }
    return HCCL_E_NOT_FOUND;
}
}  // namespace

HcclResult BuildLayeredRoceContext(HcclComm comm, const char *contextTag, uint32_t ranksPerServer, uint64_t windowBytes,
                                   LayeredRoceResource &resource)
{
    if (comm == nullptr || contextTag == nullptr || ranksPerServer == 0U || windowBytes == 0U) {
        return HCCL_E_PARA;
    }

    HcclResult ret = HcclGetRankId(comm, &resource.rankId);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    ret = HcclGetRankSize(comm, &resource.rankNum);
    if (ret != HCCL_SUCCESS || resource.rankNum == 0U || resource.rankId >= resource.rankNum ||
        resource.rankNum > CAM_MOE_LAYERED_MAX_RANK_NUM || resource.rankNum % ranksPerServer != 0U) {
        return ret == HCCL_SUCCESS ? HCCL_E_PARA : ret;
    }

    uint64_t existingSize = 0U;
    ret = HcclEngineCtxGet(comm, contextTag, CommEngine::COMM_ENGINE_AIV, &resource.deviceContext, &existingSize);
    if (ret == HCCL_SUCCESS) {
        if (existingSize != sizeof(CamMoeLayeredRoceContext)) {
            return HCCL_E_PARA;
        }
        void *window = nullptr;
        uint64_t registeredWindowBytes = 0U;
        ret = HcclGetHcclBuffer(comm, &window, &registeredWindowBytes);
        if (ret != HCCL_SUCCESS || window == nullptr || registeredWindowBytes < windowBytes) {
            return ret == HCCL_SUCCESS ? HCCL_E_MEMORY : ret;
        }
        resource.deviceWindow = window;
        resource.windowBytes = registeredWindowBytes;
        return HCCL_SUCCESS;
    }

    resource.context = {};
    resource.context.rankNum = resource.rankNum;
    resource.context.channelsPerRank = kSingleChannelPerRank;
    void *window = nullptr;
    uint64_t registeredWindowBytes = 0U;
    ret = HcclGetHcclBuffer(comm, &window, &registeredWindowBytes);
    if (ret != HCCL_SUCCESS || window == nullptr || registeredWindowBytes < windowBytes) {
        return ret == HCCL_SUCCESS ? HCCL_E_MEMORY : ret;
    }
    resource.deviceWindow = window;
    resource.windowBytes = registeredWindowBytes;
    resource.context.windowBytes = registeredWindowBytes;

    const uint32_t localServer = resource.rankId / ranksPerServer;
    const uint32_t remoteRankCount = resource.rankNum - ranksPerServer;
    std::vector<HcclChannelDesc> descriptors;
    std::vector<ChannelHandle> channels;
    descriptors.reserve(remoteRankCount);
    channels.resize(remoteRankCount);

    HcclMemHandle windowMemHandle = nullptr;
    if (remoteRankCount != 0U) {
        CommMem windowMem{COMM_MEM_TYPE_DEVICE, window, registeredWindowBytes};
        ret = HcclCommMemReg(comm, kLayeredWindowMemTag, &windowMem, &windowMemHandle);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
    }

    for (uint32_t dstRank = 0; dstRank < resource.rankNum; ++dstRank) {
        if (dstRank / ranksPerServer == localServer) {
            continue;
        }
        CommLink link{};
        ret = FindRoceLink(comm, resource.rankId, dstRank, link);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }

        HcclChannelDesc desc{};
        ret = HcclChannelDescInit(&desc, 1U);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        desc.remoteRank = dstRank;
        desc.channelProtocol = CommProtocol::COMM_PROTOCOL_ROCE;
        desc.localEndpoint = link.srcEndpointDesc;
        desc.remoteEndpoint = link.dstEndpointDesc;
        desc.memHandles = &windowMemHandle;
        desc.memHandleNum = 1U;
        descriptors.push_back(desc);
    }

    if (!descriptors.empty()) {
        ret = HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AIV, descriptors.data(),
                                 static_cast<uint32_t>(descriptors.size()), channels.data());
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
    }

    uint32_t channelIndex = 0U;
    for (uint32_t dstRank = 0; dstRank < resource.rankNum; ++dstRank) {
        if (dstRank / ranksPerServer == localServer) {
            continue;
        }
        resource.context.hcommHandle[dstRank] = channels[channelIndex++];
    }

    ret = HcclEngineCtxCreate(comm, contextTag, CommEngine::COMM_ENGINE_AIV, sizeof(CamMoeLayeredRoceContext),
                              &resource.deviceContext);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    return HcclEngineCtxCopy(comm, CommEngine::COMM_ENGINE_AIV, contextTag, &resource.context,
                             sizeof(CamMoeLayeredRoceContext), 0U);
}

}  // namespace deep_ep

#endif  // __DAV_C310__
