#include "cam_moe_layered_comm_context.hpp"

#ifdef __DAV_C310__
#include "hccl/hccl_rank_graph.h"
#include "hccl/hccl_res.h"

#include <acl/acl_rt.h>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace deep_ep {
namespace {
constexpr uint64_t kDefaultPayloadBytes = 200ULL * 1024ULL * 1024ULL;
constexpr uint64_t kMiB = 1024ULL * 1024ULL;

struct SharedLayeredWindow {
    std::mutex mutex;
    void *buffer = nullptr;
    HcclMemHandle memHandle = nullptr;
    uint64_t payloadBytes = 0U;
    uint64_t registeredBytes = 0U;
    uint32_t rank = 0U;
    uint32_t world = 0U;
    uint32_t ranksPerServer = 0U;
    CamMoeLayeredProtocol protocol = CAM_MOE_LAYERED_PROTOCOL_UBOE;
    CamMoeLayeredCommContext host{};
    bool hostReady = false;
    bool contextReady = false;
};

std::shared_ptr<SharedLayeredWindow> GetSharedWindow(HcclComm comm, const std::string &contextTag)
{
    static std::mutex poolMutex;
    static std::map<std::pair<uintptr_t, std::string>, std::shared_ptr<SharedLayeredWindow>> pool;
    std::lock_guard<std::mutex> lock(poolMutex);
    auto &entry = pool[std::make_pair(reinterpret_cast<uintptr_t>(comm), contextTag)];
    if (!entry) {
        entry = std::make_shared<SharedLayeredWindow>();
    }
    return entry;
}

void FillResource(const SharedLayeredWindow &entry, void *engineContext, LayeredCommResource &resource)
{
    resource.engineContext = engineContext;
    resource.windowBytes = entry.payloadBytes;
}

CommProtocol ToHcclProtocol(CamMoeLayeredProtocol protocol)
{
    return protocol == CAM_MOE_LAYERED_PROTOCOL_ROCE ? CommProtocol::COMM_PROTOCOL_ROCE
                                                     : CommProtocol::COMM_PROTOCOL_UB_CTP;
}

uint64_t ParseCapacity(const char *name)
{
    const char *value = std::getenv(name);
    if (value == nullptr) {
        return 0U;
    }
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0' || parsed == 0U ||
        parsed > (std::numeric_limits<uint64_t>::max() - CAM_MOE_LAYERED_A5_STATE_BYTES) / kMiB) {
        throw std::invalid_argument(std::string(name) + " must be a positive MiB count");
    }
    return static_cast<uint64_t>(parsed) * kMiB;
}

uint64_t GetPayloadCapacity()
{
    const uint64_t deepep = ParseCapacity("DEEPEP_HCCL_BUFFSIZE");
    if (deepep != 0U) {
        return deepep;
    }
    const uint64_t hccl = ParseCapacity("HCCL_BUFFSIZE");
    return hccl != 0U ? hccl : kDefaultPayloadBytes;
}

HcclResult FindProtocolLink(HcclComm comm, uint32_t src, uint32_t dst, CommProtocol protocol, CommLink &result)
{
    uint32_t *layers = nullptr;
    uint32_t layerCount = 0U;
    HcclResult ret = HcclRankGraphGetLayers(comm, &layers, &layerCount);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    for (uint32_t layer = 0U; layer < layerCount; ++layer) {
        CommLink *links = nullptr;
        uint32_t linkCount = 0U;
        ret = HcclRankGraphGetLinks(comm, layers[layer], src, dst, &links, &linkCount);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        for (uint32_t i = 0U; i < linkCount; ++i) {
            if (links[i].linkAttr.linkProtocol == protocol) {
                result = links[i];
                return HCCL_SUCCESS;
            }
        }
    }
    return HCCL_E_NOT_FOUND;
}
}  // namespace

LayeredCommContextManager::~LayeredCommContextManager() = default;

HcclResult LayeredCommContextManager::Create(HcclComm comm, const std::string &contextTag,
                                             CamMoeLayeredProtocol protocol, uint32_t ranksPerServer,
                                             uint64_t minimumWindowBytes, LayeredCommResource &resource)
{
    if (comm == nullptr || ranksPerServer == 0U || minimumWindowBytes == 0U ||
        (protocol != CAM_MOE_LAYERED_PROTOCOL_ROCE && protocol != CAM_MOE_LAYERED_PROTOCOL_UBOE)) {
        return HCCL_E_PARA;
    }
    uint32_t rank = 0U;
    uint32_t world = 0U;
    HcclResult ret = HcclGetRankId(comm, &rank);
    if (ret != HCCL_SUCCESS) return ret;
    ret = HcclGetRankSize(comm, &world);
    if (ret != HCCL_SUCCESS) return ret;
    if (world == 0U || rank >= world || world > CAM_MOE_LAYERED_MAX_RANK_NUM || world % ranksPerServer != 0U) {
        return HCCL_E_PARA;
    }

    const uint64_t requestedBytes = GetPayloadCapacity();
    if (requestedBytes < minimumWindowBytes ||
        requestedBytes > std::numeric_limits<uint64_t>::max() - CAM_MOE_LAYERED_A5_STATE_BYTES) {
        return HCCL_E_MEMORY;
    }
    const uint64_t maxTag = 255U;
    const std::string memTag = contextTag + "_buffer";
    if (contextTag.size() > maxTag || memTag.size() > maxTag) {
        return HCCL_E_PARA;
    }

    auto shared = GetSharedWindow(comm, contextTag);
    std::lock_guard<std::mutex> lock(shared->mutex);
    void *engineContext = nullptr;
    uint64_t contextBytes = 0U;
    const HcclResult getRet =
        HcclEngineCtxGet(comm, contextTag.c_str(), CommEngine::COMM_ENGINE_AIV, &engineContext, &contextBytes);
    if (getRet != HCCL_SUCCESS && getRet != HCCL_E_NOT_FOUND) {
        return getRet;
    }
    const bool contextExists = getRet == HCCL_SUCCESS;
    if (contextExists && (engineContext == nullptr || contextBytes != sizeof(CamMoeLayeredCommContext))) {
        return HCCL_E_PARA;
    }
    // A matching HCCL context must never outlive its registered window.
    if (contextExists && (shared->buffer == nullptr || shared->memHandle == nullptr || !shared->hostReady)) {
        return HCCL_E_NOT_FOUND;
    }
    if (!contextExists && shared->contextReady) {
        return HCCL_E_NOT_FOUND;
    }
    if (shared->buffer != nullptr && shared->memHandle == nullptr) {
        return HCCL_E_NOT_FOUND;
    }
    if (shared->buffer != nullptr) {
        if (!shared->hostReady) {
            throw std::runtime_error("layered context '" + contextTag +
                                     "' has an incomplete registered window; use a new communication domain");
        }
        if (requestedBytes > shared->payloadBytes || minimumWindowBytes > shared->payloadBytes) {
            throw std::runtime_error("layered context '" + contextTag + "' has a fixed registered payload of " +
                                     std::to_string(shared->payloadBytes) + " bytes; requested " +
                                     std::to_string(requestedBytes) + " bytes with a minimum of " +
                                     std::to_string(minimumWindowBytes) + ". Use a new communication domain");
        }
        if (shared->rank != rank || shared->world != world || shared->ranksPerServer != ranksPerServer ||
            shared->protocol != protocol) {
            return HCCL_E_PARA;
        }
        if (contextExists && shared->contextReady) {
            FillResource(*shared, engineContext, resource);
            return HCCL_SUCCESS;
        }
    } else {
        const uint64_t registeredBytes = requestedBytes + CAM_MOE_LAYERED_A5_STATE_BYTES;
        void *buffer = nullptr;
        aclError aclRet = aclrtMalloc(&buffer, registeredBytes, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (aclRet != ACL_SUCCESS) {
            return HCCL_E_MEMORY;
        }
        aclRet = aclrtMemset(buffer, registeredBytes, 0, registeredBytes);
        if (aclRet != ACL_SUCCESS) {
            (void)aclrtFree(buffer);
            return HCCL_E_MEMORY;
        }
        CommMem mem{};
        mem.type = COMM_MEM_TYPE_DEVICE;
        mem.addr = buffer;
        mem.size = registeredBytes;
        HcclMemHandle memHandle = nullptr;
        ret = HcclCommMemReg(comm, memTag.c_str(), &mem, &memHandle);
        if (ret != HCCL_SUCCESS) {
            (void)aclrtFree(buffer);
            return ret;
        }
        // HCCL has no matching unregister operation. Keep successfully
        // registered storage even if a later channel/context step fails.
        shared->buffer = buffer;
        shared->memHandle = memHandle;
        shared->payloadBytes = requestedBytes;
        shared->registeredBytes = registeredBytes;
        shared->rank = rank;
        shared->world = world;
        shared->ranksPerServer = ranksPerServer;
        shared->protocol = protocol;
        if (memHandle == nullptr) {
            return HCCL_E_NOT_FOUND;
        }
    }

    if (shared->hostReady) {
        if (!contextExists) {
            ret = HcclEngineCtxCreate(comm, contextTag.c_str(), CommEngine::COMM_ENGINE_AIV,
                                      sizeof(CamMoeLayeredCommContext), &engineContext);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            if (engineContext == nullptr) {
                return HCCL_E_NOT_FOUND;
            }
        }
        ret = HcclEngineCtxCopy(comm, CommEngine::COMM_ENGINE_AIV, contextTag.c_str(), &shared->host,
                                sizeof(CamMoeLayeredCommContext), 0U);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        shared->contextReady = true;
        FillResource(*shared, engineContext, resource);
        return HCCL_SUCCESS;
    }

    CamMoeLayeredCommContext host{};
    host.protocol = static_cast<uint32_t>(protocol);
    host.windowBytes = shared->payloadBytes;
    host.ipcBuffer[rank] = reinterpret_cast<uint64_t>(shared->buffer);

    std::vector<HcclChannelDesc> descs;
    std::vector<uint32_t> peers;
    const CommProtocol selectedProtocol = ToHcclProtocol(protocol);
    descs.reserve(world - 1U);
    peers.reserve(world - 1U);
    for (uint32_t peer = 0U; peer < world; ++peer) {
        if (peer == rank) continue;
        const bool intraServer = peer / ranksPerServer == rank / ranksPerServer;
        const CommProtocol peerProtocol = intraServer ? CommProtocol::COMM_PROTOCOL_UB_MEM : selectedProtocol;
        CommLink link{};
        ret = FindProtocolLink(comm, rank, peer, peerProtocol, link);
        if (ret != HCCL_SUCCESS) return ret;
        HcclChannelDesc desc{};
        ret = HcclChannelDescInit(&desc, 1U);
        if (ret != HCCL_SUCCESS) return ret;
        desc.remoteRank = peer;
        desc.channelProtocol = peerProtocol;
        desc.localEndpoint = link.srcEndpointDesc;
        desc.remoteEndpoint = link.dstEndpointDesc;
        desc.memHandles = &shared->memHandle;
        desc.memHandleNum = 1U;
        descs.push_back(desc);
        peers.push_back(peer);
    }

    std::vector<ChannelHandle> channels(descs.size());
    if (!descs.empty()) {
        ret = HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AIV, descs.data(), static_cast<uint32_t>(descs.size()),
                                 channels.data());
        if (ret != HCCL_SUCCESS) return ret;
    }
    for (size_t i = 0U; i < peers.size(); ++i) {
        const uint32_t peer = peers[i];
        const uint32_t compact = peer < rank ? peer : peer - 1U;
        host.hcommHandle[compact] = static_cast<uint64_t>(channels[i]);
        uint32_t remoteCount = 0U;
        CommMem *remoteMems = nullptr;
        char **remoteTags = nullptr;
        ret = HcclChannelGetRemoteMems(comm, channels[i], &remoteCount, &remoteMems, &remoteTags);
        if (ret != HCCL_SUCCESS) return ret;
        bool found = false;
        for (uint32_t j = 0U; j < remoteCount; ++j) {
            if (remoteTags != nullptr && remoteMems != nullptr && remoteTags[j] != nullptr && memTag == remoteTags[j] &&
                remoteMems[j].addr != nullptr && remoteMems[j].size >= shared->registeredBytes) {
                host.ipcBuffer[peer] = reinterpret_cast<uint64_t>(remoteMems[j].addr);
                found = true;
                break;
            }
        }
        if (!found) return HCCL_E_NOT_FOUND;
    }

    shared->host = host;
    shared->hostReady = true;
    ret = HcclEngineCtxCreate(comm, contextTag.c_str(), CommEngine::COMM_ENGINE_AIV, sizeof(CamMoeLayeredCommContext),
                              &engineContext);
    if (ret != HCCL_SUCCESS) return ret;
    if (engineContext == nullptr) return HCCL_E_NOT_FOUND;
    ret = HcclEngineCtxCopy(comm, CommEngine::COMM_ENGINE_AIV, contextTag.c_str(), &host,
                            sizeof(CamMoeLayeredCommContext), 0U);
    if (ret != HCCL_SUCCESS) return ret;
    shared->contextReady = true;
    FillResource(*shared, engineContext, resource);
    return HCCL_SUCCESS;
}

}  // namespace deep_ep
#else
namespace deep_ep {
LayeredCommContextManager::~LayeredCommContextManager() = default;
}  // namespace deep_ep
#endif  // __DAV_C310__
