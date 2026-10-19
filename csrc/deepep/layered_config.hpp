#pragma once

#include <cstdlib>
#include <cstdint>
#include <limits>

#include "exception.hpp"

namespace deep_ep {

// Host-side mirror of the serialized device plan ABI. These values are counts
// of int32 elements, not bytes. Keep them in sync with cam_moe_layered_plan.h.
constexpr int64_t kLayeredPlanHeaderElements = 32;
constexpr int64_t kLayeredSourcePacketDescElements = 4;
constexpr int64_t kLayeredReversePacketDescElements = 2;
constexpr int64_t kLayeredMemberDescElements = 3;
constexpr int64_t kLayeredReverseMemberElements = 4;

struct LayeredTopology {
    // 每个 server 内包含的 EP rank 数量。
    int64_t ranks_per_server;
    // 当前 EP 通信域中的 server 总数。
    int64_t server_num;
    // 当前 rank 在所属 server 内的局部 rank 编号，范围为 [0, ranks_per_server)。
    int64_t rank_in_server;
};

struct LayeredPlanLayout {
    // 单轮 packet count 区域，形状为 [server]。
    int64_t packet_count;
    // 单轮 packet member count 区域，形状为 [server]。
    int64_t dispatch_member_count;
    // packetSlot section 的起始偏移，形状为 [max_bs, server]。
    int64_t packet_slot;
    // expertCount section 的起始偏移，形状为 [server, expert_in_server]。
    int64_t expert_count;
    // localExpertOffset section 的起始偏移，单位为 int32 元素；记录当前 rank 每个 local
    // expert 在 compact receive buffer 中的首行位置。
    int64_t local_expert_offset;
    // localExpertCount section 的起始偏移，单位为 int32 元素；记录当前 rank 每个 local
    // expert 实际接收的 member 行数。
    int64_t local_expert_count;
    // source packet descriptor section 的起始偏移，单位为 int32 元素。
    int64_t packet_desc;
    // memberDesc section 的起始偏移，单位为 int32 元素；每个 MemberDesc 描述 packet 中
    // 的 expert、top-k 位置和当前/最终 expert row。
    int64_t member_desc;
    int64_t source_return_base;
    int64_t send_ingress_base;
    int64_t send_ingress_bytes;
    int64_t recv_ingress_base;
    int64_t recv_ingress_bytes;
    int64_t reverse_count;
    int64_t reverse_return_base;
    int64_t reverse_packet;
    int64_t reverse_member;
    // overflow 标志的偏移，单位为 int32 元素；非零表示 plan 容量或路由规划发生溢出。
    int64_t overflow;
    // error 状态的偏移，单位为 int32 元素；保存 transport、协议或其他 plan 错误码。
    int64_t error;
    // layered plan 所需的总 int32 元素数量，用于分配 plan tensor 的容量。
    int64_t total;
};

constexpr LayeredPlanLayout layered_plan_layout(int64_t bs, int64_t max_bs, int64_t topk, int64_t experts,
                                                int64_t world, int64_t servers, int64_t ranks_per_server,
                                                int64_t rounds, int64_t slots)
{
    (void)bs;
    (void)ranks_per_server;
    const int64_t local_experts = experts / world;
    const int64_t experts_per_server = experts / servers;
    LayeredPlanLayout layout{};
    layout.packet_count = kLayeredPlanHeaderElements;
    layout.dispatch_member_count = layout.packet_count + rounds * servers;
    layout.packet_slot = layout.dispatch_member_count + rounds * servers;
    layout.expert_count = layout.packet_slot + max_bs * servers;
    layout.local_expert_offset = layout.expert_count + rounds * servers * experts_per_server;
    layout.local_expert_count = layout.local_expert_offset + local_experts;
    layout.packet_desc = layout.local_expert_count + local_experts;
    layout.member_desc = layout.packet_desc + rounds * servers * slots * kLayeredSourcePacketDescElements;
    layout.source_return_base = layout.member_desc + rounds * servers * slots * topk * kLayeredMemberDescElements;
    layout.send_ingress_base = layout.source_return_base + servers;
    layout.send_ingress_bytes = layout.send_ingress_base + 2 * servers;
    layout.recv_ingress_base = layout.send_ingress_bytes + 2 * servers;
    layout.recv_ingress_bytes = layout.recv_ingress_base + 2 * servers;
    layout.reverse_count = layout.recv_ingress_bytes + 2 * servers;
    layout.reverse_return_base = layout.reverse_count + servers;
    layout.reverse_packet = layout.reverse_return_base + servers;
    layout.reverse_member = layout.reverse_packet + world * max_bs * kLayeredReversePacketDescElements;
    layout.overflow = layout.reverse_member + world * max_bs * topk * kLayeredReverseMemberElements;
    layout.error = layout.overflow + 1;
    layout.total = layout.error + 1;
    return layout;
}

inline bool layered_enabled()
{
    const char *value = std::getenv("DEEPEP_ENABLE_NORMAL_LAYERED");
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}

inline LayeredTopology layered_topology(int64_t rank, int64_t world_size)
{
    const char *value = std::getenv("DEEPEP_NORMAL_LAYERED_RANKS_PER_SERVER");
    EP_HOST_ASSERT_S(value != nullptr, "DEEPEP_NORMAL_LAYERED_RANKS_PER_SERVER",
                     " must be set when DEEPEP_ENABLE_NORMAL_LAYERED=1");
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    EP_HOST_ASSERT_S(end != value && *end == '\0' && parsed > 0 && parsed <= std::numeric_limits<int32_t>::max(),
                     "DEEPEP_NORMAL_LAYERED_RANKS_PER_SERVER", " must be a positive int32 value");
    const int64_t ranks_per_server = static_cast<int64_t>(parsed);
    EP_HOST_ASSERT_S(world_size > 0 && world_size % ranks_per_server == 0, "DEEPEP_NORMAL_LAYERED_RANKS_PER_SERVER",
                     " must divide ep world size");
    EP_HOST_ASSERT_S(world_size / ranks_per_server <= 32, "DEEPEP_NORMAL_LAYERED_RANKS_PER_SERVER",
                     " produces more than 32 servers");
    EP_HOST_ASSERT_S(rank >= 0 && rank < world_size, "normal layered rank", " is outside ep world size");
    return {ranks_per_server, world_size / ranks_per_server, rank % ranks_per_server};
}

}  // namespace deep_ep
