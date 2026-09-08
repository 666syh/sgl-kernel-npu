#ifndef TILING_ARGS_H
#define TILING_ARGS_H

#include <cstdint>

namespace Moe {
namespace A3WindowLayout {
// Keep these host-side layout values synchronized with op_kernel/window_layout.h.
constexpr uint64_t KB = 1024UL;
constexpr uint64_t MB = 1024UL * KB;

constexpr uint64_t kNotifyDispatchSize = 102UL * MB;
constexpr uint64_t kNormalCombineStateSize = 4UL * MB;
constexpr uint64_t kAivCount = 48UL;
constexpr uint64_t kAivMetadataStride = 512UL;
constexpr uint64_t kV2SelectorMetadataSize = kAivCount * kAivMetadataStride;
constexpr uint64_t kV2StateSize = 1UL * MB;
constexpr uint64_t kV2StateTimeoutOffset = 1000UL * KB;
constexpr uint64_t kV2StateTimeoutBytes = 8UL * sizeof(float);
constexpr uint64_t kV2StateEntrySize = 32UL;
constexpr uint64_t kV2MaxBs = 512UL;
constexpr uint64_t kV2MaxTopK = 16UL;
constexpr uint64_t kV2MaxSharedExpertNum = 4UL;

// Keep these legacy values synchronized with op_kernel/window_layout.h.
constexpr uint64_t kLegacyNormalDataOffset = kNotifyDispatchSize + kNormalCombineStateSize;
constexpr uint64_t kLegacyV2StateHalfSize = 500UL * KB;
constexpr uint64_t kLegacyV2CombineStateOffset = 64UL * KB;
constexpr uint64_t kLegacyV2DispatchSelectorOffset = 950UL * KB;
constexpr uint64_t kLegacyV2CombineSelectorOffset = 975UL * KB;

constexpr uint64_t kV2DispatchSelectorOffset = kNotifyDispatchSize + kNormalCombineStateSize;
constexpr uint64_t kV2DispatchStateOffset = kV2DispatchSelectorOffset + kV2SelectorMetadataSize;
constexpr uint64_t kV2CombineSelectorOffset = kV2DispatchStateOffset + kV2StateSize;
constexpr uint64_t kV2CombineStateOffset = kV2CombineSelectorOffset + kV2SelectorMetadataSize;
constexpr uint64_t kDataOffset = kV2CombineStateOffset + kV2StateSize;
constexpr uint64_t kPerHalfReservedSize = kDataOffset;

static_assert(kV2StateTimeoutOffset + kV2StateTimeoutBytes <= kV2StateSize,
              "V2 timeout probe must remain inside its state slot");
static_assert(kV2MaxBs * (kV2MaxTopK + kV2MaxSharedExpertNum) * kV2StateEntrySize <= kV2StateSize,
              "V2 combine state must remain inside its state slot");
static_assert(kV2MaxBs * (kV2MaxTopK + kV2MaxSharedExpertNum) * kV2StateEntrySize <= kV2StateTimeoutOffset,
              "V2 combine state must not overlap the timeout probe");
}  // namespace A3WindowLayout
}  // namespace Moe

#endif  // TILING_ARGS_H
