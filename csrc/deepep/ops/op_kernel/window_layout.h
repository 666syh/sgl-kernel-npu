#ifndef WINDOW_LAYOUT_H
#define WINDOW_LAYOUT_H

#include <cstdint>

namespace Moe {
namespace A3WindowLayout {
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
constexpr uint64_t kV2MaxBs = 256UL;
constexpr uint64_t kV2MaxTopK = 16UL;
constexpr uint64_t kV2MaxSharedExpertNum = 4UL;

constexpr uint64_t kV2DispatchSelectorOffset = kNotifyDispatchSize + kNormalCombineStateSize;
constexpr uint64_t kV2DispatchStateOffset = kV2DispatchSelectorOffset + kV2SelectorMetadataSize;
constexpr uint64_t kV2CombineSelectorOffset = kV2DispatchStateOffset + kV2StateSize;
constexpr uint64_t kV2CombineStateOffset = kV2CombineSelectorOffset + kV2SelectorMetadataSize;
constexpr uint64_t kDataOffset = kV2CombineStateOffset + kV2StateSize;
constexpr uint64_t kPerHalfReservedSize = kDataOffset;

static_assert(kV2StateTimeoutOffset + kV2StateTimeoutBytes <= kV2StateSize,
              "V2 timeout probe must remain inside its state slot");
}  // namespace A3WindowLayout
}  // namespace Moe

#endif  // WINDOW_LAYOUT_H
