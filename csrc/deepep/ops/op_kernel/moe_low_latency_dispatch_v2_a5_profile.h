#ifndef MOE_LOW_LATENCY_DISPATCH_V2_A5_PROFILE_H
#define MOE_LOW_LATENCY_DISPATCH_V2_A5_PROFILE_H

#include "profiling/kernel/profile_writer_kernel.h"
#include "profiling/adapters/moe_low_latency_dispatch_v2_a5/moe_low_latency_dispatch_v2_a5_profile_stage.h"

namespace Cam {

using MoeLowLatencyDispatchV2A5ProfileStage = deep_ep::profiling::moe_low_latency_dispatch_v2_a5::ProfileStage;

struct MoeLowLatencyDispatchV2A5ProfileWriter : public ProfileWriter {
    __aicore__ inline void Init(GM_ADDR profileGM, bool enable, uint32_t launchId_, uint32_t coreType_,
                                uint64_t profileBufferBytes_)
    {
        ProfileWriter::Init(
            profileGM, enable, launchId_, coreType_,
            static_cast<uint32_t>(deep_ep::profiling::moe_low_latency_dispatch_v2_a5::ProfileStage::Count),
            profileBufferBytes_);
    }

    __aicore__ inline void Record(deep_ep::profiling::moe_low_latency_dispatch_v2_a5::ProfileStage stage,
                                  uint64_t startCycle, uint64_t endCycle) const
    {
        ProfileWriter::Record(static_cast<uint32_t>(stage), startCycle, endCycle);
    }
};

}  // namespace Cam

#endif  // MOE_LOW_LATENCY_DISPATCH_V2_A5_PROFILE_H
