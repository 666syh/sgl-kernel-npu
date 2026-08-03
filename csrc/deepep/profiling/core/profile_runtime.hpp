/*
 * SPDX-License-Identifier: MIT
 */
#ifndef DEEPEP_PROFILING_CORE_PROFILE_RUNTIME_HPP
#define DEEPEP_PROFILING_CORE_PROFILE_RUNTIME_HPP

#include <cstdint>
#include <string>

#include <torch/types.h>

#include "profiling/common/profile_protocol_common.h"
#include "profiling/core/profile_schema.hpp"

namespace deep_ep::profiling::runtime {

struct ProfileLaunchContext {
    bool enabled{false};
    bool sessionActive{false};
    int64_t launchId{0};
    int64_t profileBufferBytes{0};
    const at::Tensor *profileBuffer{nullptr};
    at::Tensor ownedProfileBuffer;
};

bool IsDebugEnabled();
void DebugPrint(const std::string &msg);
bool IsSessionActive();

void BeginSession(const ProfileSchema &schema, int64_t numWarmups, int64_t numTests,
                  const std::string &profileTraceDir);
void EndSession(int64_t rank);

ProfileLaunchContext PrepareLaunch(bool profileEnable, const std::string &profileTraceDir, uint32_t groupCountCapacity,
                                   const Cam::ProfileStageLayout &stageLayout, const ProfileSchema &schema);
void CompleteLaunch(const ProfileLaunchContext &ctx, int64_t rank, const std::string &profileTraceDir,
                    const ProfileSchema &schema);

}  // namespace deep_ep::profiling::runtime

#endif  // DEEPEP_PROFILING_CORE_PROFILE_RUNTIME_HPP
