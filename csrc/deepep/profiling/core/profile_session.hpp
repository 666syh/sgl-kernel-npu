/*
 * SPDX-License-Identifier: MIT
 */
#ifndef DEEPEP_PROFILING_CORE_PROFILE_SESSION_HPP
#define DEEPEP_PROFILING_CORE_PROFILE_SESSION_HPP

#include <cstdint>
#include <string>

#include <torch/types.h>

#include "profiling/common/profile_protocol_common.h"
#include "profiling/core/profile_schema.hpp"

namespace deep_ep::profiling::session {

struct SessionState {
    bool active{false};
    bool initialized{false};
    int64_t numProfileSkipLaunches{0};
    int64_t numProfileActiveLaunches{0};
    int64_t expectedLaunches{0};
    int64_t capturedLaunches{0};
    int64_t droppedLaunches{0};
    uint32_t groupCountCapacity{0};
    Cam::ProfileStageLayout stageLayout{};
    uint32_t recordsPerLaunch{0};
    std::string profileTraceDir;
    at::Tensor profileBuffer;
    uint64_t profileBufferBytes{0};
    uint64_t perLaunchBytes{0};
    uint32_t launchCountCapacity{0};
    const ProfileSchema *schema{nullptr};

    void Reset();
};

bool IsActive();
bool IsInitialized();
int64_t GetExpectedLaunches();
int64_t GetCapturedLaunches();
int64_t GetDroppedLaunches();
uint32_t GetLaunchCountCapacity();
uint64_t GetProfileBufferBytes();
std::string GetProfileTraceDir();
const at::Tensor &GetProfileBuffer();
int64_t GetNumProfileSkipLaunches();
int64_t GetNumProfileActiveLaunches();
uint32_t GetGroupCountCapacity();
const Cam::ProfileStageLayout &GetStageLayout();
const ProfileSchema &GetSchema();

uint32_t GetRecordsPerLaunch(const Cam::ProfileStageLayout &stageLayout, const ProfileSchema &schema);
uint64_t GetPerLaunchBytes(const Cam::ProfileStageLayout &stageLayout, const ProfileSchema &schema);
uint32_t GetRecordsPerLaunch(const Cam::ProfileStageLayout &stageLayout);
uint64_t GetPerLaunchBytes(const Cam::ProfileStageLayout &stageLayout);
uint64_t GetTotalBytes(uint64_t launchCountCapacity, uint64_t perLaunchBytes);
at::Tensor AllocateBuffer(uint64_t totalBytes, uint32_t launchCountCapacity, uint32_t groupCountCapacity,
                          const Cam::ProfileStageLayout &stageLayout, const ProfileSchema &schema);

void Begin(const ProfileSchema &schema, int64_t numProfileSkipLaunches, int64_t numProfileActiveLaunches,
           const std::string &profileTraceDir);
void EnsureInitialized(uint32_t groupCountCapacity, const Cam::ProfileStageLayout &stageLayout);
void IncrementCapturedLaunches();
void IncrementDroppedLaunches();
void ExportAndReset(int64_t rank);

}  // namespace deep_ep::profiling::session

#endif  // DEEPEP_PROFILING_CORE_PROFILE_SESSION_HPP
