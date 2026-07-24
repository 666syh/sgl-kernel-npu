/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */
#ifndef DEEPEP_PROFILE_TRACE_HPP
#define DEEPEP_PROFILE_TRACE_HPP

#include <cstdint>
#include <optional>
#include <string>

#include <torch/types.h>

#include "ops/utils/common/fused_deep_moe_profile_common.h"

namespace deep_ep::profile_trace {

bool IsDebugEnabled();
void DebugPrint(const std::string &msg);

uint32_t GetGroupCountCapacity(int64_t numExperts, int64_t numRanks);
uint64_t BuildStageOccurrencesPacked(uint32_t groupCountCapacity);
uint64_t GetPerLaunchBytes(uint64_t stageOccurrencesPacked);
uint64_t GetTotalBytes(uint64_t launchCountCapacity, uint64_t perLaunchBytes);
at::Tensor AllocateBuffer(uint64_t totalBytes, uint32_t launchCountCapacity, uint32_t groupCountCapacity);

bool IsActive();
bool IsInitialized();
int64_t GetExpectedLaunches();
int64_t GetCapturedLaunches();
uint32_t GetLaunchCountCapacity();
uint64_t GetProfileBufferBytes();
std::string GetProfileTraceDir();
const at::Tensor &GetProfileBuffer();
int64_t GetNumWarmups();
int64_t GetNumTests();

void Begin(int64_t numWarmups, int64_t numTests, const std::string &profileTraceDir);
void EnsureInitialized(int64_t numExperts, int64_t numRanks);
void IncrementCapturedLaunches();
void ExportBufferToTrace(const at::Tensor &profileBuffer, int64_t rank, const std::string &profileTraceDir,
                         int64_t numWarmups, int64_t launchCountCaptured);
void ExportAndReset(int64_t rank);

}  // namespace deep_ep::profile_trace

#endif  // DEEPEP_PROFILE_TRACE_HPP
