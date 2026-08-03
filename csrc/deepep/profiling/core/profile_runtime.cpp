/*
 * SPDX-License-Identifier: MIT
 */

#include "profiling/core/profile_runtime.hpp"

#include <sstream>

#include "profiling/core/profile_debug.hpp"
#include "profiling/core/profile_exporter.hpp"
#include "profiling/core/profile_session.hpp"

namespace deep_ep::profiling::runtime {

bool IsDebugEnabled()
{
    return debug::IsEnabled();
}

void DebugPrint(const std::string &msg)
{
    debug::Print(msg);
}

bool IsSessionActive()
{
    return session::IsActive();
}

void BeginSession(const ProfileSchema &schema, int64_t numProfileSkipLaunches, int64_t numProfileActiveLaunches,
                  const std::string &profileTraceDir)
{
    session::Begin(schema, numProfileSkipLaunches, numProfileActiveLaunches, profileTraceDir);
}

void EndSession(int64_t rank)
{
    session::ExportAndReset(rank);
}

ProfileLaunchContext PrepareLaunch(bool profileEnable, uint32_t groupCountCapacity,
                                   const Cam::ProfileStageLayout &stageLayout, const ProfileSchema &schema)
{
    ProfileLaunchContext ctx{};
    ctx.sessionActive = session::IsActive();
    ctx.enabled = profileEnable || ctx.sessionActive;
    if (!ctx.enabled) {
        return ctx;
    }
    if (ctx.sessionActive) {
        session::EnsureInitialized(groupCountCapacity, stageLayout);
        TORCH_CHECK(session::GetProfileBuffer().defined() && session::GetProfileBuffer().numel() > 0,
                    schema.opName ? schema.opName : "profile", " session is active but profile buffer is missing.");
        TORCH_CHECK(session::GetCapturedLaunches() < session::GetExpectedLaunches(),
                    schema.opName ? schema.opName : "profile", " session expected ", session::GetExpectedLaunches(),
                    " launches but received ", session::GetCapturedLaunches(), ".");
        ctx.launchId = session::GetCapturedLaunches();
        ctx.profileBuffer = &session::GetProfileBuffer();
        ctx.profileBufferBytes = static_cast<int64_t>(session::GetProfileBufferBytes());
        return ctx;
    }

    uint64_t perLaunchBytes = session::GetPerLaunchBytes(stageLayout, schema);
    uint64_t totalBytes = session::GetTotalBytes(1, perLaunchBytes);
    ctx.ownedProfileBuffer = session::AllocateBuffer(totalBytes, 1U, groupCountCapacity, stageLayout, schema);
    ctx.profileBuffer = &ctx.ownedProfileBuffer;
    ctx.profileBufferBytes = static_cast<int64_t>(ctx.ownedProfileBuffer.numel());
    ctx.launchId = 0;
    if (debug::IsEnabled()) {
        std::ostringstream oss;
        oss << "prepare one-shot launch: op=" << (schema.opName ? schema.opName : "profile")
            << ", bytes=" << ctx.profileBufferBytes << ", trace_dir=" << session::GetProfileTraceDir();
        debug::Print(oss.str());
    }
    return ctx;
}

void CompleteLaunch(const ProfileLaunchContext &ctx, int64_t rank, const ProfileSchema &schema)
{
    if (!ctx.enabled) {
        return;
    }
    if (ctx.sessionActive) {
        session::IncrementCapturedLaunches();
        if (debug::IsEnabled()) {
            std::ostringstream oss;
            oss << "captured launch=" << (session::GetCapturedLaunches() - 1)
                << ", op=" << (schema.opName ? schema.opName : "profile")
                << ", trace_dir=" << session::GetProfileTraceDir();
            debug::Print(oss.str());
        }
        return;
    }

    TORCH_CHECK(ctx.profileBuffer != nullptr && ctx.profileBuffer->defined(), schema.opName ? schema.opName : "profile",
                " one-shot profile buffer is missing.");
    exporter::ExportBufferToTrace(*ctx.profileBuffer, rank, session::GetProfileTraceDir(), 0, 1, schema);
}

}  // namespace deep_ep::profiling::runtime
