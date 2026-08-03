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

std::string GetProfileTraceDir()
{
    return session::GetProfileTraceDir();
}

int64_t GetNumProfileSkipLaunches()
{
    return session::GetNumProfileSkipLaunches();
}

int64_t GetNumProfileActiveLaunches()
{
    return session::GetNumProfileActiveLaunches();
}

int64_t GetExpectedLaunches()
{
    return session::GetExpectedLaunches();
}

void BeginSession(int64_t numProfileSkipLaunches, int64_t numProfileActiveLaunches, const std::string &profileTraceDir)
{
    session::Begin(numProfileSkipLaunches, numProfileActiveLaunches, profileTraceDir);
}

void EndSession(int64_t rank)
{
    session::ExportAndReset(rank);
}

ProfileLaunchContext PrepareLaunch(const ProfileOpRegistration &registration, const ProfileLaunchConfig &launchConfig,
                                   bool profileEnable)
{
    const auto &schema = registration.schemaProvider();
    ProfileLaunchContext ctx{};
    ctx.sessionActive = session::IsActive();
    ctx.enabled = profileEnable || ctx.sessionActive;
    ctx.registration = &registration;
    if (!ctx.enabled) {
        return ctx;
    }
    if (ctx.sessionActive) {
        session::EnsureOpSession(registration, launchConfig);
        TORCH_CHECK(session::GetProfileBuffer(registration.opKey).defined() &&
                        session::GetProfileBuffer(registration.opKey).numel() > 0,
                    schema.opName ? schema.opName : "profile", " session is active but profile buffer is missing.");
        if (session::GetCapturedLaunches(registration.opKey) >= session::GetExpectedLaunches()) {
            session::IncrementDroppedLaunches(registration.opKey);
            ctx.enabled = false;
            if (debug::IsEnabled()) {
                std::ostringstream oss;
                oss << "drop extra launch: op=" << (schema.opName ? schema.opName : "profile")
                    << ", captured=" << session::GetCapturedLaunches(registration.opKey)
                    << ", expected=" << session::GetExpectedLaunches()
                    << ", dropped=" << session::GetDroppedLaunches(registration.opKey);
                debug::Print(oss.str());
            }
            return ctx;
        }
        ctx.launchId = session::GetCapturedLaunches(registration.opKey);
        ctx.profileBuffer = &session::GetProfileBuffer(registration.opKey);
        ctx.profileBufferBytes = static_cast<int64_t>(session::GetProfileBufferBytes(registration.opKey));
        return ctx;
    }

    uint64_t perLaunchBytes = session::GetPerLaunchBytes(launchConfig.stageLayout, schema);
    uint64_t totalBytes = session::GetTotalBytes(1, perLaunchBytes);
    ctx.ownedProfileBuffer =
        session::AllocateBuffer(totalBytes, 1U, launchConfig.groupCountCapacity, launchConfig.stageLayout, schema);
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

void CompleteLaunch(const ProfileLaunchContext &ctx, int64_t rank)
{
    if (!ctx.enabled) {
        return;
    }
    TORCH_CHECK(ctx.registration != nullptr && ctx.registration->schemaProvider != nullptr,
                "profile launch context registration is missing.");
    const auto &schema = ctx.registration->schemaProvider();
    if (ctx.sessionActive) {
        session::IncrementCapturedLaunches(ctx.registration->opKey);
        if (debug::IsEnabled()) {
            std::ostringstream oss;
            oss << "captured launch=" << (session::GetCapturedLaunches(ctx.registration->opKey) - 1)
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
