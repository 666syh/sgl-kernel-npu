/*
 * SPDX-License-Identifier: MIT
 */

#include "profiling/core/profile_runtime.hpp"

#include "profiling/core/profile_session.hpp"

namespace deep_ep::profiling::runtime {

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
    ctx.enabled = profileEnable && ctx.sessionActive;
    ctx.registration = &registration;
    if (!ctx.enabled) {
        return ctx;
    }
    session::EnsureOpSession(registration, launchConfig);
    TORCH_CHECK(session::GetProfileBuffer(registration.opKey).defined() &&
                    session::GetProfileBuffer(registration.opKey).numel() > 0,
                schema.opName ? schema.opName : "profile", " session is active but profile buffer is missing.");
    if (session::GetCapturedLaunches(registration.opKey) >= session::GetExpectedLaunches()) {
        session::IncrementDroppedLaunches(registration.opKey);
        ctx.enabled = false;
        return ctx;
    }
    ctx.launchId = session::GetCapturedLaunches(registration.opKey);
    ctx.profileBuffer = &session::GetProfileBuffer(registration.opKey);
    ctx.profileBufferBytes = static_cast<int64_t>(session::GetProfileBufferBytes(registration.opKey));
    return ctx;
}

void CompleteLaunch(const ProfileLaunchContext &ctx, int64_t rank)
{
    if (!ctx.enabled) {
        return;
    }
    TORCH_CHECK(ctx.registration != nullptr && ctx.registration->schemaProvider != nullptr,
                "profile launch context registration is missing.");
    session::IncrementCapturedLaunches(ctx.registration->opKey);
}

}  // namespace deep_ep::profiling::runtime
