/*
 * SPDX-License-Identifier: MIT
 */

#include "profiling/core/profile_session.hpp"

#include <cstring>
#include <sstream>

#include "exception.hpp"
#include "profiling/core/profile_debug.hpp"
#include "profiling/core/profile_exporter.hpp"
#include "pytorch_npu_helper.hpp"

namespace deep_ep::profiling::session {
namespace {

constexpr uint64_t kMaxBytesPerRank = 128ULL * 1024ULL * 1024ULL;

SessionState &GetSessionImpl()
{
    static SessionState session;
    return session;
}

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

Cam::ProfileHeader BuildHeader(uint32_t launchCountCapacity, uint32_t groupCountCapacity,
                               const Cam::ProfileStageLayout &stageLayout, const ProfileSchema &schema)
{
    Cam::ProfileHeader header{};
    header.magic = Cam::PROFILE_MAGIC;
    header.version = Cam::PROFILE_VERSION;
    header.cycleToUs = Cam::PROFILE_CYCLE_TO_US;
    header.launchCountsPacked = Cam::PackProfileLaunchCounts(launchCountCapacity, 0U);
    header.layoutPacked0 = Cam::PackProfileLayout0(static_cast<uint16_t>(schema.stageCount),
                                                   static_cast<uint16_t>(groupCountCapacity), 1U, 2U);
    header.stageOccurrencesPacked = 0;
    header.layoutPacked1 =
        Cam::PackProfileLayout1(schema.topology.logicalCoreCount,
                                Cam::GetProfileRecordsPerLaunch(schema.topology.logicalCoreCount, stageLayout));
    header.flagsPacked = Cam::PackProfileFlags(static_cast<uint32_t>(Cam::PROFILE_FLAG_SESSION_BUFFER), 0U);
    return header;
}

void CopyHeaderToDevice(const at::Tensor &profileBuffer, const Cam::ProfileHeader &header,
                        const Cam::ProfileStageLayout &stageLayout)
{
    TORCH_CHECK(
        profileBuffer.defined() &&
            profileBuffer.numel() >= static_cast<int64_t>(sizeof(Cam::ProfileHeader) + sizeof(Cam::ProfileStageLayout)),
        "profile buffer is not large enough for profile metadata.");
    auto headerCpu =
        at::empty({static_cast<int64_t>(sizeof(header))}, at::TensorOptions().device(at::kCPU).dtype(c10::kByte));
    std::memcpy(headerCpu.data_ptr<uint8_t>(), &header, sizeof(header));
    profileBuffer.narrow(0, 0, static_cast<int64_t>(sizeof(header))).copy_(headerCpu);
    auto layoutCpu =
        at::empty({static_cast<int64_t>(sizeof(stageLayout))}, at::TensorOptions().device(at::kCPU).dtype(c10::kByte));
    std::memcpy(layoutCpu.data_ptr<uint8_t>(), &stageLayout, sizeof(stageLayout));
    profileBuffer.narrow(0, static_cast<int64_t>(sizeof(Cam::ProfileHeader)), static_cast<int64_t>(sizeof(stageLayout)))
        .copy_(layoutCpu);
}

}  // namespace

void SessionState::Reset()
{
    active = false;
    initialized = false;
    numProfileSkipLaunches = 0;
    numProfileActiveLaunches = 0;
    expectedLaunches = 0;
    capturedLaunches = 0;
    droppedLaunches = 0;
    groupCountCapacity = 0;
    stageLayout = Cam::ProfileStageLayout{};
    recordsPerLaunch = 0;
    profileTraceDir.clear();
    profileBuffer = at::Tensor();
    profileBufferBytes = 0;
    perLaunchBytes = 0;
    launchCountCapacity = 0;
    schema = nullptr;
}

bool IsActive()
{
    return GetSessionImpl().active;
}
bool IsInitialized()
{
    return GetSessionImpl().initialized;
}
int64_t GetExpectedLaunches()
{
    return GetSessionImpl().expectedLaunches;
}
int64_t GetCapturedLaunches()
{
    return GetSessionImpl().capturedLaunches;
}
int64_t GetDroppedLaunches()
{
    return GetSessionImpl().droppedLaunches;
}
uint32_t GetLaunchCountCapacity()
{
    return GetSessionImpl().launchCountCapacity;
}
uint64_t GetProfileBufferBytes()
{
    return GetSessionImpl().profileBufferBytes;
}
std::string GetProfileTraceDir()
{
    return GetSessionImpl().profileTraceDir;
}
const at::Tensor &GetProfileBuffer()
{
    return GetSessionImpl().profileBuffer;
}
int64_t GetNumProfileSkipLaunches()
{
    return GetSessionImpl().numProfileSkipLaunches;
}
int64_t GetNumProfileActiveLaunches()
{
    return GetSessionImpl().numProfileActiveLaunches;
}
uint32_t GetGroupCountCapacity()
{
    return GetSessionImpl().groupCountCapacity;
}
const Cam::ProfileStageLayout &GetStageLayout()
{
    return GetSessionImpl().stageLayout;
}
const ProfileSchema &GetSchema()
{
    return *GetSessionImpl().schema;
}

uint32_t GetRecordsPerLaunch(const Cam::ProfileStageLayout &stageLayout, const ProfileSchema &schema)
{
    return Cam::GetProfileRecordsPerLaunch(schema.topology.logicalCoreCount, stageLayout);
}

uint64_t GetPerLaunchBytes(const Cam::ProfileStageLayout &stageLayout, const ProfileSchema &schema)
{
    return static_cast<uint64_t>(GetRecordsPerLaunch(stageLayout, schema)) * sizeof(Cam::ProfileRecord);
}

uint32_t GetRecordsPerLaunch(const Cam::ProfileStageLayout &stageLayout)
{
    return GetRecordsPerLaunch(stageLayout, GetSchema());
}

uint64_t GetPerLaunchBytes(const Cam::ProfileStageLayout &stageLayout)
{
    return GetPerLaunchBytes(stageLayout, GetSchema());
}

uint64_t GetTotalBytes(uint64_t launchCountCapacity, uint64_t perLaunchBytes)
{
    return Cam::GetProfileDataOffset() + launchCountCapacity * AlignUp(perLaunchBytes, 64ULL);
}

at::Tensor AllocateBuffer(uint64_t totalBytes, uint32_t launchCountCapacity, uint32_t groupCountCapacity,
                          const Cam::ProfileStageLayout &stageLayout, const ProfileSchema &schema)
{
    at::TensorOptions options = at::TensorOptions(torch_npu::utils::get_npu_device_type());
    auto profileBuffer = at::zeros({static_cast<int64_t>(totalBytes)}, options.dtype(c10::kByte));
    auto header = BuildHeader(launchCountCapacity, groupCountCapacity, stageLayout, schema);
    header.layoutPacked1 =
        Cam::PackProfileLayout1(schema.topology.logicalCoreCount, GetRecordsPerLaunch(stageLayout, schema));
    CopyHeaderToDevice(profileBuffer, header, stageLayout);
    return profileBuffer;
}

void Begin(const ProfileSchema &schema, int64_t numProfileSkipLaunches, int64_t numProfileActiveLaunches,
           const std::string &profileTraceDir)
{
    TORCH_CHECK(numProfileSkipLaunches >= 0, "num_profile_skip_launches must be non-negative");
    TORCH_CHECK(numProfileActiveLaunches >= 0, "num_profile_active_launches must be non-negative");
    TORCH_CHECK(!GetSessionImpl().active, "profile session is already active.");
    int64_t expectedLaunches = numProfileSkipLaunches + numProfileActiveLaunches;
    TORCH_CHECK(expectedLaunches > 0, "profile session needs at least one launch.");
    TORCH_CHECK(static_cast<uint64_t>(expectedLaunches) <= UINT32_MAX,
                "profile session launch count exceeds uint32 capacity.");
    auto &session = GetSessionImpl();
    session.Reset();
    session.active = true;
    session.numProfileSkipLaunches = numProfileSkipLaunches;
    session.numProfileActiveLaunches = numProfileActiveLaunches;
    session.expectedLaunches = expectedLaunches;
    session.profileTraceDir = profileTraceDir;
    session.launchCountCapacity = static_cast<uint32_t>(expectedLaunches);
    session.schema = &schema;
    if (debug::IsEnabled()) {
        std::ostringstream oss;
        oss << "begin session: op=" << (schema.opName ? schema.opName : "profile")
            << ", num_profile_skip_launches=" << numProfileSkipLaunches
            << ", num_profile_active_launches=" << numProfileActiveLaunches
            << ", launch_capacity=" << session.launchCountCapacity << ", trace_dir=" << profileTraceDir;
        debug::Print(oss.str());
    }
}

void EnsureInitialized(uint32_t groupCountCapacity, const Cam::ProfileStageLayout &stageLayout)
{
    auto &session = GetSessionImpl();
    TORCH_CHECK(session.active, "profile session is not active.");
    TORCH_CHECK(session.launchCountCapacity > 0, "profile session launch capacity is not set.");
    TORCH_CHECK(groupCountCapacity >= 1U && groupCountCapacity <= Cam::PROFILE_MAX_GROUP_COUNT_CAPACITY,
                "profile session group count exceeds occurrence capacity: groupCountCapacity=", groupCountCapacity,
                ", valid_range=[1, ", Cam::PROFILE_MAX_GROUP_COUNT_CAPACITY, "]");
    if (session.initialized) {
        TORCH_CHECK(session.groupCountCapacity == groupCountCapacity,
                    "profile session group count changed within one session: expected ", session.groupCountCapacity,
                    ", got ", groupCountCapacity, ".");
        return;
    }
    session.groupCountCapacity = groupCountCapacity;
    session.stageLayout = stageLayout;
    session.recordsPerLaunch = GetRecordsPerLaunch(stageLayout);
    session.perLaunchBytes = GetPerLaunchBytes(stageLayout);
    uint64_t totalBytes = GetTotalBytes(static_cast<uint64_t>(session.launchCountCapacity), session.perLaunchBytes);
    TORCH_CHECK(totalBytes <= kMaxBytesPerRank, "profile buffer would exceed the per-rank cap: totalBytes=", totalBytes,
                ", cap=", kMaxBytesPerRank);
    session.profileBufferBytes = totalBytes;
    session.profileBuffer = AllocateBuffer(totalBytes, session.launchCountCapacity, session.groupCountCapacity,
                                           session.stageLayout, GetSchema());
    session.initialized = true;
}

void IncrementCapturedLaunches()
{
    ++GetSessionImpl().capturedLaunches;
}

void IncrementDroppedLaunches()
{
    ++GetSessionImpl().droppedLaunches;
}

void ExportAndReset(int64_t rank)
{
    auto &session = GetSessionImpl();
    if (!session.active) {
        if (debug::IsEnabled()) {
            debug::Print("end session ignored: inactive");
        }
        return;
    }
    if (session.capturedLaunches != session.expectedLaunches || session.droppedLaunches > 0) {
        TORCH_WARN("profile session captured ", session.capturedLaunches, " launches, expected ",
                   session.expectedLaunches, ", dropped extra launches=", session.droppedLaunches, ".");
    }
    if (debug::IsEnabled()) {
        std::ostringstream oss;
        oss << "end session: launches=" << session.capturedLaunches << ", expected=" << session.expectedLaunches
            << ", dropped=" << session.droppedLaunches << ", trace_dir=" << session.profileTraceDir;
        debug::Print(oss.str());
    }
    exporter::ExportBufferToTrace(session.profileBuffer, rank, session.profileTraceDir, session.numProfileSkipLaunches,
                                  session.capturedLaunches, *session.schema);
    session.Reset();
}

}  // namespace deep_ep::profiling::session
