/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */

#include "profile_trace.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <vector>

#include "exception.hpp"
#include "pytorch_npu_helper.hpp"

namespace deep_ep::profile_trace {
namespace {

constexpr uint64_t kCoreTypeAic = Cam::FUSED_DEEP_MOE_PROFILE_CORE_TYPE_AIC;
constexpr uint64_t kCoreTypeAiv = Cam::FUSED_DEEP_MOE_PROFILE_CORE_TYPE_AIV;
constexpr uint64_t kMaxBytesPerRank = 128ULL * 1024ULL * 1024ULL;
constexpr uint64_t kHeaderBytes = sizeof(Cam::FusedDeepMoeProfileHeader);
constexpr uint64_t kStageLayoutBytes = sizeof(Cam::FusedDeepMoeProfileStageLayout);
constexpr uint64_t kRecordBytes = sizeof(Cam::FusedDeepMoeProfileRecord);

struct TraceEventRow {
    double ts_us{0.0};
    double dur_us{0.0};
    int64_t launchId{0};
    bool isWarmup{false};
    uint64_t coreType{0};
    uint64_t coreIdx{0};
    uint64_t stageId{0};
    uint64_t occurrenceId{0};
    uint64_t startCycle{0};
    uint64_t endCycle{0};
};

struct Session {
    bool active{false};
    bool initialized{false};
    int64_t numWarmups{0};
    int64_t numTests{0};
    int64_t expectedLaunches{0};
    int64_t capturedLaunches{0};
    uint32_t groupCountCapacity{0};
    Cam::FusedDeepMoeProfileStageLayout stageLayout{};
    uint32_t recordsPerLaunch{0};
    std::string profileTraceDir;
    at::Tensor profileBuffer;
    uint64_t profileBufferBytes{0};
    uint64_t perLaunchBytes{0};
    uint32_t launchCountCapacity{0};

    void Reset()
    {
        active = false;
        initialized = false;
        numWarmups = 0;
        numTests = 0;
        expectedLaunches = 0;
        capturedLaunches = 0;
        groupCountCapacity = 0;
        stageLayout = Cam::FusedDeepMoeProfileStageLayout{};
        recordsPerLaunch = 0;
        profileTraceDir.clear();
        profileBuffer = at::Tensor();
        profileBufferBytes = 0;
        perLaunchBytes = 0;
        launchCountCapacity = 0;
    }
};

Session &GetSessionImpl()
{
    static Session session;
    return session;
}

static const char *StageName(uint64_t stageId)
{
    switch (static_cast<Cam::FusedDeepMoeProfileStage>(stageId)) {
        case Cam::FusedDeepMoeProfileStage::Dispatch:
            return "dispatch";
        case Cam::FusedDeepMoeProfileStage::Gmm1:
            return "gmm1";
        case Cam::FusedDeepMoeProfileStage::SwigluQuant:
            return "swiglu_quant";
        case Cam::FusedDeepMoeProfileStage::Gmm2:
            return "gmm2";
        case Cam::FusedDeepMoeProfileStage::Combine:
            return "combine";
        case Cam::FusedDeepMoeProfileStage::WeightSum:
            return "weight_sum";
        default:
            return "unknown";
    }
}

static std::string StageDisplayName(uint64_t stageId, uint64_t occurrenceId,
                                    const Cam::FusedDeepMoeProfileStageLayout &stageLayout)
{
    std::ostringstream oss;
    oss << StageName(stageId);
    uint32_t stageOccurrenceCount = Cam::GetProfileStageOccurrenceCount(stageLayout, static_cast<uint32_t>(stageId));
    auto stage = static_cast<Cam::FusedDeepMoeProfileStage>(stageId);
    if (stage == Cam::FusedDeepMoeProfileStage::Dispatch || stage == Cam::FusedDeepMoeProfileStage::Gmm1 ||
        stage == Cam::FusedDeepMoeProfileStage::Gmm2 || stage == Cam::FusedDeepMoeProfileStage::Combine) {
        oss << "[group=" << occurrenceId << "]";
    } else if (stageOccurrenceCount > 1U || occurrenceId != 0U) {
        oss << "[occ=" << occurrenceId << "]";
    }
    return oss.str();
}

static std::string CoreTypeName(uint64_t coreType)
{
    return (coreType == kCoreTypeAiv) ? "AIV" : "AIC";
}

static std::string JsonEscape(const std::string &value)
{
    std::ostringstream oss;
    oss << '"';
    for (char c : value) {
        switch (c) {
            case '\\':
                oss << "\\\\";
                break;
            case '"':
                oss << "\\\"";
                break;
            case '\b':
                oss << "\\b";
                break;
            case '\f':
                oss << "\\f";
                break;
            case '\n':
                oss << "\\n";
                break;
            case '\r':
                oss << "\\r";
                break;
            case '\t':
                oss << "\\t";
                break;
            default:
                if (static_cast<unsigned char>(c) < 0x20U) {
                    oss << "\\u";
                    oss << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(c)) << std::dec << std::setfill(' ');
                } else {
                    oss << c;
                }
                break;
        }
    }
    oss << '"';
    return oss.str();
}

static std::string ResolveProfileTraceDir(const std::string &profileTraceDir)
{
    if (!profileTraceDir.empty()) {
        return profileTraceDir;
    }
    const char *envTraceDir = std::getenv("DEEPEP_FUSED_PROFILE_DIR");
    if (envTraceDir != nullptr && envTraceDir[0] != '\0') {
        return std::string(envTraceDir);
    }
    return std::filesystem::current_path().string();
}

static uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

static Cam::FusedDeepMoeProfileHeader BuildHeader(uint32_t launchCountCapacity, uint32_t groupCountCapacity,
                                                  const Cam::FusedDeepMoeProfileStageLayout &stageLayout)
{
    Cam::FusedDeepMoeProfileHeader header{};
    header.magic = Cam::FUSED_DEEP_MOE_PROFILE_MAGIC;
    header.version = Cam::FUSED_DEEP_MOE_PROFILE_VERSION;
    header.cycleToUs = Cam::FUSED_DEEP_MOE_PROFILE_CYCLE_TO_US;
    header.launchCountsPacked = Cam::PackProfileLaunchCounts(launchCountCapacity, 0U);
    header.layoutPacked0 = Cam::PackProfileLayout0(static_cast<uint16_t>(Cam::FUSED_DEEP_MOE_PROFILE_STAGE_COUNT),
                                                   static_cast<uint16_t>(groupCountCapacity), 1U, 2U);
    header.stageOccurrencesPacked = 0;
    header.layoutPacked1 =
        Cam::PackProfileLayout1(Cam::FUSED_DEEP_MOE_PROFILE_LOGICAL_CORE_COUNT_CAPACITY,
                                Cam::GetFusedDeepMoeProfileRecordsPerLaunch(
                                    Cam::FUSED_DEEP_MOE_PROFILE_LOGICAL_CORE_COUNT_CAPACITY, stageLayout));
    header.flagsPacked =
        Cam::PackProfileFlags(static_cast<uint32_t>(Cam::FUSED_DEEP_MOE_PROFILE_FLAG_SESSION_BUFFER), 0U);
    return header;
}

static void CopyHeaderToDevice(const at::Tensor &profileBuffer, const Cam::FusedDeepMoeProfileHeader &header,
                               const Cam::FusedDeepMoeProfileStageLayout &stageLayout)
{
    TORCH_CHECK(
        profileBuffer.defined() && profileBuffer.numel() >= static_cast<int64_t>(kHeaderBytes + kStageLayoutBytes),
        "profile buffer is not large enough for profile metadata.");
    auto headerCpu =
        at::empty({static_cast<int64_t>(sizeof(header))}, at::TensorOptions().device(at::kCPU).dtype(c10::kByte));
    std::memcpy(headerCpu.data_ptr<uint8_t>(), &header, sizeof(header));
    profileBuffer.narrow(0, 0, static_cast<int64_t>(sizeof(header))).copy_(headerCpu);
    auto layoutCpu =
        at::empty({static_cast<int64_t>(sizeof(stageLayout))}, at::TensorOptions().device(at::kCPU).dtype(c10::kByte));
    std::memcpy(layoutCpu.data_ptr<uint8_t>(), &stageLayout, sizeof(stageLayout));
    profileBuffer.narrow(0, static_cast<int64_t>(kHeaderBytes), static_cast<int64_t>(sizeof(stageLayout)))
        .copy_(layoutCpu);
}

void ExportBufferToTraceImpl(const at::Tensor &profileBuffer, int64_t rank, const std::string &profileTraceDir,
                             int64_t numWarmups, int64_t launchCountCaptured)
{
    struct LaunchTraceBundle {
        int64_t launchId{0};
        bool isWarmup{false};
        uint64_t minStartCycle{0};
        uint64_t maxEndCycle{0};
        std::vector<TraceEventRow> rows;
    };

    if (!profileBuffer.defined() || profileBuffer.numel() == 0) {
        return;
    }

    auto profileCpu = profileBuffer.to(at::kCPU).contiguous();
    auto *base = profileCpu.data_ptr<uint8_t>();
    auto *header = reinterpret_cast<const Cam::FusedDeepMoeProfileHeader *>(base);
    if (header == nullptr || header->magic != Cam::FUSED_DEEP_MOE_PROFILE_MAGIC ||
        header->version != Cam::FUSED_DEEP_MOE_PROFILE_VERSION || header->cycleToUs == 0) {
        return;
    }
    auto *stageLayout = reinterpret_cast<const Cam::FusedDeepMoeProfileStageLayout *>(base + kHeaderBytes);

    uint32_t launchCountCapacity = Cam::UnpackProfileLaunchCapacity(header->launchCountsPacked);
    uint32_t stageCount = Cam::UnpackProfileStageCount(header->layoutPacked0);
    uint32_t groupCountCapacity = Cam::UnpackProfileGroupCountCapacity(header->layoutPacked0);
    uint32_t logicalCoreCountCapacity = Cam::UnpackProfileLogicalCoreCountCapacity(header->layoutPacked1);
    uint32_t recordsPerLaunch = Cam::UnpackProfileRecordsPerLaunch(header->layoutPacked1);
    uint32_t dispatchOccurrence = Cam::GetProfileStageOccurrenceCount(
        *stageLayout, static_cast<uint32_t>(Cam::FusedDeepMoeProfileStage::Dispatch));
    uint32_t gmm1Occurrence =
        Cam::GetProfileStageOccurrenceCount(*stageLayout, static_cast<uint32_t>(Cam::FusedDeepMoeProfileStage::Gmm1));
    uint32_t swigluOccurrence = Cam::GetProfileStageOccurrenceCount(
        *stageLayout, static_cast<uint32_t>(Cam::FusedDeepMoeProfileStage::SwigluQuant));
    uint32_t gmm2Occurrence =
        Cam::GetProfileStageOccurrenceCount(*stageLayout, static_cast<uint32_t>(Cam::FusedDeepMoeProfileStage::Gmm2));
    uint32_t combineOccurrence = Cam::GetProfileStageOccurrenceCount(
        *stageLayout, static_cast<uint32_t>(Cam::FusedDeepMoeProfileStage::Combine));
    uint32_t weightSumOccurrence = Cam::GetProfileStageOccurrenceCount(
        *stageLayout, static_cast<uint32_t>(Cam::FusedDeepMoeProfileStage::WeightSum));
    uint32_t expectedRecordsPerLaunch =
        Cam::GetFusedDeepMoeProfileRecordsPerLaunch(logicalCoreCountCapacity, *stageLayout);
    if (stageCount == 0U || stageCount > Cam::FUSED_DEEP_MOE_PROFILE_ACTIVE_STAGE_CAPACITY ||
        stageCount > Cam::FUSED_DEEP_MOE_PROFILE_RESERVED_STAGE_CAPACITY || stageLayout->stageCount != stageCount ||
        stageLayout->activeStageCapacity != Cam::FUSED_DEEP_MOE_PROFILE_ACTIVE_STAGE_CAPACITY ||
        groupCountCapacity == 0U || groupCountCapacity > Cam::FUSED_DEEP_MOE_PROFILE_MAX_GROUP_COUNT_CAPACITY ||
        logicalCoreCountCapacity != Cam::FUSED_DEEP_MOE_PROFILE_LOGICAL_CORE_COUNT_CAPACITY ||
        recordsPerLaunch != expectedRecordsPerLaunch || dispatchOccurrence == 0U || gmm1Occurrence == 0U ||
        swigluOccurrence == 0U || gmm2Occurrence == 0U || combineOccurrence == 0U || weightSumOccurrence == 0U) {
        TORCH_WARN("Unexpected fused deep moe profile layout, skip export.");
        return;
    }

    uint64_t perLaunchBytes = static_cast<uint64_t>(recordsPerLaunch) * sizeof(Cam::FusedDeepMoeProfileRecord);
    uint64_t requiredBytes =
        Cam::GetFusedDeepMoeProfileDataOffset() + launchCountCapacity * AlignUp(perLaunchBytes, 64ULL);
    if (profileCpu.numel() < static_cast<int64_t>(requiredBytes)) {
        TORCH_WARN("profile buffer is smaller than expected, skip export.");
        return;
    }

    uint64_t captured = static_cast<uint64_t>(std::max<int64_t>(0, launchCountCaptured));
    captured = std::min<uint64_t>(captured, launchCountCapacity);
    if (captured == 0) {
        return;
    }

    std::vector<LaunchTraceBundle> launches;
    launches.reserve(static_cast<size_t>(captured));
    for (uint64_t launchId = 0; launchId < captured; ++launchId) {
        auto *launchBase = base + Cam::GetFusedDeepMoeProfileLaunchOffset(
                                      Cam::GetFusedDeepMoeProfileDataOffset(), recordsPerLaunch,
                                      static_cast<uint32_t>(sizeof(Cam::FusedDeepMoeProfileRecord)), launchId);
        auto *records = reinterpret_cast<const Cam::FusedDeepMoeProfileRecord *>(launchBase);
        LaunchTraceBundle bundle;
        bundle.launchId = static_cast<int64_t>(launchId);
        bundle.isWarmup = static_cast<int64_t>(launchId) < numWarmups;
        bool haveRange = false;
        for (uint64_t i = 0; i < recordsPerLaunch; ++i) {
            const auto &record = records[i];
            if (record.endCycle <= record.startCycle) {
                continue;
            }
            bundle.rows.push_back({
                static_cast<double>(record.startCycle) / static_cast<double>(header->cycleToUs),
                static_cast<double>(record.endCycle - record.startCycle) / static_cast<double>(header->cycleToUs),
                static_cast<int64_t>(record.launchId),
                static_cast<int64_t>(record.launchId) < numWarmups,
                record.coreType,
                record.coreIdx,
                record.stageId,
                record.occurrenceId,
                record.startCycle,
                record.endCycle,
            });
            if (!haveRange) {
                bundle.minStartCycle = record.startCycle;
                bundle.maxEndCycle = record.endCycle;
                haveRange = true;
            } else {
                bundle.minStartCycle = std::min(bundle.minStartCycle, record.startCycle);
                bundle.maxEndCycle = std::max(bundle.maxEndCycle, record.endCycle);
            }
        }
        if (!bundle.rows.empty()) {
            std::sort(bundle.rows.begin(), bundle.rows.end(), [](const TraceEventRow &lhs, const TraceEventRow &rhs) {
                if (lhs.ts_us != rhs.ts_us) {
                    return lhs.ts_us < rhs.ts_us;
                }
                if (lhs.coreType != rhs.coreType) {
                    return lhs.coreType < rhs.coreType;
                }
                if (lhs.coreIdx != rhs.coreIdx) {
                    return lhs.coreIdx < rhs.coreIdx;
                }
                if (lhs.stageId != rhs.stageId) {
                    return lhs.stageId < rhs.stageId;
                }
                return lhs.occurrenceId < rhs.occurrenceId;
            });
            launches.push_back(std::move(bundle));
        }
    }

    if (launches.empty()) {
        return;
    }

    std::filesystem::path traceDir = std::filesystem::path(ResolveProfileTraceDir(profileTraceDir));
    std::error_code ec;
    std::filesystem::create_directories(traceDir, ec);
    if (ec) {
        TORCH_WARN("Failed to create profile trace dir: ", traceDir.string(), ", error=", ec.message());
        return;
    }

    std::filesystem::path tracePath = traceDir / ("fused_deep_moe_rank" + std::to_string(rank) + ".json");
    std::ofstream ofs(tracePath, std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) {
        TORCH_WARN("Failed to open profile trace file: ", tracePath.string());
        return;
    }

    ofs << "{\n";
    ofs << "  \"traceEvents\": [\n";
    bool needComma = false;

    auto emitEvent = [&](const std::string &name, const std::string &ph, uint64_t pid, uint64_t tid, double ts,
                         double dur, const std::string &argsJson) {
        if (needComma) {
            ofs << ",\n";
        }
        needComma = true;
        ofs << "    {"
            << "\"name\":" << JsonEscape(name) << ","
            << "\"cat\":\"fused_deep_moe\","
            << "\"ph\":" << JsonEscape(ph) << ","
            << "\"ts\":" << std::fixed << std::setprecision(3) << ts << ","
            << "\"pid\":" << pid << ","
            << "\"tid\":" << tid;
        if (ph == "X") {
            ofs << ",\"dur\":" << std::fixed << std::setprecision(3) << dur;
        }
        if (!argsJson.empty()) {
            ofs << ",\"args\":" << argsJson;
        }
        ofs << "}";
    };

    emitEvent("process_name", "M", static_cast<uint64_t>(rank), 0, 0.0, 0.0,
              std::string("{\"name\":") + JsonEscape("rank" + std::to_string(rank)) + "}");

    std::set<uint64_t> seenThreads;
    for (const auto &bundle : launches) {
        uint64_t launchTid = 1000000ULL + static_cast<uint64_t>(bundle.launchId);
        emitEvent("fused_deep_moe_launch", "X", static_cast<uint64_t>(rank), launchTid,
                  static_cast<double>(bundle.minStartCycle) / Cam::FUSED_DEEP_MOE_PROFILE_CYCLE_TO_US,
                  static_cast<double>(bundle.maxEndCycle - bundle.minStartCycle) /
                      static_cast<double>(Cam::FUSED_DEEP_MOE_PROFILE_CYCLE_TO_US),
                  std::string("{\"rank\":") + std::to_string(rank) + ",\"launch_id\":" +
                      std::to_string(bundle.launchId) + ",\"iteration_id\":" + std::to_string(bundle.launchId) +
                      ",\"is_warmup\":" + (bundle.isWarmup ? std::string("true") : std::string("false")) + "}");

        for (const auto &row : bundle.rows) {
            uint64_t tid = Cam::GetFusedDeepMoeProfileLogicalCoreLinear(row.coreType, row.coreIdx);
            if (tid == UINT32_MAX) {
                continue;
            }
            if (seenThreads.insert(tid).second) {
                emitEvent("thread_name", "M", static_cast<uint64_t>(rank), tid, 0.0, 0.0,
                          std::string("{\"name\":") +
                              JsonEscape(CoreTypeName(row.coreType) + "-" + std::to_string(row.coreIdx)) + "}");
            }
        }

        for (const auto &row : bundle.rows) {
            uint64_t tid = Cam::GetFusedDeepMoeProfileLogicalCoreLinear(row.coreType, row.coreIdx);
            if (tid == UINT32_MAX) {
                continue;
            }
            std::ostringstream args;
            args << "{";
            args << "\"rank\":" << rank << ",";
            args << "\"core_type\":" << JsonEscape(CoreTypeName(row.coreType)) << ",";
            args << "\"core_type_raw\":" << row.coreType << ",";
            args << "\"core_idx\":" << row.coreIdx << ",";
            args << "\"stage_id\":" << row.stageId << ",";
            args << "\"stage_name\":" << JsonEscape(StageName(row.stageId)) << ",";
            args << "\"occurrence_id\":" << row.occurrenceId << ",";
            args << "\"stage_label\":" << JsonEscape(StageDisplayName(row.stageId, row.occurrenceId, *stageLayout))
                 << ",";
            args << "\"launch_id\":" << row.launchId << ",";
            args << "\"iteration_id\":" << row.launchId << ",";
            args << "\"is_warmup\":" << (row.isWarmup ? "true" : "false") << ",";
            args << "\"start_cycle\":" << row.startCycle << ",";
            args << "\"end_cycle\":" << row.endCycle;
            args << "}";
            emitEvent(StageDisplayName(row.stageId, row.occurrenceId, *stageLayout), "X", static_cast<uint64_t>(rank),
                      tid, row.ts_us, row.dur_us, args.str());
        }
    }

    ofs << "\n  ]\n}\n";
}

}  // namespace

bool IsDebugEnabled()
{
    const char *env = std::getenv("DEEPEP_FUSED_PROFILE_DEBUG");
    return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

void DebugPrint(const std::string &msg)
{
    if (!IsDebugEnabled()) {
        return;
    }
    std::fprintf(stderr, "[FusedDeepMoeProfileDebug] %s\n", msg.c_str());
}

uint32_t GetGroupCountCapacity(int64_t numExperts, int64_t numRanks)
{
    TORCH_CHECK(numRanks > 0, "num_ranks must be positive for fused deep moe profiling.");
    TORCH_CHECK(numExperts > 0, "num_experts must be positive for fused deep moe profiling.");
    TORCH_CHECK(numExperts % numRanks == 0, "num_experts must be divisible by num_ranks for fused deep moe profiling.");
    return static_cast<uint32_t>(numExperts / numRanks);
}

Cam::FusedDeepMoeProfileStageLayout BuildStageLayout(uint32_t groupCountCapacity)
{
    TORCH_CHECK(groupCountCapacity >= 1U && groupCountCapacity <= Cam::FUSED_DEEP_MOE_PROFILE_MAX_GROUP_COUNT_CAPACITY,
                "groupCountCapacity must be in [1, 64].");
    Cam::FusedDeepMoeProfileStageLayout layout{};
    layout.stageCount = static_cast<uint16_t>(Cam::FUSED_DEEP_MOE_PROFILE_STAGE_COUNT);
    layout.activeStageCapacity = static_cast<uint16_t>(Cam::FUSED_DEEP_MOE_PROFILE_ACTIVE_STAGE_CAPACITY);
    TORCH_CHECK(Cam::SetProfileStageOccurrenceCount(
                    layout, static_cast<uint32_t>(Cam::FusedDeepMoeProfileStage::Dispatch), groupCountCapacity),
                "invalid dispatch occurrence capacity.");
    TORCH_CHECK(Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(Cam::FusedDeepMoeProfileStage::Gmm1),
                                                    groupCountCapacity),
                "invalid gmm1 occurrence capacity.");
    TORCH_CHECK(Cam::SetProfileStageOccurrenceCount(
                    layout, static_cast<uint32_t>(Cam::FusedDeepMoeProfileStage::SwigluQuant), 1U),
                "invalid swiglu occurrence capacity.");
    TORCH_CHECK(Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(Cam::FusedDeepMoeProfileStage::Gmm2),
                                                    groupCountCapacity),
                "invalid gmm2 occurrence capacity.");
    TORCH_CHECK(Cam::SetProfileStageOccurrenceCount(
                    layout, static_cast<uint32_t>(Cam::FusedDeepMoeProfileStage::Combine), groupCountCapacity),
                "invalid combine occurrence capacity.");
    TORCH_CHECK(Cam::SetProfileStageOccurrenceCount(
                    layout, static_cast<uint32_t>(Cam::FusedDeepMoeProfileStage::WeightSum), 1U),
                "invalid weight sum occurrence capacity.");
    return layout;
}

uint32_t GetRecordsPerLaunch(const Cam::FusedDeepMoeProfileStageLayout &stageLayout)
{
    return Cam::GetFusedDeepMoeProfileRecordsPerLaunch(Cam::FUSED_DEEP_MOE_PROFILE_LOGICAL_CORE_COUNT_CAPACITY,
                                                       stageLayout);
}

uint64_t GetPerLaunchBytes(const Cam::FusedDeepMoeProfileStageLayout &stageLayout)
{
    return static_cast<uint64_t>(GetRecordsPerLaunch(stageLayout)) * sizeof(Cam::FusedDeepMoeProfileRecord);
}

uint64_t GetTotalBytes(uint64_t launchCountCapacity, uint64_t perLaunchBytes)
{
    return Cam::GetFusedDeepMoeProfileDataOffset() + launchCountCapacity * AlignUp(perLaunchBytes, 64ULL);
}

at::Tensor AllocateBuffer(uint64_t totalBytes, uint32_t launchCountCapacity, uint32_t groupCountCapacity)
{
    auto stageLayout = BuildStageLayout(groupCountCapacity);
    at::TensorOptions options = at::TensorOptions(torch_npu::utils::get_npu_device_type());
    auto profileBuffer = at::zeros({static_cast<int64_t>(totalBytes)}, options.dtype(c10::kByte));
    auto header = BuildHeader(launchCountCapacity, groupCountCapacity, stageLayout);
    header.layoutPacked1 = Cam::PackProfileLayout1(Cam::FUSED_DEEP_MOE_PROFILE_LOGICAL_CORE_COUNT_CAPACITY,
                                                   GetRecordsPerLaunch(stageLayout));
    CopyHeaderToDevice(profileBuffer, header, stageLayout);
    return profileBuffer;
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

int64_t GetNumWarmups()
{
    return GetSessionImpl().numWarmups;
}

int64_t GetNumTests()
{
    return GetSessionImpl().numTests;
}

void Begin(int64_t numWarmups, int64_t numTests, const std::string &profileTraceDir)
{
    TORCH_CHECK(numWarmups >= 0, "num_warmups must be non-negative");
    TORCH_CHECK(numTests >= 0, "num_tests must be non-negative");
    TORCH_CHECK(!GetSessionImpl().active, "profile session is already active.");
    int64_t expectedLaunches = numWarmups + numTests;
    TORCH_CHECK(expectedLaunches > 0, "profile session needs at least one launch.");
    TORCH_CHECK(static_cast<uint64_t>(expectedLaunches) <= UINT32_MAX,
                "profile session launch count exceeds uint32 capacity.");
    auto &session = GetSessionImpl();
    session.Reset();
    session.active = true;
    session.numWarmups = numWarmups;
    session.numTests = numTests;
    session.expectedLaunches = expectedLaunches;
    session.capturedLaunches = 0;
    session.profileTraceDir = profileTraceDir;
    session.launchCountCapacity = static_cast<uint32_t>(expectedLaunches);
    if (IsDebugEnabled()) {
        std::ostringstream oss;
        oss << "begin session: num_warmups=" << numWarmups << ", num_tests=" << numTests
            << ", launch_capacity=" << session.launchCountCapacity << ", trace_dir=" << profileTraceDir;
        DebugPrint(oss.str());
    }
}

void EnsureInitialized(int64_t numExperts, int64_t numRanks)
{
    auto &session = GetSessionImpl();
    TORCH_CHECK(session.active, "profile session is not active.");
    TORCH_CHECK(session.launchCountCapacity > 0, "profile session launch capacity is not set.");
    uint32_t groupCountCapacity = GetGroupCountCapacity(numExperts, numRanks);
    TORCH_CHECK(groupCountCapacity >= 1U && groupCountCapacity <= Cam::FUSED_DEEP_MOE_PROFILE_MAX_GROUP_COUNT_CAPACITY,
                "profile session group count exceeds occurrence capacity: groupCountCapacity=", groupCountCapacity,
                ", valid_range=[1, ", Cam::FUSED_DEEP_MOE_PROFILE_MAX_GROUP_COUNT_CAPACITY, "]");
    if (session.initialized) {
        TORCH_CHECK(session.groupCountCapacity == groupCountCapacity,
                    "profile session group count changed within one session: expected ", session.groupCountCapacity,
                    ", got ", groupCountCapacity, ".");
        return;
    }

    session.groupCountCapacity = groupCountCapacity;
    session.stageLayout = BuildStageLayout(groupCountCapacity);
    session.recordsPerLaunch = GetRecordsPerLaunch(session.stageLayout);
    session.perLaunchBytes = GetPerLaunchBytes(session.stageLayout);
    uint64_t totalBytes = GetTotalBytes(static_cast<uint64_t>(session.launchCountCapacity), session.perLaunchBytes);
    TORCH_CHECK(totalBytes <= kMaxBytesPerRank, "profile buffer would exceed the per-rank cap: totalBytes=", totalBytes,
                ", cap=", kMaxBytesPerRank);
    session.profileBufferBytes = totalBytes;
    session.profileBuffer = AllocateBuffer(totalBytes, session.launchCountCapacity, session.groupCountCapacity);
    session.initialized = true;
}

void IncrementCapturedLaunches()
{
    ++GetSessionImpl().capturedLaunches;
}

void ExportBufferToTrace(const at::Tensor &profileBuffer, int64_t rank, const std::string &profileTraceDir,
                         int64_t numWarmups, int64_t launchCountCaptured)
{
    ExportBufferToTraceImpl(profileBuffer, rank, profileTraceDir, numWarmups, launchCountCaptured);
}

void ExportAndReset(int64_t rank)
{
    auto &session = GetSessionImpl();
    if (!session.active) {
        if (IsDebugEnabled()) {
            DebugPrint("end session ignored: inactive");
        }
        return;
    }
    if (session.capturedLaunches != session.expectedLaunches) {
        TORCH_WARN("profile session captured ", session.capturedLaunches, " launches, expected ",
                   session.expectedLaunches, ".");
    }
    if (IsDebugEnabled()) {
        std::ostringstream oss;
        oss << "end session: launches=" << session.capturedLaunches << ", expected=" << session.expectedLaunches
            << ", trace_dir=" << session.profileTraceDir;
        DebugPrint(oss.str());
    }
    ExportBufferToTraceImpl(session.profileBuffer, rank, session.profileTraceDir, session.numWarmups,
                            session.capturedLaunches);
    session.Reset();
}

}  // namespace deep_ep::profile_trace
