/*
 * SPDX-License-Identifier: MIT
 */

#include "profiling/core/profile_exporter.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <vector>

#include "exception.hpp"

namespace deep_ep::profiling::exporter {
namespace {

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

static uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

static std::string CoreTypeName(uint64_t coreType)
{
    return (coreType == Cam::PROFILE_CORE_TYPE_AIV) ? "AIV" : "AIC";
}

}  // namespace

void ExportBufferToTrace(const at::Tensor &profileBuffer, int64_t rank, const std::string &profileTraceDir,
                         int64_t numProfileSkipLaunches, int64_t launchCountCaptured, const ProfileSchema &schema)
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
    auto *header = reinterpret_cast<const Cam::ProfileHeader *>(base);
    if (header == nullptr || header->magic != Cam::PROFILE_MAGIC || header->version != Cam::PROFILE_VERSION ||
        header->cycleToUs == 0) {
        return;
    }
    auto *stageLayout = reinterpret_cast<const Cam::ProfileStageLayout *>(base + sizeof(Cam::ProfileHeader));

    uint32_t launchCountCapacity = Cam::UnpackProfileLaunchCapacity(header->launchCountsPacked);
    uint32_t stageCount = Cam::UnpackProfileStageCount(header->layoutPacked0);
    uint32_t groupCountCapacity = Cam::UnpackProfileGroupCountCapacity(header->layoutPacked0);
    uint32_t logicalCoreCountCapacity = Cam::UnpackProfileLogicalCoreCountCapacity(header->layoutPacked1);
    uint32_t recordsPerLaunch = Cam::UnpackProfileRecordsPerLaunch(header->layoutPacked1);
    uint32_t expectedRecordsPerLaunch = Cam::GetProfileRecordsPerLaunch(logicalCoreCountCapacity, *stageLayout);
    if (stageCount == 0U || stageCount > Cam::PROFILE_ACTIVE_STAGE_CAPACITY ||
        stageCount > Cam::PROFILE_RESERVED_STAGE_CAPACITY || stageLayout->stageCount != stageCount ||
        stageLayout->activeStageCapacity != Cam::PROFILE_ACTIVE_STAGE_CAPACITY || groupCountCapacity == 0U ||
        groupCountCapacity > Cam::PROFILE_MAX_GROUP_COUNT_CAPACITY ||
        logicalCoreCountCapacity != schema.topology.logicalCoreCount || recordsPerLaunch != expectedRecordsPerLaunch) {
        TORCH_WARN("Unexpected profile layout, skip export.");
        return;
    }

    uint64_t perLaunchBytes = static_cast<uint64_t>(recordsPerLaunch) * sizeof(Cam::ProfileRecord);
    uint64_t requiredBytes = Cam::GetProfileDataOffset() + launchCountCapacity * AlignUp(perLaunchBytes, 64ULL);
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
        auto *launchBase =
            base + Cam::GetProfileLaunchOffset(Cam::GetProfileDataOffset(), recordsPerLaunch,
                                               static_cast<uint32_t>(sizeof(Cam::ProfileRecord)), launchId);
        auto *records = reinterpret_cast<const Cam::ProfileRecord *>(launchBase);
        LaunchTraceBundle bundle;
        bundle.launchId = static_cast<int64_t>(launchId);
        bundle.isWarmup = static_cast<int64_t>(launchId) < numProfileSkipLaunches;
        bool haveRange = false;
        for (uint64_t i = 0; i < recordsPerLaunch; ++i) {
            const auto &record = records[i];
            if (record.endCycle <= record.startCycle) {
                continue;
            }
            bundle.rows.push_back(
                {static_cast<double>(record.startCycle) / static_cast<double>(header->cycleToUs),
                 static_cast<double>(record.endCycle - record.startCycle) / static_cast<double>(header->cycleToUs),
                 static_cast<int64_t>(record.launchId), static_cast<int64_t>(record.launchId) < numProfileSkipLaunches,
                 record.coreType, record.coreIdx, record.stageId, record.occurrenceId, record.startCycle,
                 record.endCycle});
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

    std::filesystem::path traceDir =
        std::filesystem::path(profileTraceDir.empty() ? std::filesystem::current_path().string() : profileTraceDir);
    std::error_code ec;
    std::filesystem::create_directories(traceDir, ec);
    if (ec) {
        TORCH_WARN("Failed to create profile trace dir: ", traceDir.string(), ", error=", ec.message());
        return;
    }

    std::filesystem::path tracePath =
        traceDir / (std::string(schema.opName ? schema.opName : "profile") + "_rank" + std::to_string(rank) + ".json");
    std::ofstream ofs(tracePath, std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) {
        TORCH_WARN("Failed to open profile trace file: ", tracePath.string());
        return;
    }

    ofs << "{\n  \"traceEvents\": [\n";
    bool needComma = false;
    auto emitEvent = [&](const std::string &name, const std::string &ph, uint64_t pid, uint64_t tid, double ts,
                         double dur, const std::string &argsJson) {
        if (needComma) {
            ofs << ",\n";
        }
        needComma = true;
        ofs << "    {"
            << "\"name\":" << JsonEscape(name) << ","
            << "\"cat\":" << JsonEscape(schema.opName ? schema.opName : "profile") << ","
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
        emitEvent(std::string(schema.opName ? schema.opName : "profile") + "_launch", "X", static_cast<uint64_t>(rank),
                  launchTid, static_cast<double>(bundle.minStartCycle) / Cam::PROFILE_CYCLE_TO_US,
                  static_cast<double>(bundle.maxEndCycle - bundle.minStartCycle) /
                      static_cast<double>(Cam::PROFILE_CYCLE_TO_US),
                  std::string("{\"rank\":") + std::to_string(rank) + ",\"launch_id\":" +
                      std::to_string(bundle.launchId) + ",\"iteration_id\":" + std::to_string(bundle.launchId) +
                      ",\"is_warmup\":" + (bundle.isWarmup ? std::string("true") : std::string("false")) + "}");

        for (const auto &row : bundle.rows) {
            uint64_t tid = Cam::GetProfileLogicalCoreLinear(row.coreType, row.coreIdx);
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
            uint64_t tid = Cam::GetProfileLogicalCoreLinear(row.coreType, row.coreIdx);
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
            args << "\"stage_name\":" << JsonEscape(schema.stageName ? schema.stageName(row.stageId) : "unknown")
                 << ",";
            args << "\"occurrence_id\":" << row.occurrenceId << ",";
            args << "\"stage_label\":"
                 << JsonEscape(schema.stageDisplayName
                                   ? schema.stageDisplayName(row.stageId, row.occurrenceId, *stageLayout)
                                   : std::string("unknown"))
                 << ",";
            args << "\"launch_id\":" << row.launchId << ",";
            args << "\"iteration_id\":" << row.launchId << ",";
            args << "\"is_warmup\":" << (row.isWarmup ? "true" : "false") << ",";
            args << "\"start_cycle\":" << row.startCycle << ",";
            args << "\"end_cycle\":" << row.endCycle;
            args << "}";
            emitEvent(schema.stageDisplayName ? schema.stageDisplayName(row.stageId, row.occurrenceId, *stageLayout)
                                              : std::string("unknown"),
                      "X", static_cast<uint64_t>(rank), tid, row.ts_us, row.dur_us, args.str());
        }
    }

    ofs << "\n  ]\n}\n";
}

}  // namespace deep_ep::profiling::exporter
