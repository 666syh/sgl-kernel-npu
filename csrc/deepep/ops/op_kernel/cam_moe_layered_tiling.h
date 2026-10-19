#ifndef CAM_MOE_LAYERED_TILING_H
#define CAM_MOE_LAYERED_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct CamMoeLayeredInfo {
    uint32_t epWorldSize;
    uint32_t epRankId;
    uint32_t rankInServer;
    uint32_t serverNum;
    uint32_t ranksPerServer;
    uint32_t moeExpertNum;
    uint32_t bs;
    uint32_t k;
    uint32_t h;
    uint32_t realMaxBs;
    uint32_t maxBs;
    uint32_t validRounds;
    uint32_t physicalRounds;
    uint32_t slotsPerServer;
    uint32_t recordStride;
    uint32_t dispatchChunkPackets;
    uint32_t dispatchMaxChunks;
    uint32_t partialStride;
    uint32_t controlStride;
    uint32_t ipcTilePackets;
    uint32_t expertAlignment;
    uint32_t elementBytes;
    uint32_t planElements;
    uint32_t packetCountOffset;
    uint32_t dispatchMemberCountOffset;
    uint32_t packetSlotOffset;
    uint32_t expertCountOffset;
    uint32_t localExpertOffset;
    uint32_t localExpertCountOffset;
    uint32_t overflowOffset;
    uint32_t packetDescOffset;
    uint32_t memberDescOffset;
    uint32_t errorOffset;
    uint64_t epoch;
    uint32_t aivNum;
    uint64_t totalUbSize;
    uint64_t totalWinSize;
    uint64_t activeWindowBytes;
    uint64_t windowRequiredBytes;
};

struct CamMoeLayeredTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    CamMoeLayeredInfo info;
};

#endif
