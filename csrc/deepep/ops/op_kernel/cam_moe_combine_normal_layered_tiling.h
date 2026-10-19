#ifndef CAM_MOE_COMBINE_NORMAL_LAYERED_TILING_H
#define CAM_MOE_COMBINE_NORMAL_LAYERED_TILING_H

#include "cam_moe_combine_normal_tiling.h"
#include "cam_moe_layered_tiling.h"

struct CamMoeCombineNormalLayeredTilingData {
    CamMoeCombineNormalTilingData normal;
    CamMoeLayeredInfo layered;
    uint32_t ownerCapacity;
    uint32_t chunkPackets;
};

#endif
