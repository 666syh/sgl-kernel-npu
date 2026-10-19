#ifndef CAM_MOE_DISPATCH_NORMAL_LAYERED_TILING_H
#define CAM_MOE_DISPATCH_NORMAL_LAYERED_TILING_H

#include "cam_moe_dispatch_normal_tiling.h"
#include "cam_moe_layered_tiling.h"

struct CamMoeDispatchNormalLayeredTilingData {
    CamMoeDispatchNormalTilingData normal;
    CamMoeLayeredInfo layered;
};

#endif
