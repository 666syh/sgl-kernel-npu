#ifndef NOTIFY_DISPATCH_NORMAL_LAYERED_TILING_H
#define NOTIFY_DISPATCH_NORMAL_LAYERED_TILING_H
#include "cam_moe_layered_tiling.h"

struct NotifyDispatchNormalLayeredTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    CamMoeLayeredInfo layered;
};
#endif
