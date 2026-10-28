#include "register/op_def_registry.h"

namespace ops {
class CamMoeDispatchNormalLayered : public OpDef
{
public:
    explicit CamMoeDispatchNormalLayered(const char *name) : OpDef(name)
    {
        this->Input("roce_context").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Input("x").ParamType(REQUIRED).DataTypeList({ge::DT_BF16, ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND});
        this->Input("topk_weights").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
        this->Input("topk_idx").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Input("layered_plan").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("recv_x")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        this->Output("layered_assist").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("recv_topk_idx").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("recv_topk_weights").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
        this->Attr("group_ep").String();
        this->Attr("ep_world_size").Int();
        this->Attr("ep_rank_id").Int();
        this->Attr("group_tp").String();
        this->Attr("moe_expert_num").Int();
        this->Attr("ranks_per_server").Int();
        this->Attr("max_bs").Int();
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_true");
        this->AICore().AddConfig("ascend910_93", config);
#ifdef __DAV_C310__
        this->AICore().AddConfig("ascend950", config);
#endif
        this->MC2().HcclGroup({"group_ep", "group_tp"});
    }
};
OP_ADD(CamMoeDispatchNormalLayered);
}  // namespace ops
