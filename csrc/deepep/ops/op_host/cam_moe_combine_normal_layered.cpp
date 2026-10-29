#include "register/op_def_registry.h"

namespace ops {
class CamMoeCombineNormalLayered : public OpDef
{
public:
    explicit CamMoeCombineNormalLayered(const char *name) : OpDef(name)
    {
        this->Input("roce_context").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Input("recv_x")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        this->Input("layered_plan").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("x").ParamType(REQUIRED).DataTypeList({ge::DT_BF16, ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND});
        this->Attr("ep_group_name").String();
        this->Attr("ep_world_size").Int();
        this->Attr("ep_rank_id").Int();
        this->Attr("tp_group_name").String();
        this->Attr("moe_expert_num").Int();
        this->Attr("real_max_bs").Int();
        this->Attr("round").Int();
        this->Attr("per_round_tokens").Int();
        this->Attr("ranks_per_server").Int();
        this->Attr("topk").Int();
        this->Attr("max_bs").Int();
        this->Attr("owner_capacity").Int();
        this->Attr("invocation_id").Int();
        this->Attr("window_capacity").Int();
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
        this->MC2().HcclGroup({"ep_group_name", "tp_group_name"});
    }
};
OP_ADD(CamMoeCombineNormalLayered);
}  // namespace ops
