#include "register/op_def_registry.h"

namespace ops {
class NotifyDispatchNormalLayered : public OpDef
{
public:
    explicit NotifyDispatchNormalLayered(const char *name) : OpDef(name)
    {
        this->Input("roce_context").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("topk_idx").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("recvTokensPerExpert").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("layeredPlan").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("expertTotalCount").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Attr("comm_group").String();
        this->Attr("rank_size").Int();
        this->Attr("rank_id").Int();
        this->Attr("ranks_per_server").Int();
        this->Attr("moe_expert_num").Int();
        this->Attr("slots_per_server").Int();
        this->Attr("epoch").Int();
        this->Attr("hidden_size").Int();
        this->Attr("max_bs").Int();
        this->Attr("expert_alignment").Int();
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
        this->MC2().HcclGroup("comm_group");
    }
};
OP_ADD(NotifyDispatchNormalLayered);
}  // namespace ops
