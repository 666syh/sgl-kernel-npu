#include "register/op_impl_registry.h"
#include "cam_moe_layered_plan.h"

using namespace ge;
using namespace CamMoeLayered;

namespace ops {
namespace {
ge::graphStatus InferNotify(gert::InferShapeContext *ctx)
{
    const auto *topk = ctx->GetInputShape(1);
    auto *recvCounts = ctx->GetOutputShape(0);
    auto *plan = ctx->GetOutputShape(1);
    auto *total = ctx->GetOutputShape(2);
    auto *attrs = ctx->GetAttrs();
    if (topk == nullptr || recvCounts == nullptr || plan == nullptr || total == nullptr || attrs == nullptr ||
        topk->GetDimNum() != 2) {
        return GRAPH_FAILED;
    }
    auto get = [&](uint32_t index) -> uint32_t {
        auto *value = attrs->GetAttrPointer<int64_t>(static_cast<int>(index));
        return value == nullptr ? 0U : static_cast<uint32_t>(*value);
    };
    const uint32_t world = get(1U);
    const uint32_t ranksPerServer = get(3U);
    const uint32_t servers = ranksPerServer == 0U ? 0U : world / ranksPerServer;
    const uint32_t experts = get(4U);
    const uint32_t slots = get(5U);
    const uint32_t maxBs = get(9U);
    const uint32_t bs = static_cast<uint32_t>(topk->GetDim(0));
    const uint32_t rounds = slots == 0U || maxBs == 0U ? 0U : (maxBs + slots - 1U) / slots;
    if (world == 0U || servers == 0U || experts == 0U || slots == 0U || maxBs < bs || rounds == 0U) {
        return GRAPH_FAILED;
    }
    const auto layout = MakePlanLayout(bs, maxBs, static_cast<uint32_t>(topk->GetDim(1)), experts, world, servers,
                                       ranksPerServer, rounds, slots);
    plan->SetDimNum(1);
    plan->SetDim(0, layout.total);
    recvCounts->SetDimNum(1);
    recvCounts->SetDim(0, experts / world);
    total->SetDimNum(1);
    total->SetDim(0, 1);
    return GRAPH_SUCCESS;
}

ge::graphStatus InferNotifyType(gert::InferDataTypeContext *ctx)
{
    for (uint32_t output = 0; output < 3U; ++output) {
        ctx->SetOutputDataType(output, DT_INT32);
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDispatch(gert::InferShapeContext *ctx)
{
    const auto *x = ctx->GetInputShape(1);
    const auto *weights = ctx->GetInputShape(2);
    const auto *topk = ctx->GetInputShape(3);
    const auto *plan = ctx->GetInputShape(4);
    auto *recv = ctx->GetOutputShape(0);
    auto *recvTopk = ctx->GetOutputShape(1);
    auto *recvWeights = ctx->GetOutputShape(2);
    if (x == nullptr || weights == nullptr || topk == nullptr || plan == nullptr || recv == nullptr ||
        recvTopk == nullptr || recvWeights == nullptr || x->GetDimNum() != 2 || weights->GetDimNum() != 2 ||
        topk->GetDimNum() != 2 || plan->GetDimNum() != 1)
        return GRAPH_FAILED;
    recv->SetDimNum(2);
    recv->SetDim(0, -1);
    recv->SetDim(1, x->GetDim(1));
    recvTopk->SetDimNum(2);
    recvTopk->SetDim(0, -1);
    recvTopk->SetDim(1, topk->GetDim(1));
    recvWeights->SetDimNum(2);
    recvWeights->SetDim(0, -1);
    recvWeights->SetDim(1, topk->GetDim(1));
    return GRAPH_SUCCESS;
}
ge::graphStatus InferDispatchType(gert::InferDataTypeContext *ctx)
{
    ctx->SetOutputDataType(0, ctx->GetInputDataType(1));
    ctx->SetOutputDataType(1, DT_INT32);
    ctx->SetOutputDataType(2, DT_FLOAT);
    return GRAPH_SUCCESS;
}

ge::graphStatus InferCombine(gert::InferShapeContext *ctx)
{
    const auto *recv = ctx->GetInputShape(1);
    const auto *plan = ctx->GetInputShape(2);
    auto *out = ctx->GetOutputShape(0);
    auto *attrs = ctx->GetAttrs();
    auto *bs = attrs == nullptr ? nullptr : attrs->GetAttrPointer<int64_t>(7);
    if (recv == nullptr || plan == nullptr || out == nullptr || bs == nullptr || *bs <= 0 || recv->GetDimNum() != 2 ||
        plan->GetDimNum() != 1) {
        return GRAPH_FAILED;
    }
    out->SetDimNum(2);
    out->SetDim(0, *bs);
    out->SetDim(1, recv->GetDim(1));
    return GRAPH_SUCCESS;
}

ge::graphStatus InferCombineType(gert::InferDataTypeContext *ctx)
{
    ctx->SetOutputDataType(0, ctx->GetInputDataType(1));
    return GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_INFERSHAPE(NotifyDispatchNormalLayered).InferShape(InferNotify).InferDataType(InferNotifyType);
IMPL_OP_INFERSHAPE(CamMoeDispatchNormalLayered).InferShape(InferDispatch).InferDataType(InferDispatchType);
IMPL_OP_INFERSHAPE(CamMoeCombineNormalLayered).InferShape(InferCombine).InferDataType(InferCombineType);
}  // namespace ops
