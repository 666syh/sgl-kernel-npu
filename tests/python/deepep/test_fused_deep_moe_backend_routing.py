import argparse
import os
import random
from typing import Iterable, List, Optional

import deep_ep
import torch
import torch.distributed as dist
import torch_npu
from deep_ep.buffer import FuseMode
from utils import calc_diff, init_dist

torch_npu.npu.config.allow_internal_format = True
os.environ["MOE_EXPERT_TOKEN_NUMS_TYPE"] = "1"


def warn_rank0(rank: int, message: str):
    if rank == 0:
        print(f"[rank0][WARNING] {message}", flush=True)


def info_rank0(rank: int, message: str):
    if rank == 0:
        print(f"[rank0] {message}", flush=True)


def set_env_if_provided(name: str, value):
    if value is not None:
        os.environ[name] = str(value)


def apply_runtime_env_from_args(args: argparse.Namespace):
    set_env_if_provided("MASTER_ADDR", args.master_addr)
    set_env_if_provided("MASTER_PORT", args.master_port)
    set_env_if_provided("WORLD_SIZE", args.num_servers)
    set_env_if_provided("RANK", args.server_index)
    set_env_if_provided("HCCL_BUFFSIZE", args.hccl_buffsize)


def parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_linear_beta_list(raw: str) -> List[Optional[float]]:
    values: List[Optional[float]] = []
    for item in parse_csv_list(raw):
        value = float(item)
        values.append(None if value == 0 else value)
    return values


def parse_activation_clamp_list(raw: str) -> List[Optional[float]]:
    values: List[Optional[float]] = []
    for item in parse_csv_list(raw):
        value = float(item)
        values.append(None if value == 0 else value)
    return values


def parse_beta_list(raw: str) -> List[float]:
    return [float(item) for item in parse_csv_list(raw)]


def swiglu_reference(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    gate = x[..., :d].to(torch.float32)
    up = x[..., d:].to(torch.float32)
    return (gate * torch.sigmoid(gate) * up).to(x.dtype)


def situ_reference(
    x: torch.Tensor,
    beta: float = 1.0,
    linear_beta: Optional[float] = None,
    activation_clamp: Optional[float] = None,
) -> torch.Tensor:
    d = x.shape[-1] // 2
    gate = x[..., :d].to(torch.float32)
    up = x[..., d:].to(torch.float32)
    if activation_clamp is not None and activation_clamp > 0:
        gate = torch.clamp(gate, -activation_clamp, activation_clamp)
    situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None and linear_beta > 0:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (situ_a * up).to(x.dtype)


def apply_activation(
    x: torch.Tensor,
    activation: str,
    *,
    beta: float,
    linear_beta: Optional[float],
    activation_clamp: Optional[float],
) -> torch.Tensor:
    if activation == "swiglu":
        return swiglu_reference(x)
    if activation == "situ":
        return situ_reference(
            x,
            beta=beta,
            linear_beta=linear_beta,
            activation_clamp=activation_clamp,
        )
    raise ValueError(f"Unsupported activation: {activation}")


def make_topk_inputs(
    num_tokens: int,
    num_experts: int,
    num_ranks: int,
    num_topk: int,
    active_ranks: Iterable[int],
    device: str,
):
    active_ranks = list(active_ranks)
    if not active_ranks:
        raise ValueError("`active_ranks` must not be empty.")
    experts_per_rank = num_experts // num_ranks
    num_active_experts = len(active_ranks) * experts_per_rank
    if num_active_experts < num_topk:
        raise ValueError(
            f"Active experts ({num_active_experts}) must be >= num_topk ({num_topk})."
        )
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device=device)
    active_expert_mask = torch.zeros(num_experts, dtype=torch.bool, device=device)
    for rank_id in active_ranks:
        start = rank_id * experts_per_rank
        end = (rank_id + 1) * experts_per_rank
        active_expert_mask[start:end] = True
    masked_scores = scores.masked_fill(~active_expert_mask.unsqueeze(0), float("-inf"))
    topk_weights, topk_ids = torch.topk(
        masked_scores, num_topk, dim=-1, largest=True, sorted=False
    )
    return topk_ids.to(torch.int32), topk_weights


def quantize_per_channel_int8(
    weight: torch.Tensor, limit: float
) -> tuple[torch.Tensor, torch.Tensor]:
    max_abs = torch.amax(torch.abs(weight.to(torch.float32)), dim=0, keepdim=True)
    scale = torch.clamp(max_abs / limit, min=1e-8)
    q_weight = torch.round(weight.to(torch.float32) / scale).clamp(-limit, limit)
    return q_weight.to(torch.int8), scale.squeeze(0).to(torch.float32)


def pack_scale_to_uint64(scale: torch.Tensor) -> torch.Tensor:
    return scale.contiguous().view(torch.int32).to(torch.int64).view(torch.uint64)


def trans_quant_param_for_matmul(scale: torch.Tensor) -> torch.Tensor:
    flat = scale.to(torch.float32).contiguous().reshape(-1)
    packed = torch_npu.npu_trans_quant_param(flat)
    return packed.reshape(scale.shape)


def expand_per_channel_scale_for_a8w4(
    scale: torch.Tensor, input_k: int, quant_group_size: int = 256
) -> torch.Tensor:
    if input_k % quant_group_size != 0:
        raise ValueError(
            f"A8W4 grouped scale requires input_k divisible by {quant_group_size}, "
            f"but got input_k={input_k}."
        )
    group_num = input_k // quant_group_size
    return scale.unsqueeze(0).repeat(group_num, 1).contiguous()


def pack_int4_to_int8(weight: torch.Tensor) -> torch.Tensor:
    packed = (
        (weight.to(torch.int16) + 8)
        .to(torch.uint8)
        .reshape(weight.shape[0], weight.shape[1] // 2, 2)
    )
    return ((packed[..., 1] << 4) | packed[..., 0]).view(torch.int8)


def cast_weights_to_fractal_nz(weights: List[torch.Tensor]) -> List[torch.Tensor]:
    return [
        torch_npu.npu_format_cast(w.contiguous(), torch_npu.Format.FRACTAL_NZ)
        for w in weights
    ]


def pack_int4_weights_for_fused(weights: List[torch.Tensor]) -> List[torch.Tensor]:
    packed_int8 = [pack_int4_to_int8(w) for w in weights]
    return [
        torch_npu.npu_format_cast(w.contiguous(), torch_npu.Format.FRACTAL_NZ).view(
            torch.int32
        )
        for w in packed_int8
    ]


def pack_int4_weights_for_grouped_matmul(
    weights: List[torch.Tensor],
) -> List[torch.Tensor]:
    packed = []
    for weight in weights:
        packed.append(
            torch_npu.npu_convert_weight_to_int4pack(
                weight.to(torch.int32).contiguous()
            )
        )
    return packed


def run_a8w4_quant_matmul(
    x_in: torch.Tensor,
    weight_packed: torch.Tensor,
    scale_for_matmul: torch.Tensor,
    offset: torch.Tensor,
) -> torch.Tensor:
    qx, pertoken_scale = quantize_tokens_to_int8(x_in)
    return torch_npu.npu_quant_matmul(
        qx,
        weight_packed,
        scale=scale_for_matmul,
        offset=offset,
        pertoken_scale=pertoken_scale,
        bias=None,
        output_dtype=torch.bfloat16,
        group_sizes=[0, 0, 256],
    )


def stack_expert_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack([tensor.contiguous() for tensor in tensors], dim=0)


def quantize_tokens_to_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = torch.clamp(
        torch.amax(torch.abs(x.to(torch.float32)), dim=1, keepdim=True) / 127.0,
        min=1e-8,
    )
    qx = torch.round(x.to(torch.float32) / scale).clamp(-127, 127)
    return qx.to(torch.int8), scale


def validate_topk_inputs(
    topk_ids_int32: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
):
    assert topk_ids_int32.dtype == torch.int32
    assert topk_weights.dtype == torch.float32
    assert topk_ids_int32.dim() == 2
    assert topk_weights.shape == topk_ids_int32.shape
    assert torch.all(topk_ids_int32 >= 0), "topk_ids must be non-negative."
    assert torch.all(topk_ids_int32 < num_experts), "topk_ids exceed num_experts."
    sorted_ids = torch.sort(topk_ids_int32, dim=-1).values
    if sorted_ids.size(1) > 1:
        has_duplicates = torch.any(sorted_ids[:, 1:] == sorted_ids[:, :-1]).item()
        assert not has_duplicates, "Each token must route to unique experts."


def validate_fused_weight_inputs(
    scene: str,
    weights: dict,
    num_local_experts: int,
    hidden: int,
    intermediate_hidden: int,
):
    fused_l1_weights = weights["fused_l1_weights"]
    fused_l2_weights = weights["fused_l2_weights"]
    assert len(fused_l1_weights) == num_local_experts
    assert len(fused_l2_weights) == num_local_experts

    if scene == "a16w16":
        for weight in fused_l1_weights + fused_l2_weights:
            assert (
                weight.dtype == torch.bfloat16
            ), f"{scene} fused weights must be BF16, but got {weight.dtype}"
        return

    if scene == "a8w8_int":
        for weight in fused_l1_weights + fused_l2_weights:
            assert (
                weight.dtype == torch.int8
            ), f"{scene} fused weights must be INT8, but got {weight.dtype}"
        return

    if scene == "a8w4_int":
        for weight in fused_l1_weights:
            assert (
                weight.dtype == torch.int32
            ), f"{scene} fused l1 weights must be packed INT4 exposed as INT32, got {weight.dtype}"
            assert weight.shape == (
                hidden,
                (intermediate_hidden * 2) // 8,
            ), (
                f"{scene} fused l1 weights must use packed shape "
                f"({hidden}, {(intermediate_hidden * 2) // 8}), but got {tuple(weight.shape)}"
            )
        for weight in fused_l2_weights:
            assert (
                weight.dtype == torch.int32
            ), f"{scene} fused l2 weights must be packed INT4 exposed as INT32, got {weight.dtype}"
            assert weight.shape == (
                intermediate_hidden,
                hidden // 8,
            ), (
                f"{scene} fused l2 weights must use packed shape "
                f"({intermediate_hidden}, {hidden // 8}), but got {tuple(weight.shape)}"
            )
        return

    raise ValueError(f"Unsupported scene: {scene}")


def validate_demo_aligned_inputs(
    scene: str,
    weights: dict,
    num_local_experts: int,
    hidden: int,
    intermediate_hidden: int,
    topk_ids_int32: torch.Tensor,
    topk_weights: torch.Tensor,
):
    validate_topk_inputs(topk_ids_int32, topk_weights, weights["num_experts"])
    assert (
        len(weights["fused_l1_weights"]) == num_local_experts
    ), "Unexpected number of local l1 expert weights."
    assert (
        len(weights["fused_l2_weights"]) == num_local_experts
    ), "Unexpected number of local l2 expert weights."
    if scene == "a8w8_int":
        assert weights["fused_l1_scales"] is not None
        assert weights["fused_l2_scales"] is not None
        for scale in weights["fused_l1_scales"] + weights["fused_l2_scales"]:
            assert scale.dtype == torch.uint64, "A8W8 fused scales must be UINT64."
    if scene == "a8w4_int":
        assert weights["fused_l1_bias"] is not None
        assert weights["fused_l2_bias"] is not None
        assert hidden % 256 == 0, "A8W4 baseline requires hidden divisible by 256."
        assert (
            intermediate_hidden % 256 == 0
        ), "A8W4 baseline requires intermediate_hidden divisible by 256."
        for weight in weights["fused_l1_weights"]:
            assert weight.shape == (
                hidden,
                (intermediate_hidden * 2) // 8,
            ), (
                f"A8W4 fused l1 weight should be packed to "
                f"({hidden}, {(intermediate_hidden * 2) // 8}), got {tuple(weight.shape)}"
            )
        for weight in weights["fused_l2_weights"]:
            assert weight.shape == (
                intermediate_hidden,
                hidden // 8,
            ), (
                f"A8W4 fused l2 weight should be packed to "
                f"({intermediate_hidden}, {hidden // 8}), got {tuple(weight.shape)}"
            )
        for scale in weights["fused_l1_scales"] + weights["fused_l2_scales"]:
            assert scale.dtype == torch.uint64, "A8W4 fused scales must be UINT64."
        for bias in weights["fused_l1_bias"] + weights["fused_l2_bias"]:
            assert bias.dtype == torch.float32, "A8W4 fused bias must be FLOAT32."
        for scale in weights["baseline_l1_scales_grouped_packed"]:
            assert scale.shape == (
                hidden // 256,
                intermediate_hidden * 2,
            ), (
                "A8W4 baseline grouped l1 scale must use shape "
                f"({hidden // 256}, {intermediate_hidden * 2}), got {tuple(scale.shape)}"
            )
        for scale in weights["baseline_l2_scales_grouped_packed"]:
            assert scale.shape == (
                intermediate_hidden // 256,
                hidden,
            ), (
                "A8W4 baseline grouped l2 scale must use shape "
                f"({intermediate_hidden // 256}, {hidden}), got {tuple(scale.shape)}"
            )
        for bias in weights["baseline_l1_bias"]:
            assert bias.shape == (
                intermediate_hidden * 2,
            ), f"A8W4 baseline l1 bias shape mismatch: {tuple(bias.shape)}"
        for bias in weights["baseline_l2_bias"]:
            assert bias.shape == (
                hidden,
            ), f"A8W4 baseline l2 bias shape mismatch: {tuple(bias.shape)}"


def init_scene_weights(
    scene: str,
    num_local_experts: int,
    hidden: int,
    intermediate_hidden: int,
    device: str = "npu",
):
    l1_bf16 = [
        torch.randn(
            (hidden, intermediate_hidden * 2),
            dtype=torch.bfloat16,
            device=device,
        )
        for _ in range(num_local_experts)
    ]
    l2_bf16 = [
        torch.randn(
            (intermediate_hidden, hidden),
            dtype=torch.bfloat16,
            device=device,
        )
        for _ in range(num_local_experts)
    ]

    if scene == "a16w16":
        return {
            "fused_l1_weights": l1_bf16,
            "fused_l2_weights": l2_bf16,
            "fused_l1_scales": None,
            "fused_l2_scales": None,
            "fused_l1_bias": None,
            "fused_l2_bias": None,
            "baseline_l1_weights": l1_bf16,
            "baseline_l2_weights": l2_bf16,
            "baseline_l1_scales": None,
            "baseline_l2_scales": None,
            "dispatch_quant_mode": 0,
            "dispatch_quant_out_dtype": None,
        }

    if scene == "a8w8_int":
        l1_int8: List[torch.Tensor] = []
        l2_int8: List[torch.Tensor] = []
        l1_scales_float: List[torch.Tensor] = []
        l2_scales_float: List[torch.Tensor] = []
        l1_scales_packed: List[torch.Tensor] = []
        l2_scales_packed: List[torch.Tensor] = []

        for w in l1_bf16:
            q_weight, scale = quantize_per_channel_int8(w, limit=127.0)
            l1_int8.append(q_weight)
            l1_scales_float.append(scale)
            l1_scales_packed.append(pack_scale_to_uint64(scale))
        for w in l2_bf16:
            q_weight, scale = quantize_per_channel_int8(w, limit=127.0)
            l2_int8.append(q_weight)
            l2_scales_float.append(scale)
            l2_scales_packed.append(pack_scale_to_uint64(scale))

        return {
            "fused_l1_weights": cast_weights_to_fractal_nz(l1_int8),
            "fused_l2_weights": cast_weights_to_fractal_nz(l2_int8),
            "fused_l1_scales": l1_scales_packed,
            "fused_l2_scales": l2_scales_packed,
            "fused_l1_bias": None,
            "fused_l2_bias": None,
            "baseline_l1_weights": l1_int8,
            "baseline_l2_weights": l2_int8,
            "baseline_l1_scales": l1_scales_float,
            "baseline_l2_scales": l2_scales_float,
            "dispatch_quant_mode": 2,
            "dispatch_quant_out_dtype": torch.int8,
        }

    if scene == "a8w4_int":
        l1_int4: List[torch.Tensor] = []
        l2_int4: List[torch.Tensor] = []
        l1_int4_pack: List[torch.Tensor] = []
        l2_int4_pack: List[torch.Tensor] = []
        l1_scales_float = []
        l2_scales_float = []
        l1_scales_packed = []
        l2_scales_packed = []
        l1_scales_grouped_float = []
        l2_scales_grouped_float = []
        l1_scales_grouped_packed = []
        l2_scales_grouped_packed = []
        l1_scales_quant_matmul = []
        l2_scales_quant_matmul = []
        l1_bias = []
        l2_bias = []

        for w in l1_bf16:
            q_weight, scale = quantize_per_channel_int8(w, limit=7.0)
            l1_int4.append(q_weight)
            l1_int4_pack.append(
                torch_npu.npu_convert_weight_to_int4pack(q_weight.to(torch.int32))
            )
            l1_scales_float.append(scale)
            l1_scales_packed.append(pack_scale_to_uint64(scale))
            grouped_scale = expand_per_channel_scale_for_a8w4(scale, hidden)
            l1_scales_grouped_float.append(grouped_scale)
            l1_scales_grouped_packed.append(pack_scale_to_uint64(grouped_scale))
            l1_scales_quant_matmul.append(trans_quant_param_for_matmul(grouped_scale))
            l1_bias.append(
                (q_weight.to(torch.float32) * scale.unsqueeze(0)).sum(dim=0) * 8.0
            )
        for w in l2_bf16:
            q_weight, scale = quantize_per_channel_int8(w, limit=7.0)
            l2_int4.append(q_weight)
            l2_int4_pack.append(
                torch_npu.npu_convert_weight_to_int4pack(q_weight.to(torch.int32))
            )
            l2_scales_float.append(scale)
            l2_scales_packed.append(pack_scale_to_uint64(scale))
            grouped_scale = expand_per_channel_scale_for_a8w4(
                scale, intermediate_hidden
            )
            l2_scales_grouped_float.append(grouped_scale)
            l2_scales_grouped_packed.append(pack_scale_to_uint64(grouped_scale))
            l2_scales_quant_matmul.append(trans_quant_param_for_matmul(grouped_scale))
            l2_bias.append(
                (q_weight.to(torch.float32) * scale.unsqueeze(0)).sum(dim=0) * 8.0
            )

        fused_l1 = pack_int4_weights_for_fused(l1_int4)
        fused_l2 = pack_int4_weights_for_fused(l2_int4)
        return {
            "fused_l1_weights": fused_l1,
            "fused_l2_weights": fused_l2,
            "fused_l1_scales": l1_scales_packed,
            "fused_l2_scales": l2_scales_packed,
            "fused_l1_bias": l1_bias,
            "fused_l2_bias": l2_bias,
            "baseline_l1_weights": l1_int4,
            "baseline_l2_weights": l2_int4,
            "baseline_l1_weights_packed": l1_int4_pack,
            "baseline_l2_weights_packed": l2_int4_pack,
            "baseline_l1_scales": l1_scales_float,
            "baseline_l2_scales": l2_scales_float,
            "baseline_l1_scales_packed": l1_scales_packed,
            "baseline_l2_scales_packed": l2_scales_packed,
            "baseline_l1_scales_grouped_float": l1_scales_grouped_float,
            "baseline_l2_scales_grouped_float": l2_scales_grouped_float,
            "baseline_l1_scales_grouped_packed": l1_scales_grouped_packed,
            "baseline_l2_scales_grouped_packed": l2_scales_grouped_packed,
            "baseline_l1_scales_quant_matmul": l1_scales_quant_matmul,
            "baseline_l2_scales_quant_matmul": l2_scales_quant_matmul,
            "baseline_l1_bias": l1_bias,
            "baseline_l2_bias": l2_bias,
            "dispatch_quant_mode": 2,
            "dispatch_quant_out_dtype": torch.int8,
        }

    raise ValueError(f"Unsupported scene: {scene}")


def normalize_expert_counts(
    counts: List[int], expected_local_experts: int, device: torch.device
) -> torch.Tensor:
    if len(counts) != expected_local_experts:
        raise ValueError(
            f"Expected {expected_local_experts} local expert counts, but got {len(counts)}."
        )
    return torch.tensor(counts, dtype=torch.int32, device=device)


def compute_local_expert_outputs(
    recv_x: torch.Tensor,
    recv_counts: List[int],
    weights: dict,
    activation: str,
    beta: float,
    linear_beta: Optional[float],
    activation_clamp: Optional[float],
    scene: str,
) -> torch.Tensor:
    if recv_x.numel() == 0:
        return recv_x

    outputs: List[torch.Tensor] = []
    start = 0
    for expert_id, token_count in enumerate(recv_counts):
        end = start + token_count
        if token_count == 0:
            start = end
            continue
        x_chunk = recv_x[start:end]

        if scene == "a16w16":
            h = torch.matmul(x_chunk, weights["baseline_l1_weights"][expert_id])
            a = apply_activation(
                h,
                activation,
                beta=beta,
                linear_beta=linear_beta,
                activation_clamp=activation_clamp,
            )
            y = torch.matmul(a, weights["baseline_l2_weights"][expert_id])
        elif scene == "a8w8_int":
            l1_scale = weights["baseline_l1_scales"][expert_id]
            l2_scale = weights["baseline_l2_scales"][expert_id]
            qx, qx_scale = quantize_tokens_to_int8(x_chunk)
            h_int = torch.matmul(
                qx.to(torch.float32),
                weights["baseline_l1_weights"][expert_id].to(torch.float32),
            )
            h = (h_int * qx_scale) * l1_scale.unsqueeze(0)
            a = apply_activation(
                h.to(torch.bfloat16),
                activation,
                beta=beta,
                linear_beta=linear_beta,
                activation_clamp=activation_clamp,
            )
            qa, qa_scale = quantize_tokens_to_int8(a)
            y_int = torch.matmul(
                qa.to(torch.float32),
                weights["baseline_l2_weights"][expert_id].to(torch.float32),
            )
            y = ((y_int * qa_scale) * l2_scale.unsqueeze(0)).to(torch.bfloat16)
        else:
            h = run_a8w4_quant_matmul(
                x_chunk,
                weights["baseline_l1_weights_packed"][expert_id],
                weights["baseline_l1_scales_quant_matmul"][expert_id],
                weights["baseline_l1_bias"][expert_id],
            )
            a = apply_activation(
                h,
                activation,
                beta=beta,
                linear_beta=linear_beta,
                activation_clamp=activation_clamp,
            )
            y = run_a8w4_quant_matmul(
                a.to(torch.bfloat16),
                weights["baseline_l2_weights_packed"][expert_id],
                weights["baseline_l2_scales_quant_matmul"][expert_id],
                weights["baseline_l2_bias"][expert_id],
            )

        outputs.append(y.to(torch.bfloat16))
        start = end

    if not outputs:
        return torch.empty_like(recv_x)
    return torch.cat(outputs, dim=0)


def run_baseline_reference(
    buffer: deep_ep.Buffer,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    weights: dict,
    activation: str,
    beta: float,
    linear_beta: Optional[float],
    activation_clamp: Optional[float],
    scene: str,
    rank: int,
    num_ranks: int,
):
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        _,
    ) = buffer.get_dispatch_layout(topk_idx, num_experts)

    dist.barrier()
    recv_x, _, _, recv_counts, handle, _ = buffer.dispatch(
        x=x,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        expert_alignment=1,
        async_finish=False,
    )
    dist.barrier()
    if isinstance(recv_x, tuple):
        raise NotImplementedError("This baseline expects BF16 dispatch outputs.")

    dist.barrier()
    local_out = compute_local_expert_outputs(
        recv_x,
        recv_counts,
        weights,
        activation,
        beta,
        linear_beta,
        activation_clamp,
        scene,
    )
    dist.barrier()
    dist.barrier()
    combined_x, _, _ = buffer.combine(
        x=local_out,
        handle=handle,
        topk_weights=handle[7],
        async_finish=False,
    )
    dist.barrier()
    return combined_x, normalize_expert_counts(
        recv_counts, num_experts // dist.get_world_size(), x.device
    )


def run_fused_reference(
    buffer: deep_ep.Buffer,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_tokens: int,
    num_experts: int,
    weights: dict,
    activation: str,
    activation_clamp: Optional[float],
    beta: float,
    linear_beta: Optional[float],
    rank: int,
    num_ranks: int,
):
    torch.npu.synchronize()
    fused_output, fused_counts = buffer.fused_deep_moe(
        x=x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        gmm1_permuted_weight=weights["fused_l1_weights"],
        gmm1_permuted_weight_scale=weights["fused_l1_scales"],
        gmm2_weight=weights["fused_l2_weights"],
        gmm2_weight_scale=weights["fused_l2_scales"],
        num_max_dispatch_tokens_per_rank=num_tokens,
        num_experts=num_experts,
        backend="mega_moe",
        fuse_mode=FuseMode.FUSED_DEEP_MOE,
        activation=activation,
        activation_clamp=activation_clamp,
        beta=beta,
        linear_beta=linear_beta,
        l1_bias=weights["fused_l1_bias"],
        l2_bias=weights["fused_l2_bias"],
        dispatch_quant_mode=weights["dispatch_quant_mode"],
        dispatch_quant_out_dtype=weights["dispatch_quant_out_dtype"],
    )
    return fused_output, fused_counts


def get_threshold(scene: str) -> float:
    if scene == "a16w16":
        return 5e-4
    if scene == "a8w8_int":
        return 2e-3
    if scene == "a8w4_int":
        return 5e-3
    raise ValueError(f"Unsupported scene: {scene}")


def run_case(
    baseline_buffer: deep_ep.Buffer,
    fused_buffer: deep_ep.Buffer,
    rank: int,
    num_ranks: int,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_ids_int32: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: dict,
    scene: str,
    activation: str,
    activation_clamp: Optional[float],
    beta: float,
    linear_beta: Optional[float],
    enable_performance: bool,
    performance_iterations: int,
):
    baseline_output, baseline_counts = run_baseline_reference(
        buffer=baseline_buffer,
        x=x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=weights["num_experts"],
        weights=weights,
        activation=activation,
        beta=beta,
        linear_beta=linear_beta,
        activation_clamp=activation_clamp,
        scene=scene,
        rank=rank,
        num_ranks=num_ranks,
    )
    fused_output, fused_counts = run_fused_reference(
        buffer=fused_buffer,
        x=x,
        topk_idx=topk_ids_int32,
        topk_weights=topk_weights,
        num_tokens=x.size(0),
        num_experts=weights["num_experts"],
        weights=weights,
        activation=activation,
        activation_clamp=activation_clamp,
        beta=beta,
        linear_beta=linear_beta,
        rank=rank,
        num_ranks=num_ranks,
    )
    torch.npu.synchronize()
    similarity_diff = calc_diff(fused_output.float(), baseline_output.float())
    threshold = get_threshold(scene)
    counts_match = torch.equal(
        baseline_counts.cpu(), fused_counts.to(torch.int32).cpu()
    )
    case_desc = (
        f"scene={scene}, activation={activation}, activation_clamp={activation_clamp}, "
        f"beta={beta}, linear_beta={linear_beta}"
    )
    if not counts_match:
        warn_rank0(
            rank,
            f"expert_token_nums mismatch for {case_desc}: "
            f"baseline={baseline_counts.tolist()}, fused={fused_counts.tolist()}",
        )
    if similarity_diff >= threshold:
        warn_rank0(
            rank,
            f"calc_diff={similarity_diff} exceeds threshold={threshold} for {case_desc}",
        )
    else:
        info_rank0(
            rank,
            f"calc_diff={similarity_diff} within threshold={threshold} for {case_desc}",
        )

    if enable_performance:
        for _ in range(performance_iterations):
            stress_baseline_output, stress_baseline_counts = run_baseline_reference(
                buffer=baseline_buffer,
                x=x,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                num_experts=weights["num_experts"],
                weights=weights,
                activation=activation,
                beta=beta,
                linear_beta=linear_beta,
                activation_clamp=activation_clamp,
                scene=scene,
                rank=rank,
                num_ranks=num_ranks,
            )
            torch.npu.synchronize()
            del stress_baseline_output, stress_baseline_counts
            dist.barrier()
            stress_fused_output, stress_fused_counts = run_fused_reference(
                buffer=fused_buffer,
                x=x,
                topk_idx=topk_ids_int32,
                topk_weights=topk_weights,
                num_tokens=x.size(0),
                num_experts=weights["num_experts"],
                weights=weights,
                activation=activation,
                activation_clamp=activation_clamp,
                beta=beta,
                linear_beta=linear_beta,
                rank=rank,
                num_ranks=num_ranks,
            )
            torch.npu.synchronize()
            del stress_fused_output, stress_fused_counts
            dist.barrier()
        info_rank0(
            rank,
            f"stress run completed for {case_desc}: "
            f"iterations={performance_iterations}",
        )

    return {
        "baseline_output": baseline_output.detach().cpu(),
        "fused_output": fused_output.detach().cpu(),
        "baseline_counts": baseline_counts.detach().cpu(),
        "fused_counts": fused_counts.detach().cpu(),
        "calc_diff": similarity_diff,
        "threshold": threshold,
        "counts_match": counts_match,
    }


def test_main(
    args: argparse.Namespace,
    rank: int,
    num_ranks: int,
    baseline_buffer: deep_ep.Buffer,
    fused_buffer: deep_ep.Buffer,
):
    device = torch.device("npu")
    scenes = parse_csv_list(args.scene)
    activations = parse_csv_list(args.activation)
    activation_clamps = parse_activation_clamp_list(args.activation_clamp)
    betas = parse_beta_list(args.beta)
    linear_betas = parse_linear_beta_list(args.linear_beta)

    if args.active_ranks:
        active_ranks = [int(item) for item in parse_csv_list(args.active_ranks)]
    else:
        active_ranks = list(range(num_ranks))

    results_by_case = {}
    for scene in scenes:
        num_local_experts = args.num_experts // num_ranks
        torch.manual_seed(args.seed + rank)
        random.seed(args.seed + rank)
        weights = init_scene_weights(
            scene=scene,
            num_local_experts=num_local_experts,
            hidden=args.hidden,
            intermediate_hidden=args.moe_intermediate_size,
            device="npu",
        )
        validate_fused_weight_inputs(
            scene=scene,
            weights=weights,
            num_local_experts=num_local_experts,
            hidden=args.hidden,
            intermediate_hidden=args.moe_intermediate_size,
        )
        weights["num_experts"] = args.num_experts

        torch.manual_seed(args.seed + rank)
        random.seed(args.seed + rank)
        x = torch.randn(
            (args.num_tokens, args.hidden), dtype=torch.bfloat16, device=device
        )
        topk_ids_int32, topk_weights = make_topk_inputs(
            num_tokens=args.num_tokens,
            num_experts=args.num_experts,
            num_ranks=num_ranks,
            num_topk=args.num_topk,
            active_ranks=active_ranks,
            device="npu",
        )
        topk_idx = topk_ids_int32.to(torch.int64)
        validate_demo_aligned_inputs(
            scene=scene,
            weights=weights,
            num_local_experts=num_local_experts,
            hidden=args.hidden,
            intermediate_hidden=args.moe_intermediate_size,
            topk_ids_int32=topk_ids_int32,
            topk_weights=topk_weights,
        )

        for activation in activations:
            for activation_clamp in activation_clamps:
                for beta in betas:
                    for linear_beta in linear_betas:
                        if activation != "situ":
                            if linear_beta is not None:
                                continue
                            if beta != 1.0:
                                continue
                        result = run_case(
                            baseline_buffer=baseline_buffer,
                            fused_buffer=fused_buffer,
                            rank=rank,
                            num_ranks=num_ranks,
                            x=x,
                            topk_idx=topk_idx,
                            topk_ids_int32=topk_ids_int32,
                            topk_weights=topk_weights,
                            weights=weights,
                            scene=scene,
                            activation=activation,
                            activation_clamp=activation_clamp,
                            beta=beta,
                            linear_beta=linear_beta,
                            enable_performance=args.enable_performance,
                            performance_iterations=args.performance_iterations,
                        )
                        case_key = (
                            activation,
                            activation_clamp,
                            beta,
                            linear_beta,
                        )
                        results_by_case.setdefault(case_key, {})[scene] = result
                        dist.barrier()
        torch.npu.synchronize()
        del weights, x, topk_ids_int32, topk_idx, topk_weights

    for case_key, scene_results in results_by_case.items():
        if "a8w8_int" not in scene_results or "a8w4_int" not in scene_results:
            continue
        activation, activation_clamp, beta, linear_beta = case_key
        case_desc = (
            f"activation={activation}, activation_clamp={activation_clamp}, "
            f"beta={beta}, linear_beta={linear_beta}"
        )
        w8_result = scene_results["a8w8_int"]
        w4_result = scene_results["a8w4_int"]
        fused_cross_diff = calc_diff(
            w8_result["fused_output"].float(),
            w4_result["fused_output"].float(),
        )
        baseline_cross_diff = calc_diff(
            w8_result["baseline_output"].float(),
            w4_result["baseline_output"].float(),
        )
        threshold = args.w8_w4_compare_threshold
        if fused_cross_diff >= threshold:
            warn_rank0(
                rank,
                f"W8A8 vs W4A8 fused calc_diff={fused_cross_diff} exceeds "
                f"threshold={threshold} for {case_desc}",
            )
        else:
            info_rank0(
                rank,
                f"W8A8 vs W4A8 fused calc_diff={fused_cross_diff} within "
                f"threshold={threshold} for {case_desc}",
            )
        if baseline_cross_diff >= threshold:
            warn_rank0(
                rank,
                f"W8A8 vs W4A8 baseline calc_diff={baseline_cross_diff} exceeds "
                f"threshold={threshold} for {case_desc}",
            )
        else:
            info_rank0(
                rank,
                f"W8A8 vs W4A8 baseline calc_diff={baseline_cross_diff} within "
                f"threshold={threshold} for {case_desc}",
            )
        dist.barrier()


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    if args.enable_performance and args.performance_iterations <= 0:
        raise ValueError("--performance-iterations must be greater than zero.")
    baseline_group = dist.new_group(list(range(num_ranks)))
    assert (
        args.num_experts % num_ranks == 0
    ), "num_experts must be divisible by world_size"

    fused_buffer = deep_ep.Buffer(
        group,
        int(2e9),
        0,
        low_latency_mode=False,
        num_qps_per_rank=1,
    )
    baseline_buffer = deep_ep.Buffer(
        baseline_group,
        int(2e9),
        0,
        low_latency_mode=False,
        num_qps_per_rank=1,
    )

    test_main(args, rank, num_ranks, baseline_buffer, fused_buffer)
    info_rank0(rank, "fused_deep_moe backend routing precision test completed")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Distributed precision comparison for fused_deep_moe mega_moe backend."
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=2,
        help="Number of local processes to spawn on the current node.",
    )
    parser.add_argument(
        "--num-ranks-per-server",
        type=int,
        default=None,
        help=(
            "Number of local ranks per server. Defaults to --num-processes and "
            "must match it because this script uses torch.multiprocessing.spawn."
        ),
    )
    parser.add_argument(
        "--num-servers",
        type=int,
        default=None,
        help="Number of servers. Mapped to WORLD_SIZE for utils.init_dist.",
    )
    parser.add_argument(
        "--server-index",
        type=int,
        default=None,
        help="Current server index. Mapped to RANK for utils.init_dist.",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=64,
        help="Token count per rank. Must be identical across all ranks.",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=4096,
        help="Hidden size. Recommended to keep it divisible by 512 for A2/A3 mega_moe.",
    )
    parser.add_argument(
        "--moe-intermediate-size",
        type=int,
        default=1024,
        help="Intermediate hidden size per expert.",
    )
    parser.add_argument(
        "--num-topk",
        type=int,
        default=6,
        help="Top-k experts per token.",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=16,
        help="Global number of experts. Must be divisible by world_size.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="a8w8_int,a8w4_int",
        help="Comma-separated scene list: a16w16, a8w8_int, a8w4_int.",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="swiglu,situ",
        help="Comma-separated activation list: swiglu,situ.",
    )
    parser.add_argument(
        "--linear-beta",
        type=str,
        default="0,1.25",
        help="Comma-separated linear_beta list. 0 means None.",
    )
    parser.add_argument(
        "--activation-clamp",
        type=str,
        default="0",
        help="Comma-separated activation_clamp list. 0 means None.",
    )
    parser.add_argument(
        "--beta",
        type=str,
        default="1.0",
        help="Comma-separated beta list for situ activation.",
    )
    parser.add_argument(
        "--enable-performance",
        action="store_true",
        default=False,
        help="Run repeated stress iterations without collecting performance timings.",
    )
    parser.add_argument(
        "--performance-iterations",
        type=int,
        default=10,
        help="Number of repeated stress iterations when --enable-performance is set.",
    )
    parser.add_argument(
        "--active-ranks",
        type=str,
        default="",
        help="Optional comma-separated rank ids that may receive tokens.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Base random seed.",
    )
    parser.add_argument(
        "--w8-w4-compare-threshold",
        type=float,
        default=5e-2,
        help=(
            "Warning threshold for cross-scene W8A8 vs W4A8 calc_diff. "
            "The comparison is reported only when both scenes are enabled."
        ),
    )
    parser.add_argument(
        "--master-addr",
        type=str,
        default=None,
        help="Set MASTER_ADDR before spawning local worker processes.",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=None,
        help="Set MASTER_PORT before spawning local worker processes.",
    )
    parser.add_argument(
        "--hccl-buffsize",
        type=int,
        default=None,
        help="Set HCCL_BUFFSIZE before spawning.",
    )
    cli_args = parser.parse_args()
    if cli_args.num_ranks_per_server is None:
        cli_args.num_ranks_per_server = cli_args.num_processes
    if cli_args.num_ranks_per_server != cli_args.num_processes:
        raise ValueError(
            "--num-ranks-per-server must equal --num-processes because this "
            "script starts one local worker per process."
        )
    apply_runtime_env_from_args(cli_args)
    torch.multiprocessing.spawn(
        test_loop,
        args=(cli_args.num_ranks_per_server, cli_args),
        nprocs=cli_args.num_processes,
    )
