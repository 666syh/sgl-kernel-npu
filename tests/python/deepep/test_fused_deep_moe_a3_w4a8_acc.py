import argparse
import os
import random
from typing import List, Optional

import deep_ep
import numpy as np
import torch
import torch.distributed as dist
import torch_npu
from deep_ep.buffer import FuseMode
from utils import calc_diff, init_dist

torch_npu.npu.config.allow_internal_format = True
os.environ["MOE_EXPERT_TOKEN_NUMS_TYPE"] = "1"

ACL_FORMAT_FRACTAL_NZ = torch_npu.Format.FRACTAL_NZ
W4A8_DIFF_WARNING_THRESHOLD = 5e-3
EXPERT_TOKEN_NUMS_TYPE_CUMSUM = 0
COMM_QUANT_MODE_INT8 = 2


def info_rank0(rank: int, message: str):
    if rank == 0:
        print(f"[rank0] {message}", flush=True)


def info_rank(rank: int, message: str):
    print(f"[rank{rank}] {message}", flush=True)


def warn_rank0(rank: int, message: str):
    if rank == 0:
        print(f"[rank0][WARNING] {message}", flush=True)


def stage_barrier(
    rank: int,
    stage: str,
    *,
    group: Optional[dist.ProcessGroup] = None,
    synchronize_npu: bool = False,
):
    if synchronize_npu:
        torch.npu.synchronize()
    info_rank(rank, f"{stage}: enter barrier")
    dist.barrier(group=group)
    info_rank(rank, f"{stage}: exit barrier")


def summarize_value(value) -> str:
    if isinstance(value, torch.Tensor):
        return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype})"
    if isinstance(value, (list, tuple)):
        parts = [summarize_value(item) for item in value[:4]]
        suffix = "" if len(value) <= 4 else ", ..."
        return f"{type(value).__name__}(len={len(value)}, items=[{', '.join(parts)}{suffix}])"
    return f"{type(value).__name__}({value})"


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


def parse_optional_float_list(raw: str) -> List[Optional[float]]:
    values: List[Optional[float]] = []
    for item in parse_csv_list(raw):
        value = float(item)
        values.append(None if value == 0 else value)
    return values


def parse_float_list(raw: str) -> List[float]:
    return [float(item) for item in parse_csv_list(raw)]


def swiglu_reference(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    gate = x[..., :d].to(torch.float32)
    up = x[..., d:].to(torch.float32)
    return (gate * torch.sigmoid(gate) * up).to(x.dtype)


def situ_reference(
    x: torch.Tensor,
    *,
    beta: float,
    linear_beta: Optional[float],
    activation_clamp: Optional[float],
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
        return torch_npu.npu_swiglu(x)
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
    num_topk: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device=device)
    topk_weights, topk_ids = torch.topk(
        scores, num_topk, dim=-1, largest=True, sorted=False
    )
    return topk_ids.to(torch.int32), topk_weights


def validate_topk_inputs(topk_ids_int32: torch.Tensor, num_experts: int):
    if topk_ids_int32.dtype != torch.int32:
        raise TypeError(f"topk_ids must be int32, got {topk_ids_int32.dtype}")
    if torch.any(topk_ids_int32 < 0) or torch.any(topk_ids_int32 >= num_experts):
        raise ValueError("topk_ids contains out-of-range expert indices.")
    sorted_ids = torch.sort(topk_ids_int32, dim=-1).values
    if torch.any(sorted_ids[:, 1:] == sorted_ids[:, :-1]):
        raise ValueError("topk_ids must be unique within each token's top-k row.")


def pack_scale_to_uint64(scale: torch.Tensor) -> torch.Tensor:
    return scale.contiguous().view(torch.int32).to(torch.int64).view(torch.uint64)


def pack_scale_to_int64(scale: torch.Tensor) -> torch.Tensor:
    return scale.contiguous().view(torch.int32).to(torch.int64)


def pack_int4(values: torch.Tensor, pack_dim: int) -> torch.Tensor:
    """Pack int8 values [-8,7] into int32 (8 values per int32)."""
    values = values.clamp(-8, 7)
    values_unsigned = torch.where(values < 0, values + 16, values).to(torch.int32)
    if pack_dim == 1:
        K, N_unpacked = values_unsigned.shape
        N_packed = N_unpacked // 8
        vals = values_unsigned.view(K, N_packed, 8)
    else:
        K_unpacked, N = values_unsigned.shape
        K_packed = K_unpacked // 8
        vals = values_unsigned.view(K_packed, 8, N).permute(0, 2, 1)
    packed = torch.zeros(vals.shape[0], vals.shape[1], dtype=torch.int32)
    for i in range(8):
        packed |= (vals[..., i].to(torch.int32) << (i * 4)).to(torch.int32)
    return packed


def pack_int4_experts_to_int8(weights: torch.Tensor) -> torch.Tensor:
    if weights.dim() != 3:
        raise ValueError(f"Expected expert-stacked 3D weight, got {weights.shape}.")
    if weights.shape[-1] % 8 != 0:
        raise ValueError(
            f"logical int4 weight last dim must be divisible by 8, got {weights.shape}."
        )
    packed = [
        pack_int4(weight.contiguous(), 1).view(torch.int8)
        for weight in weights.unbind(dim=0)
    ]
    return torch.stack(packed, dim=0).contiguous()


def pack_int4_to_int8(weight: torch.Tensor) -> torch.Tensor:
    if weight.shape[-1] % 2 != 0:
        raise ValueError(
            f"logical int4 weight last dim must be divisible by 2, got {weight.shape}."
        )
    packed = (
        (weight.to(torch.int16) + 8)
        .to(torch.uint8)
        .reshape(*weight.shape[:-1], weight.shape[-1] // 2, 2)
    )
    return ((packed[..., 1] << 4) | packed[..., 0]).view(torch.int8)


def normalize_expert_counts(
    counts, num_local_experts: int, device: torch.device
) -> torch.Tensor:
    if isinstance(counts, torch.Tensor):
        out = counts.to(device=device, dtype=torch.int64)
    else:
        out = torch.tensor(list(counts), device=device, dtype=torch.int64)
    if out.numel() != num_local_experts:
        raise ValueError(
            f"Expected {num_local_experts} local expert counts, got {out.numel()}."
        )
    return out


def process_scale(
    weight: torch.Tensor,
    scale: torch.Tensor,
    per_group_scale: torch.Tensor,
    *,
    new_quant_version: bool,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Process scale/bias for packed-int8 weights:
    #   l1: [E, hidden, intermediate_hidden]
    #   l2: [E, intermediate_hidden, hidden // 2]
    # where the last dim stores two INT4 values per INT8 element.
    scale = scale.transpose(1, 2).contiguous()
    per_group_scale = per_group_scale.transpose(1, 2).contiguous()

    group_num, k_dim, n_packed = weight.shape
    n_dim = n_packed * 2 if new_quant_version else n_packed
    per_group_scale = per_group_scale.reshape(group_num, -1, n_dim)
    _, quant_group_num, n_dim = per_group_scale.shape
    bias = torch.randn((group_num, n_dim), dtype=torch.float32, device=weight.device)

    scale_fp32 = (scale * per_group_scale).to(torch.float16).to(torch.float32)
    scale_fp32_np = scale_fp32.cpu().numpy()
    scale_fp32_np.dtype = np.uint32
    packed = np.zeros((group_num, quant_group_num, n_dim * 2), dtype=np.uint32)
    packed[..., ::2] = scale_fp32_np
    packed_buffer = np.frombuffer(packed.tobytes(), dtype=np.int64).copy()
    packed_tensor = (
        torch.from_numpy(packed_buffer).reshape(group_num, quant_group_num, n_dim).npu()
    )
    return packed_tensor, bias


def pack_to_int32(weight: torch.Tensor, *, new_quant_version: bool) -> torch.Tensor:
    if not new_quant_version:
        raise ValueError("This test only supports new_quant_version=True for A3 W4A8.")
    if weight.shape[-1] % 4 != 0:
        raise ValueError(
            f"Packed int4 weight last dim must be divisible by 4, got {weight.shape}."
        )
    return weight.view(torch.int32)

def unpack_int4(packed: torch.Tensor, pack_dim: int) -> torch.Tensor:
    """Unpack int32 → int8. Returns int8 tensor."""
    K, N = packed.shape
    shifts = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28], device=packed.device)
    vals = (packed.unsqueeze(-1) >> shifts) & 0xF
    vals = torch.where(vals >= 8, vals - 16, vals).to(torch.int8)
    if pack_dim == 1:
        return vals.reshape(K, -1)
    else:
        return vals.permute(0, 2, 1).reshape(-1, N)

def pack_float_into_int64(dequant_scale_origin: torch.Tensor) -> torch.Tensor:
    """Pack float32 scale into int64 with marker bit."""
    scale_int = dequant_scale_origin.view(torch.int32)
    scale_int = scale_int & 0xFFFFE000  # clear low 13 mantissa bits
    # Convert to int64 without sign-extension: view as unsigned first
    out = torch.bitwise_and(scale_int.to(torch.int64),
                            torch.tensor(0xFFFFFFFF, dtype=torch.int64))
    return out


def compute_bias(weight_packed: torch.Tensor, scale_packed: torch.Tensor,
                 pack_dim: int = 1) -> torch.Tensor:
    """Compute bias = sum(unpacked_weight * scale, dim=0) * 8."""
    w_int4 = unpack_int4(weight_packed, pack_dim).float()
    scale_fp32 = extract_float_from_int64(scale_packed)
    return (w_int4 * scale_fp32).sum(dim=0) * 8.0

def prepare_scene_weights(
    hidden: int,
    intermediate_hidden: int,
    num_local_experts: int,
    device: torch.device,
) -> dict:
    rank = dist.get_rank()
    barrier_group = dist.group.WORLD
    group_size = 256
    new_quant_version = True
    if hidden % group_size != 0:
        raise ValueError("W4A8 requires hidden divisible by 256.")
    if intermediate_hidden % group_size != 0:
        raise ValueError("W4A8 requires intermediate_hidden divisible by 256.")

    stage_barrier(rank, "prepare_scene_weights_pre", group=barrier_group)
    info_rank(rank, "prepare_scene_weights: raw logical int4 tensor generation start")
    # Generate logical int4 weights in int8 containers first. These tensors are
    # not packed yet; the last dim still represents the full output-channel dim.
    w13_weight_raw_int4 = torch.randint(
        -8,
        8,
        (num_local_experts, hidden, 2 * intermediate_hidden),
        dtype=torch.int8,
        device=device,
    )
    w2_weight_raw_int4 = torch.randint(
        -8,
        8,
        (num_local_experts, intermediate_hidden, hidden),
        dtype=torch.int8,
        device=device,
    )
    w13_weight_scale = torch.randn(
        (num_local_experts, 2 * intermediate_hidden, 1),
        dtype=torch.float32,
        device=device,
    )
    w2_weight_scale = torch.randn(
        (num_local_experts, hidden, 1), dtype=torch.float32, device=device
    )
    w13_weight_scale_second = torch.randn(
        (num_local_experts, 2 * intermediate_hidden, hidden // group_size),
        dtype=torch.float32,
        device=device,
    )
    w2_weight_scale_second = torch.randn(
        (num_local_experts, hidden, intermediate_hidden // group_size),
        dtype=torch.float32,
        device=device,
    )
    info_rank(
        rank,
        "prepare_scene_weights: raw logical int4 tensor generation done "
        f"w13_weight_raw_int4={tuple(w13_weight_raw_int4.shape)}/{w13_weight_raw_int4.dtype} "
        f"w2_weight_raw_int4={tuple(w2_weight_raw_int4.shape)}/{w2_weight_raw_int4.dtype} "
        f"w13_scale={tuple(w13_weight_scale.shape)}/{w13_weight_scale.dtype} "
        f"w2_scale={tuple(w2_weight_scale.shape)}/{w2_weight_scale.dtype}",
    )
    stage_barrier(
        rank,
        "prepare_scene_weights_raw_logical_int4_tensors",
        group=barrier_group,
        synchronize_npu=True,
    )

    info_rank(rank, "prepare_scene_weights: pack_int4_to_int8 start")
    w13_weight_packed_int8 = pack_int4_experts_to_int8(w13_weight_raw_int4)
    w2_weight_packed_int8 = pack_int4_experts_to_int8(w2_weight_raw_int4)
    expected_w13_packed_int8_shape = (
        num_local_experts,
        hidden,
        intermediate_hidden,
    )
    expected_w2_packed_int8_shape = (
        num_local_experts,
        intermediate_hidden,
        hidden // 2,
    )
    if tuple(w13_weight_packed_int8.shape) != expected_w13_packed_int8_shape:
        raise ValueError(
            f"Invalid packed int8 w13 shape: got {tuple(w13_weight_packed_int8.shape)}, "
            f"expected {expected_w13_packed_int8_shape}."
        )
    if tuple(w2_weight_packed_int8.shape) != expected_w2_packed_int8_shape:
        raise ValueError(
            f"Invalid packed int8 w2 shape: got {tuple(w2_weight_packed_int8.shape)}, "
            f"expected {expected_w2_packed_int8_shape}."
        )
    info_rank(
        rank,
        "prepare_scene_weights: pack_int4_to_int8 done "
        f"w13_weight_packed_int8={tuple(w13_weight_packed_int8.shape)}/{w13_weight_packed_int8.dtype} "
        f"w2_weight_packed_int8={tuple(w2_weight_packed_int8.shape)}/{w2_weight_packed_int8.dtype}",
    )
    stage_barrier(
        rank,
        "prepare_scene_weights_packed_int8_tensors",
        group=barrier_group,
        synchronize_npu=True,
    )

    info_rank(rank, "prepare_scene_weights: process_scale l1 start")
    w13_scale_int64, w13_bias = process_scale(
        w13_weight_packed_int8,
        w13_weight_scale,
        w13_weight_scale_second,
        new_quant_version=new_quant_version,
        group_size=group_size,
    )
    info_rank(
        rank,
        "prepare_scene_weights: process_scale l1 done "
        f"scale={tuple(w13_scale_int64.shape)}/{w13_scale_int64.dtype} "
        f"bias={tuple(w13_bias.shape)}/{w13_bias.dtype}",
    )
    stage_barrier(
        rank,
        "prepare_scene_weights_process_scale_l1",
        group=barrier_group,
        synchronize_npu=True,
    )

    info_rank(rank, "prepare_scene_weights: process_scale l2 start")
    w2_scale_int64, w2_bias = process_scale(
        w2_weight_packed_int8,
        w2_weight_scale,
        w2_weight_scale_second,
        new_quant_version=new_quant_version,
        group_size=group_size,
    )
    info_rank(
        rank,
        "prepare_scene_weights: process_scale l2 done "
        f"scale={tuple(w2_scale_int64.shape)}/{w2_scale_int64.dtype} "
        f"bias={tuple(w2_bias.shape)}/{w2_bias.dtype}",
    )
    stage_barrier(
        rank,
        "prepare_scene_weights_process_scale_l2",
        group=barrier_group,
        synchronize_npu=True,
    )

    info_rank(rank, "prepare_scene_weights: baseline stacked pack_to_int32 start")
    w13_weight_baseline_nz = torch_npu.npu_format_cast(
        w13_weight_packed_int8, ACL_FORMAT_FRACTAL_NZ
    )
    w2_weight_baseline_nz = torch_npu.npu_format_cast(
        w2_weight_packed_int8, ACL_FORMAT_FRACTAL_NZ
    )
    w13_weight_packed_stacked = pack_to_int32(
        w13_weight_baseline_nz, new_quant_version=new_quant_version
    )
    w2_weight_packed_stacked = pack_to_int32(
        w2_weight_baseline_nz, new_quant_version=new_quant_version
    )
    info_rank(
        rank,
        "prepare_scene_weights: baseline stacked NZ+pack_to_int32 done "
        f"w13={tuple(w13_weight_packed_stacked.shape)}/{w13_weight_packed_stacked.dtype} "
        f"w2={tuple(w2_weight_packed_stacked.shape)}/{w2_weight_packed_stacked.dtype}",
    )
    stage_barrier(
        rank,
        "prepare_scene_weights_baseline_stacked_pack",
        group=barrier_group,
        synchronize_npu=True,
    )

    baseline_l1_weight_stacked = [w13_weight_packed_stacked]
    baseline_l2_weight_stacked = [w2_weight_packed_stacked]
    baseline_l1_scale_stacked = [w13_scale_int64]
    baseline_l2_scale_stacked = [w2_scale_int64]
    baseline_l1_bias_stacked = [w13_bias.contiguous()]
    baseline_l2_bias_stacked = [w2_bias.contiguous()]

    info_rank(rank, "prepare_scene_weights: fused expert split/NZ/view l1 start")
    fused_l1_weights = [
        pack_to_int32(
            torch_npu.npu_format_cast(w.contiguous(), ACL_FORMAT_FRACTAL_NZ),
            new_quant_version=new_quant_version,
        )
        for w in w13_weight_packed_int8.unbind(dim=0)
    ]
    info_rank(
        rank,
        "prepare_scene_weights: fused expert split+NZ+view l1 done "
        f"num={len(fused_l1_weights)} first={tuple(fused_l1_weights[0].shape)}/{fused_l1_weights[0].dtype}",
    )
    stage_barrier(
        rank,
        "prepare_scene_weights_fused_l1_pack",
        group=barrier_group,
        synchronize_npu=True,
    )

    info_rank(rank, "prepare_scene_weights: fused expert split/NZ/view l2 start")
    fused_l2_weights = [
        pack_to_int32(
            torch_npu.npu_format_cast(w.contiguous(), ACL_FORMAT_FRACTAL_NZ),
            new_quant_version=new_quant_version,
        )
        for w in w2_weight_packed_int8.unbind(dim=0)
    ]
    info_rank(
        rank,
        "prepare_scene_weights: fused expert split+NZ+view l2 done "
        f"num={len(fused_l2_weights)} first={tuple(fused_l2_weights[0].shape)}/{fused_l2_weights[0].dtype}",
    )
    stage_barrier(
        rank,
        "prepare_scene_weights_fused_l2_pack",
        group=barrier_group,
        synchronize_npu=True,
    )

    info_rank(rank, "prepare_scene_weights: fused scale/bias list build start")
    fused_l1_scales = [
        w.reshape(-1).view(torch.uint64) for w in w13_scale_int64.unbind(dim=0)
    ]
    fused_l2_scales = [
        w.reshape(-1).view(torch.uint64) for w in w2_scale_int64.unbind(dim=0)
    ]
    fused_l1_bias = [w.reshape(-1) for w in w13_bias.unbind(dim=0)]
    fused_l2_bias = [w.reshape(-1) for w in w2_bias.unbind(dim=0)]
    info_rank(
        rank,
        "prepare_scene_weights: fused scale/bias list build done "
        f"l1_scale0={tuple(fused_l1_scales[0].shape)}/{fused_l1_scales[0].dtype} "
        f"l2_scale0={tuple(fused_l2_scales[0].shape)}/{fused_l2_scales[0].dtype} "
        f"l1_bias0={tuple(fused_l1_bias[0].shape)}/{fused_l1_bias[0].dtype} "
        f"l2_bias0={tuple(fused_l2_bias[0].shape)}/{fused_l2_bias[0].dtype}",
    )
    stage_barrier(
        rank,
        "prepare_scene_weights_fused_lists",
        group=barrier_group,
        synchronize_npu=True,
    )

    expected_l1_shape = (hidden, (2 * intermediate_hidden) // 8)
    expected_l2_shape = (intermediate_hidden, hidden // 8)
    info_rank(rank, "prepare_scene_weights: fused shape validation start")
    for weight in fused_l1_weights:
        if weight.dtype != torch.int32 or tuple(weight.shape) != expected_l1_shape:
            raise ValueError(
                f"Invalid fused l1 weight shape/dtype: got {tuple(weight.shape)} {weight.dtype}, "
                f"expected {expected_l1_shape} torch.int32."
            )
    for weight in fused_l2_weights:
        if weight.dtype != torch.int32 or tuple(weight.shape) != expected_l2_shape:
            raise ValueError(
                f"Invalid fused l2 weight shape/dtype: got {tuple(weight.shape)} {weight.dtype}, "
                f"expected {expected_l2_shape} torch.int32."
            )
    info_rank(rank, "prepare_scene_weights: fused shape validation done")
    stage_barrier(
        rank,
        "prepare_scene_weights_post",
        group=barrier_group,
        synchronize_npu=True,
    )

    return {
        "baseline_l1_weight_stacked": baseline_l1_weight_stacked,
        "baseline_l2_weight_stacked": baseline_l2_weight_stacked,
        "baseline_l1_scale_stacked": baseline_l1_scale_stacked,
        "baseline_l2_scale_stacked": baseline_l2_scale_stacked,
        "baseline_l1_bias_stacked": baseline_l1_bias_stacked,
        "baseline_l2_bias_stacked": baseline_l2_bias_stacked,
        "fused_l1_weights": fused_l1_weights,
        "fused_l2_weights": fused_l2_weights,
        "fused_l1_scales": fused_l1_scales,
        "fused_l2_scales": fused_l2_scales,
        "fused_l1_bias": fused_l1_bias,
        "fused_l2_bias": fused_l2_bias,
        "dispatch_quant_mode": 2,
        "dispatch_quant_out_dtype": torch.int8,
    }


def run_grouped_matmul_w4a8(
    x_int8: torch.Tensor,
    per_token_scale: torch.Tensor,
    group_list: torch.Tensor,
    weight: List[torch.Tensor],
    scale: List[torch.Tensor],
    bias,
    rank: int,
    stage_name: str,
    barrier_group: dist.ProcessGroup,
) -> torch.Tensor:
    stage_barrier(rank, f"{stage_name}_pre", group=barrier_group)
    bias_list = bias if isinstance(bias, list) else [bias]
    bias_tensor = bias_list[0]
    info_rank(
        rank,
        f"{stage_name}: x={tuple(x_int8.shape)}/{x_int8.dtype} "
        f"per_token_scale={tuple(per_token_scale.shape)}/{per_token_scale.dtype} "
        f"group_list={tuple(group_list.shape)}/{group_list.dtype} "
        f"weight0={tuple(weight[0].shape)}/{weight[0].dtype} "
        f"scale0={tuple(scale[0].shape)}/{scale[0].dtype} "
        f"bias={tuple(bias_tensor.shape)}/{bias_tensor.dtype}",
    )
    out = torch_npu.npu_grouped_matmul(
        x=[x_int8],
        weight=weight,
        scale=[scale[0].to(scale[0].dtype)],
        bias=bias_list,
        per_token_scale=[per_token_scale],
        split_item=2,
        group_list_type=EXPERT_TOKEN_NUMS_TYPE_CUMSUM,
        group_type=0,
        group_list=group_list,
        output_dtype=torch.bfloat16,
    )[0]
    info_rank(rank, f"{stage_name}: out={tuple(out.shape)}/{out.dtype}")
    stage_barrier(
        rank,
        f"{stage_name}_post",
        group=barrier_group,
        synchronize_npu=True,
    )
    return out


def run_mc2_dispatch(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    *,
    num_experts: int,
    global_bs: int,
    group_ep: str,
    barrier_group: dist.ProcessGroup,
    ep_world_size: int,
    ep_rank_id: int,
    enable_dispatch_v2: bool,
):
    stage_barrier(ep_rank_id, "dispatch_pre", group=barrier_group)
    info_rank(
        ep_rank_id,
        f"dispatch: x={tuple(x.shape)}/{x.dtype} "
        f"topk_idx={tuple(topk_idx.shape)}/{topk_idx.dtype} "
        f"global_bs={global_bs} "
        f"group_ep={group_ep} ep_world_size={ep_world_size}",
    )
    kwargs = {
        "x": x,
        "expert_ids": topk_idx,
        "expert_shard_type": 0,
        "shared_expert_rank_num": 0,
        "moe_expert_num": num_experts,
        "global_bs": global_bs,
        "x_active_mask": None,
        "quant_mode": COMM_QUANT_MODE_INT8,
        "scales": None,
        "group_ep": group_ep,
        "ep_world_size": ep_world_size,
        "ep_rank_id": ep_rank_id,
        "expert_token_nums_type": EXPERT_TOKEN_NUMS_TYPE_CUMSUM,
    }
    output = (
        torch_npu.npu_moe_distribute_dispatch_v2(**kwargs)
        if enable_dispatch_v2
        else torch_npu.npu_moe_distribute_dispatch(**kwargs)
    )
    info_rank(
        ep_rank_id,
        f"dispatch: recv_x={tuple(output[0].shape)}/{output[0].dtype} "
        f"recv_x_scale={tuple(output[1].shape)}/{output[1].dtype} "
        f"assist_info={summarize_value(output[2])} "
        f"group_list={tuple(output[3].shape)}/{output[3].dtype} "
        f"ep_recv_counts={tuple(output[4].shape)}/{output[4].dtype} "
        f"tp_recv_counts={tuple(output[5].shape)}/{output[5].dtype} "
        f"expand_scales={tuple(output[6].shape)}/{output[6].dtype}",
    )
    stage_barrier(
        ep_rank_id,
        "dispatch_post",
        group=barrier_group,
        synchronize_npu=True,
    )
    return output[0:7]


def run_mc2_combine(
    mlp_out: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    global_bs: int,
    ep_recv_counts: torch.Tensor,
    tp_recv_counts: torch.Tensor,
    assist_info_for_combine,
    expand_scales: torch.Tensor,
    group_ep: str,
    barrier_group: dist.ProcessGroup,
    ep_world_size: int,
    ep_rank_id: int,
    enable_dispatch_v2: bool,
) -> torch.Tensor:
    stage_barrier(ep_rank_id, "combine_pre", group=barrier_group)
    info_rank(
        ep_rank_id,
        f"combine: expand_x={tuple(mlp_out.shape)}/{mlp_out.dtype} "
        f"topk_idx={tuple(topk_idx.shape)}/{topk_idx.dtype} "
        f"topk_weights={tuple(topk_weights.shape)}/{topk_weights.dtype} "
        f"ep_recv_counts={tuple(ep_recv_counts.shape)}/{ep_recv_counts.dtype} "
        f"tp_recv_counts={tuple(tp_recv_counts.shape)}/{tp_recv_counts.dtype} "
        f"expand_scales={tuple(expand_scales.shape)}/{expand_scales.dtype} "
        f"assist_info={summarize_value(assist_info_for_combine)} "
        f"global_bs={global_bs} group_ep={group_ep} ep_world_size={ep_world_size}",
    )
    kwargs = {
        "expand_x": mlp_out,
        "expert_ids": topk_idx,
        "expert_scales": topk_weights.to(torch.float32),
        "expert_shard_type": 0,
        "shared_expert_rank_num": 0,
        "moe_expert_num": num_experts,
        "global_bs": global_bs,
        "ep_send_counts": ep_recv_counts,
        "group_ep": group_ep,
        "ep_world_size": ep_world_size,
        "ep_rank_id": ep_rank_id,
        "expand_scales": expand_scales,
        "comm_quant_mode": 0,
    }
    if enable_dispatch_v2:
        kwargs["assist_info_for_combine"] = assist_info_for_combine
        out = torch_npu.npu_moe_distribute_combine_v2(**kwargs)
    else:
        kwargs["expand_idx"] = assist_info_for_combine
        out = torch_npu.npu_moe_distribute_combine(**kwargs)
    info_rank(ep_rank_id, f"combine: out={tuple(out.shape)}/{out.dtype}")
    stage_barrier(
        ep_rank_id,
        "combine_post",
        group=barrier_group,
        synchronize_npu=True,
    )
    return out


def run_baseline_reference(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    weights: dict,
    activation: str,
    beta: float,
    linear_beta: Optional[float],
    activation_clamp: Optional[float],
    group_ep: str,
    barrier_group: dist.ProcessGroup,
    ep_world_size: int,
    ep_rank_id: int,
    enable_dispatch_v2: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    info_rank(ep_rank_id, "baseline: start")
    global_bs = x.size(0) * ep_world_size
    (
        recv_x_int8,
        recv_x_scale,
        assist_info_for_combine,
        group_list,
        ep_recv_counts,
        tp_recv_counts,
        expand_scales,
    ) = run_mc2_dispatch(
        x,
        topk_idx,
        num_experts=num_experts,
        global_bs=global_bs,
        group_ep=group_ep,
        barrier_group=barrier_group,
        ep_world_size=ep_world_size,
        ep_rank_id=ep_rank_id,
        enable_dispatch_v2=enable_dispatch_v2,
    )

    gate_up = run_grouped_matmul_w4a8(
        recv_x_int8,
        recv_x_scale,
        group_list,
        weights["baseline_l1_weight_stacked"],
        weights["baseline_l1_scale_stacked"],
        weights["baseline_l1_bias_stacked"],
        ep_rank_id,
        "gmm1",
        barrier_group,
    )
    stage_barrier(ep_rank_id, "activation_pre", group=barrier_group)
    info_rank(ep_rank_id, f"activation_in: {tuple(gate_up.shape)}/{gate_up.dtype}")
    act_out = apply_activation(
        gate_up,
        activation,
        beta=beta,
        linear_beta=linear_beta,
        activation_clamp=activation_clamp,
    )
    info_rank(ep_rank_id, f"activation_out: {tuple(act_out.shape)}/{act_out.dtype}")
    stage_barrier(
        ep_rank_id,
        "activation_post",
        group=barrier_group,
        synchronize_npu=True,
    )
    stage_barrier(ep_rank_id, "dynamic_quant_pre", group=barrier_group)
    act_int8, act_scale = torch_npu.npu_dynamic_quant(act_out)
    info_rank(
        ep_rank_id,
        f"dynamic_quant: act_int8={tuple(act_int8.shape)}/{act_int8.dtype} "
        f"act_scale={tuple(act_scale.shape)}/{act_scale.dtype}",
    )
    stage_barrier(
        ep_rank_id,
        "dynamic_quant_post",
        group=barrier_group,
        synchronize_npu=True,
    )
    down = run_grouped_matmul_w4a8(
        act_int8,
        act_scale,
        group_list,
        weights["baseline_l2_weight_stacked"],
        weights["baseline_l2_scale_stacked"],
        weights["baseline_l2_bias_stacked"],
        ep_rank_id,
        "gmm2",
        barrier_group,
    )
    combined_x = run_mc2_combine(
        down,
        topk_idx,
        topk_weights,
        num_experts=num_experts,
        global_bs=global_bs,
        ep_recv_counts=ep_recv_counts,
        tp_recv_counts=tp_recv_counts,
        assist_info_for_combine=assist_info_for_combine,
        expand_scales=expand_scales,
        group_ep=group_ep,
        barrier_group=barrier_group,
        ep_world_size=ep_world_size,
        ep_rank_id=ep_rank_id,
        enable_dispatch_v2=enable_dispatch_v2,
    )
    group_counts = torch.cat([group_list[:1], torch.diff(group_list, dim=0)])
    info_rank(ep_rank_id, "baseline: end")
    return combined_x, normalize_expert_counts(
        group_counts,
        num_experts // dist.get_world_size(),
        x.device,
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
    beta: float,
    linear_beta: Optional[float],
    activation_clamp: Optional[float],
    barrier_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor]:
    rank = dist.get_rank()
    stage_barrier(rank, "fused_pre", group=barrier_group)
    info_rank(
        rank,
        f"fused: x={tuple(x.shape)}/{x.dtype} "
        f"topk_idx={tuple(topk_idx.shape)}/{topk_idx.dtype} "
        f"topk_weights={tuple(topk_weights.shape)}/{topk_weights.dtype} "
        f"l1_num={len(weights['fused_l1_weights'])} l2_num={len(weights['fused_l2_weights'])}",
    )
    out = buffer.fused_deep_moe(
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
    info_rank(
        rank,
        f"fused: out={tuple(out[0].shape)}/{out[0].dtype} "
        f"counts={tuple(out[1].shape)}/{out[1].dtype}",
    )
    stage_barrier(rank, "fused_post", group=barrier_group, synchronize_npu=True)
    return out


def build_case_matrix(args: argparse.Namespace) -> list[dict]:
    activations = parse_csv_list(args.activation)
    betas = parse_float_list(args.beta)
    linear_betas = parse_optional_float_list(args.linear_beta)
    activation_clamps = parse_optional_float_list(args.activation_clamp)

    cases = []
    for activation in activations:
        if activation == "swiglu":
            cases.append(
                {
                    "activation": "swiglu",
                    "beta": 1.0,
                    "linear_beta": None,
                    "activation_clamp": None,
                }
            )
            continue
        if activation != "situ":
            raise ValueError(f"Unsupported activation case: {activation}")
        for beta in betas:
            for linear_beta in linear_betas:
                for activation_clamp in activation_clamps:
                    cases.append(
                        {
                            "activation": "situ",
                            "beta": beta,
                            "linear_beta": linear_beta,
                            "activation_clamp": activation_clamp,
                        }
                    )
    return cases


def run_case(
    args: argparse.Namespace,
    rank: int,
    num_ranks: int,
    fused_buffer: deep_ep.Buffer,
    case: dict,
    iteration: int,
    baseline_group_ep: str,
    baseline_group: dist.ProcessGroup,
    mega_group: dist.ProcessGroup,
    enable_dispatch_v2: bool,
):
    device = torch.device(f"npu:{torch.npu.current_device()}")
    num_local_experts = args.num_experts // num_ranks

    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    info_rank(
        rank,
        f"run_case: start iter={iteration} activation={case['activation']} "
        f"beta={case['beta']} linear_beta={case['linear_beta']} "
        f"activation_clamp={case['activation_clamp']}",
    )
    stage_barrier(rank, "run_case_pre", group=baseline_group)

    x = torch.randn((args.num_tokens, args.hidden), dtype=torch.bfloat16, device=device)
    topk_idx, topk_weights = make_topk_inputs(
        args.num_tokens, args.num_experts, args.num_topk, device
    )
    validate_topk_inputs(topk_idx, args.num_experts)
    info_rank(
        rank,
        f"run_case: inputs ready x={tuple(x.shape)}/{x.dtype} "
        f"topk_idx={tuple(topk_idx.shape)}/{topk_idx.dtype} "
        f"topk_weights={tuple(topk_weights.shape)}/{topk_weights.dtype}",
    )
    stage_barrier(
        rank, "run_case_inputs_ready", group=baseline_group, synchronize_npu=True
    )

    info_rank(rank, "run_case: prepare_scene_weights start")
    weights = prepare_scene_weights(
        args.hidden,
        args.moe_intermediate_size,
        num_local_experts,
        device,
    )
    info_rank(rank, "run_case: prepare_scene_weights done")
    stage_barrier(
        rank, "run_case_weights_ready", group=baseline_group, synchronize_npu=True
    )

    info_rank(rank, "run_case: baseline start")
    baseline_out, baseline_counts = run_baseline_reference(
        x,
        topk_idx,
        topk_weights,
        args.num_experts,
        weights,
        case["activation"],
        case["beta"],
        case["linear_beta"],
        case["activation_clamp"],
        baseline_group_ep,
        baseline_group,
        num_ranks,
        rank,
        enable_dispatch_v2,
    )
    info_rank(
        rank,
        f"run_case: baseline done out={tuple(baseline_out.shape)}/{baseline_out.dtype} "
        f"counts={tuple(baseline_counts.shape)}/{baseline_counts.dtype}",
    )
    stage_barrier(
        rank, "run_case_baseline_done", group=baseline_group, synchronize_npu=True
    )

    info_rank(rank, "run_case: fused start")
    fused_out, fused_counts = run_fused_reference(
        fused_buffer,
        x,
        topk_idx,
        topk_weights,
        args.num_tokens,
        args.num_experts,
        weights,
        case["activation"],
        case["beta"],
        case["linear_beta"],
        case["activation_clamp"],
        mega_group,
    )
    info_rank(
        rank,
        f"run_case: fused done out={tuple(fused_out.shape)}/{fused_out.dtype} "
        f"counts={tuple(fused_counts.shape)}/{fused_counts.dtype}",
    )
    stage_barrier(rank, "run_case_fused_done", group=mega_group, synchronize_npu=True)

    local_diff = calc_diff(baseline_out.float(), fused_out.float())
    diff_tensor = torch.tensor(local_diff, dtype=torch.float32, device=device)
    dist.all_reduce(diff_tensor, op=dist.ReduceOp.MAX)

    counts_match = torch.equal(
        normalize_expert_counts(baseline_counts, num_local_experts, device),
        normalize_expert_counts(fused_counts, num_local_experts, device),
    )
    counts_mismatch = torch.tensor(
        0 if counts_match else 1, dtype=torch.int32, device=device
    )
    dist.all_reduce(counts_mismatch, op=dist.ReduceOp.MAX)

    case_desc = (
        f"iter={iteration} activation={case['activation']} beta={case['beta']} "
        f"linear_beta={case['linear_beta']} activation_clamp={case['activation_clamp']}"
    )
    if counts_mismatch.item() != 0:
        warn_rank0(rank, f"{case_desc} expert_token_nums mismatch across ranks.")
    max_diff = diff_tensor.item()
    if max_diff >= W4A8_DIFF_WARNING_THRESHOLD:
        warn_rank0(
            rank,
            f"{case_desc} calc_diff={max_diff} exceeds threshold={W4A8_DIFF_WARNING_THRESHOLD}",
        )
    else:
        info_rank0(
            rank,
            f"{case_desc} calc_diff={max_diff} token_counts_match={counts_mismatch.item() == 0}",
        )
    info_rank(rank, f"run_case: end iter={iteration} diff={max_diff}")
    dist.barrier()


def test_main(
    args: argparse.Namespace,
    rank: int,
    num_ranks: int,
    fused_buffer: deep_ep.Buffer,
    baseline_group_ep: str,
    mega_group_ep: str,
    baseline_group: dist.ProcessGroup,
    mega_group: dist.ProcessGroup,
    enable_dispatch_v2: bool,
):
    if args.num_experts % num_ranks != 0:
        raise ValueError("num_experts must be divisible by world_size.")
    if args.enable_performance and args.performance_iters <= 0:
        raise ValueError("--performance-iters must be greater than zero.")

    iterations = args.performance_iters if args.enable_performance else 1
    cases = build_case_matrix(args)
    info_rank0(
        rank,
        f"baseline_group_ep={baseline_group_ep} mega_group_ep={mega_group_ep}",
    )
    for iteration in range(iterations):
        for case in cases:
            run_case(
                args,
                rank,
                num_ranks,
                fused_buffer,
                case,
                iteration,
                baseline_group_ep,
                baseline_group,
                mega_group,
                enable_dispatch_v2,
            )
    info_rank0(rank, "A3 W4A8 fused_deep_moe accuracy test completed")


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, _ = init_dist(local_rank, num_local_ranks)
    ranks = list(range(num_ranks))
    baseline_group = dist.new_group(ranks=ranks, backend="hccl")
    mega_group = dist.new_group(ranks=ranks, backend="hccl")
    baseline_backend = baseline_group._get_backend(torch.device("npu"))
    mega_backend = mega_group._get_backend(torch.device("npu"))
    baseline_group_ep = baseline_backend.get_hccl_comm_name(rank)
    mega_group_ep = mega_backend.get_hccl_comm_name(rank)
    enable_dispatch_v2 = hasattr(torch_npu, "npu_moe_distribute_dispatch_v2")

    fused_buffer = deep_ep.Buffer(
        mega_group,
        int(2e9),
        0,
        low_latency_mode=False,
        num_qps_per_rank=1,
    )

    test_main(
        args,
        rank,
        num_ranks,
        fused_buffer,
        baseline_group_ep,
        mega_group_ep,
        baseline_group,
        mega_group,
        enable_dispatch_v2,
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A3 W4A8 fused_deep_moe vs MC2 small-op accuracy test."
    )
    parser.add_argument("--num-processes", type=int, default=16)
    parser.add_argument("--num-servers", type=int, default=1)
    parser.add_argument("--server-index", type=int, default=0)
    parser.add_argument("--master-addr", type=str, default=None)
    parser.add_argument("--master-port", type=str, default=None)
    parser.add_argument("--hccl-buffsize", type=int, default=None)
    parser.add_argument("--num-tokens", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--moe-intermediate-size", type=int, default=3072)
    parser.add_argument("--num-topk", type=int, default=6)
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--activation", type=str, default="swiglu,situ")
    parser.add_argument("--beta", type=str, default="4.0")
    parser.add_argument("--linear-beta", type=str, default="25.0")
    parser.add_argument("--activation-clamp", type=str, default="0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--enable-performance", action="store_true")
    parser.add_argument("--performance-iters", type=int, default=30)
    args = parser.parse_args()

    apply_runtime_env_from_args(args)
    torch.multiprocessing.spawn(
        test_loop,
        args=(args.num_processes, args),
        nprocs=args.num_processes,
    )
