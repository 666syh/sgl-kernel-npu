import os

import torch
import torch.distributed as dist
import torch_npu
from cann_ops_transformer.ops import get_symm_buffer_for_mega_moe, mega_moe
from utils import calc_diff

num_tokens = 64
hidden = 4096
intermediate_hidden = 1024
num_topk = 6
num_experts = 16
num_max_tokens_per_rank = 256
num_servers = 1
num_ranks_per_server = 2
server_index = 0
master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
master_port = int(os.getenv("MASTER_PORT", "50001"))
seed = 2026
w8_w4_diff_warning_threshold = 5e-2

world_size = num_servers * num_ranks_per_server
num_experts_per_rank = num_experts // world_size

torch_npu.npu.config.allow_internal_format = True


def log_rank0(rank: int, message: str):
    if rank == 0:
        print(f"[rank0] {message}", flush=True)


def warn_rank0(rank: int, message: str):
    if rank == 0:
        print(f"[rank0][WARNING] {message}", flush=True)


def tensor_sig(tensor: torch.Tensor) -> str:
    return f"shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}"


def tensor_list_sig(tensors: list[torch.Tensor]) -> str:
    return "[" + ", ".join(tensor_sig(tensor) for tensor in tensors) + "]"


def init_hccl_comm(rank: int):
    global_rank = server_index * num_ranks_per_server + rank
    print(f"device_{rank} start init_process_group", flush=True)
    options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    options.hccl_config = {"hccl_buffer_size": 200}
    dist.init_process_group(
        backend="hccl",
        rank=global_rank,
        world_size=world_size,
        init_method=f"tcp://{master_addr}:{master_port}",
        pg_options=options,
    )
    print(f"device_{rank} init_process_group success", flush=True)
    ep_group = dist.new_group(backend="hccl", ranks=list(range(world_size)))
    _ = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    return ep_group


def make_shared_inputs():
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="npu")
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="npu")
    topk_weights, topk_ids = torch.topk(
        scores, num_topk, dim=-1, largest=True, sorted=False
    )
    return x, topk_ids.to(torch.int32), topk_weights


def make_shared_bf16_weights():
    l1_weights = [
        torch.randn(
            (hidden, intermediate_hidden * 2),
            dtype=torch.bfloat16,
            device="npu",
        )
        for _ in range(num_experts_per_rank)
    ]
    l2_weights = [
        torch.randn(
            (intermediate_hidden, hidden),
            dtype=torch.bfloat16,
            device="npu",
        )
        for _ in range(num_experts_per_rank)
    ]
    return l1_weights, l2_weights


def per_channel_cast_to_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    max_abs = torch.amax(torch.abs(w), dim=0, keepdim=True)
    sf = torch.clamp(max_abs / 127.0, min=1e-8)
    q_weight = torch.round(w / sf).clamp(-127, 127).to(torch.int8)
    return q_weight, sf.squeeze().to(torch.float32)


def per_channel_cast_to_int4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    max_abs = torch.amax(torch.abs(w), dim=0, keepdim=True)
    sf = torch.clamp(max_abs / 7.0, min=1e-8)
    q_weight = torch.round(w / sf).clamp(-7, 7).to(torch.int8)
    return q_weight, sf.squeeze().to(torch.float32)


def pack_scale_to_uint64(scale: torch.Tensor) -> torch.Tensor:
    return scale.contiguous().view(torch.int32).to(torch.int64).view(torch.uint64)


def pack_int4_to_int8(x: torch.Tensor) -> torch.Tensor:
    x = (x + 8).to(torch.uint8).reshape(x.shape[0], -1, 2)
    return ((x[..., 1] << 4) | x[..., 0]).view(torch.int8)


def prepare_w8a8_weights(
    l1_weights_bf16: list[torch.Tensor],
    l2_weights_bf16: list[torch.Tensor],
):
    l1_weights_int8, l1_weights_sf_float = map(
        list, zip(*[per_channel_cast_to_int8(w) for w in l1_weights_bf16])
    )
    l2_weights_int8, l2_weights_sf_float = map(
        list, zip(*[per_channel_cast_to_int8(w) for w in l2_weights_bf16])
    )
    l1_weights = [
        torch_npu.npu_format_cast(w, torch_npu.Format.FRACTAL_NZ)
        for w in l1_weights_int8
    ]
    l2_weights = [
        torch_npu.npu_format_cast(w, torch_npu.Format.FRACTAL_NZ)
        for w in l2_weights_int8
    ]
    l1_weights_sf = [pack_scale_to_uint64(sf) for sf in l1_weights_sf_float]
    l2_weights_sf = [pack_scale_to_uint64(sf) for sf in l2_weights_sf_float]
    return {
        "l1_weights": l1_weights,
        "l2_weights": l2_weights,
        "l1_weights_sf": l1_weights_sf,
        "l2_weights_sf": l2_weights_sf,
        "l1_bias": None,
        "l2_bias": None,
    }


def prepare_w4a8_weights(
    l1_weights_bf16: list[torch.Tensor],
    l2_weights_bf16: list[torch.Tensor],
):
    l1_weights_int8, l1_weights_sf_float = map(
        list, zip(*[per_channel_cast_to_int4(w) for w in l1_weights_bf16])
    )
    l2_weights_int8, l2_weights_sf_float = map(
        list, zip(*[per_channel_cast_to_int4(w) for w in l2_weights_bf16])
    )
    l1_weights_int4 = [pack_int4_to_int8(w) for w in l1_weights_int8]
    l2_weights_int4 = [pack_int4_to_int8(w) for w in l2_weights_int8]
    l1_weights = [
        torch_npu.npu_format_cast(w, torch_npu.Format.FRACTAL_NZ).view(torch.int32)
        for w in l1_weights_int4
    ]
    l2_weights = [
        torch_npu.npu_format_cast(w, torch_npu.Format.FRACTAL_NZ).view(torch.int32)
        for w in l2_weights_int4
    ]
    l1_weights_sf = [pack_scale_to_uint64(sf) for sf in l1_weights_sf_float]
    l2_weights_sf = [pack_scale_to_uint64(sf) for sf in l2_weights_sf_float]
    l1_bias = [
        (w.float() * sf.unsqueeze(0)).sum(dim=0) * 8.0
        for w, sf in zip(l1_weights_int8, l1_weights_sf_float)
    ]
    l2_bias = [
        (w.float() * sf.unsqueeze(0)).sum(dim=0) * 8.0
        for w, sf in zip(l2_weights_int8, l2_weights_sf_float)
    ]
    return {
        "l1_weights": l1_weights,
        "l2_weights": l2_weights,
        "l1_weights_sf": l1_weights_sf,
        "l2_weights_sf": l2_weights_sf,
        "l1_bias": l1_bias,
        "l2_bias": l2_bias,
    }


def run_quantized_mega_moe(
    label: str,
    rank: int,
    ep_group,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: dict,
):
    sym_buffer = get_symm_buffer_for_mega_moe(
        ep_group,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        dispatch_quant_mode=2,
        dispatch_quant_out_dtype=torch.int8,
    )
    log_rank0(
        rank,
        f"{label} enter x={tensor_sig(x)} topk_ids={tensor_sig(topk_ids)} "
        f"topk_weights={tensor_sig(topk_weights)} "
        f"l1_weights={tensor_list_sig(weights['l1_weights'])} "
        f"l2_weights={tensor_list_sig(weights['l2_weights'])} "
        f"l1_scales={tensor_list_sig(weights['l1_weights_sf'])} "
        f"l2_scales={tensor_list_sig(weights['l2_weights_sf'])}",
    )
    y, expert_token_nums = mega_moe(
        x,
        topk_ids,
        topk_weights,
        weights["l1_weights"],
        weights["l2_weights"],
        sym_buffer,
        l1_weights_sf=weights["l1_weights_sf"],
        l2_weights_sf=weights["l2_weights_sf"],
        l1_bias=weights["l1_bias"],
        l2_bias=weights["l2_bias"],
    )
    torch.npu.synchronize()
    log_rank0(
        rank,
        f"{label} exit y={tensor_sig(y)} expert_token_nums={tensor_sig(expert_token_nums)}",
    )
    return y, expert_token_nums


def compare_w8a8_w4a8_mega_moe(rank: int):
    torch_npu.npu.set_device(rank % num_ranks_per_server)
    global_rank = server_index * num_ranks_per_server + rank
    torch.manual_seed(seed + global_rank)

    ep_group = init_hccl_comm(rank)

    x, topk_ids, topk_weights = make_shared_inputs()
    l1_weights_bf16, l2_weights_bf16 = make_shared_bf16_weights()
    w8a8_weights = prepare_w8a8_weights(l1_weights_bf16, l2_weights_bf16)
    w4a8_weights = prepare_w4a8_weights(l1_weights_bf16, l2_weights_bf16)

    log_rank0(
        rank,
        "shared inputs ready "
        f"x={tensor_sig(x)} topk_ids={tensor_sig(topk_ids)} "
        f"topk_weights={tensor_sig(topk_weights)}",
    )

    w8a8_y, w8a8_expert_token_nums = run_quantized_mega_moe(
        "W8A8", rank, ep_group, x, topk_ids, topk_weights, w8a8_weights
    )
    dist.barrier()
    w4a8_y, w4a8_expert_token_nums = run_quantized_mega_moe(
        "W4A8", rank, ep_group, x, topk_ids, topk_weights, w4a8_weights
    )
    dist.barrier()

    output_diff = calc_diff(w8a8_y.float(), w4a8_y.float())
    counts_match = torch.equal(
        w8a8_expert_token_nums.cpu(), w4a8_expert_token_nums.cpu()
    )
    output_diff_tensor = torch.tensor(output_diff, dtype=torch.float32, device="npu")
    counts_mismatch_tensor = torch.tensor(
        0 if counts_match else 1, dtype=torch.int32, device="npu"
    )
    dist.all_reduce(output_diff_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(counts_mismatch_tensor, op=dist.ReduceOp.MAX)

    if counts_mismatch_tensor.item() != 0:
        warn_rank0(
            rank,
            "W8A8/W4A8 expert_token_nums mismatch on at least one rank. "
            "Current rank values: "
            f"W8A8={w8a8_expert_token_nums.tolist()} "
            f"W4A8={w4a8_expert_token_nums.tolist()}",
        )

    max_output_diff = output_diff_tensor.item()
    if max_output_diff >= w8_w4_diff_warning_threshold:
        warn_rank0(
            rank,
            f"W8A8 vs W4A8 output calc_diff max_across_ranks={max_output_diff} "
            f"exceeds threshold={w8_w4_diff_warning_threshold}",
        )
    else:
        log_rank0(
            rank,
            f"W8A8 vs W4A8 output calc_diff max_across_ranks={max_output_diff} "
            f"within threshold={w8_w4_diff_warning_threshold}",
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    torch.multiprocessing.spawn(
        compare_w8a8_w4a8_mega_moe,
        nprocs=num_ranks_per_server,
    )
