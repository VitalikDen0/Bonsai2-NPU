import os
import sys
import time
import struct
import numpy as np
import torch
from enum import IntEnum
import gguf.constants as gc
import gguf.gguf_reader as gr

# Register PTQ1_0 (type 143: 128 weights in 28 bytes)
members = {k: int(v) for k, v in gc.GGMLQuantizationType.__members__.items()}
members["PTQ1_0"] = 143
NewEnum = IntEnum("GGMLQuantizationType", members)
gc.GGMLQuantizationType = NewEnum
gr.GGMLQuantizationType = NewEnum
gc.GGML_QUANT_SIZES[NewEnum.PTQ1_0] = (128, 28)

GGUF_PATH = r"D:\Qualcomm\models\Bonsai-2-27B-PTQ1_0\Ternary-Bonsai-2-27B-PTQ1_0.gguf"
OUT_PATH  = r"D:\Download\Bonsai2_NPU\bonsai2-27b.npubin"

KIND_Q1      = 0
KIND_F16     = 1
KIND_TERNARY = 2
KIND_F32     = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[repack] Using device: {device}")
t0 = time.time()
reader = gr.GGUFReader(GGUF_PATH)
tmap = {t.name: t for t in reader.tensors}

def field_val(f):
    if not len(f.data): return None
    if f.types[0] == gc.GGUFValueType.STRING:
        return [bytes(f.parts[i]).decode("utf-8") for i in f.data]
    if f.types[0] == gc.GGUFValueType.ARRAY:
        elem = f.types[1]
        if elem == gc.GGUFValueType.STRING:
            return [bytes(f.parts[i]).decode("utf-8") for i in f.data]
        return [f.parts[i].tolist()[0] for i in f.data]
    return f.parts[f.data[0]].tolist()[0]

fields = {k: field_val(v) for k, v in reader.fields.items()}
sign_widths = [int(x) for x in fields["prism.hadamard.sign_widths"]]
sign_values = np.asarray(fields["prism.hadamard.sign_values"], dtype=np.float32)
signs = {}
cur = 0
for w in sign_widths:
    signs[w] = sign_values[cur : cur + w].copy()
    cur += w

nv, nk, hk, hd = 48, 16, 128, 128
def vperm(unit):
    return np.arange(nv * unit).reshape(nv // nk, nk, unit).transpose(1, 0, 2).reshape(-1)

PERM_HD = torch.from_numpy(vperm(hd)).to(device)
SHIFTS  = torch.arange(8, device=device, dtype=torch.uint8)

def reorder_np(a, stem):
    if stem == "attn_qkv.weight":
        qk = 2 * nk * hk
        return np.concatenate([a[:qk], a[qk:][vperm(hd)]], axis=0)
    if stem == "attn_gate.weight":
        return a[vperm(hd)]
    if stem in ("ssm_alpha.weight", "ssm_beta.weight", "ssm_a", "ssm_dt.bias"):
        return a[vperm(1)]
    if stem == "ssm_conv1d.weight":
        qk = 2 * nk * hk
        return np.concatenate([a[:qk], a[qk:][vperm(hd)]], axis=0)
    return a

def encode_aux_bytes(gguf_name, stem, target_kind):
    t = tmap[gguf_name]
    tn = t.tensor_type.name
    if tn == "F32":
        a = np.frombuffer(t.data.tobytes(), dtype=np.float32).copy()
    elif tn == "BF16":
        u16 = np.frombuffer(t.data.tobytes(), dtype=np.uint16)
        u32 = u16.astype(np.uint32) << 16
        a = u32.view(np.float32).copy()
    elif tn == "F16":
        a = np.frombuffer(t.data.tobytes(), dtype=np.float16).astype(np.float32).copy()
    else:
        raise ValueError(f"Unexpected aux type {tn} for {gguf_name}")
    shape = tuple(int(n) for n in t.shape[::-1])
    a = a.reshape(shape)
    if stem:
        a = reorder_np(a, stem)
    if stem == "ssm_a":
        a = np.log(-a)
    out_dim = shape[0]
    in_dim = shape[1] if len(shape) > 1 else 1
    if target_kind == KIND_F16:
        raw = np.ascontiguousarray(a.astype(np.float16)).tobytes()
    elif target_kind == KIND_F32:
        raw = np.ascontiguousarray(a.astype(np.float32)).tobytes()
    else:
        raise ValueError(f"Unsupported target_kind {target_kind}")
    return target_kind, out_dim, in_dim, raw

def pack_ptq1_chunk_gpu(u8_2d, rows, width):
    # u8_2d: (rows * ng, 28) uint8
    ng = width // 128
    data = torch.from_numpy(u8_2d).to(device)
    scales_bytes = u8_2d[:, 26:28].copy().reshape(rows, ng * 2)
    pieces = []
    for lo, hi, count in [(0, 16, 5), (16, 24, 5), (24, 26, 4)]:
        packed = data[:, lo:hi].to(torch.int32)
        for trit in range(count):
            rem = (packed * (3 ** trit)) & 255
            pieces.append((rem * 3) >> 8)
    codes = torch.cat(pieces, dim=1).to(torch.uint8).reshape(rows, width // 8, 8)
    pos_bits = (((codes == 2).to(torch.uint8)) << SHIFTS).sum(dim=-1, dtype=torch.uint8)
    neg_bits = (((codes == 0).to(torch.uint8)) << SHIFTS).sum(dim=-1, dtype=torch.uint8)
    packed_bits = torch.cat([pos_bits, neg_bits], dim=-1).cpu().numpy()
    return scales_bytes, packed_bits

def pack_ptq1_tensor(gguf_name, stem=""):
    t = tmap[gguf_name]
    rows, width = tuple(int(n) for n in t.shape[::-1])
    ng = width // 128
    prow = 2 * (width // 8)
    u8_all = np.ascontiguousarray(t.data.view(np.uint8)).reshape(rows, ng, 28)
    if stem == "attn_qkv.weight":
        qk = 2 * nk * hk
        u8_all = np.concatenate([u8_all[:qk], u8_all[qk:][vperm(hd)]], axis=0)
    elif stem == "attn_gate.weight":
        u8_all = u8_all[vperm(hd)]
    # Process in chunks of up to 32768 rows to keep GPU memory low
    chunk_rows = 32768
    scales_list = []
    bits_list = []
    for r0 in range(0, rows, chunk_rows):
        r1 = min(rows, r0 + chunk_rows)
        sub = np.ascontiguousarray(u8_all[r0:r1]).reshape((r1 - r0) * ng, 28)
        sc_b, pb_b = pack_ptq1_chunk_gpu(sub, r1 - r0, width)
        scales_list.append(sc_b)
        bits_list.append(pb_b)
    scales_full = np.concatenate(scales_list, axis=0)
    bits_full   = np.concatenate(bits_list, axis=0)
    assert scales_full.nbytes == rows * ng * 2
    assert bits_full.nbytes == rows * prow
    return KIND_TERNARY, rows, width, scales_full, bits_full

# Build complete tensor plan for the unified monolithic npubin
plan = []
# 1. Global tensors
plan.append(("language_model.model.embed_tokens", "ptq1", "token_embd.weight", ""))
plan.append(("language_model.model.norm.weight", "aux", "output_norm.weight", "", KIND_F32))
plan.append(("language_model.lm_head", "ptq1", "output.weight", ""))
for w in sign_widths:
    plan.append((f"prism.hadamard.signs.{w}", "signs", w, "", KIND_F32))

# 2. 64 Transformer layers
for L in range(64):
    gp = f"blk.{L}."
    np_pfx = f"language_model.model.layers.{L}."
    plan.append((np_pfx + "input_layernorm.weight", "aux", gp + "attn_norm.weight", "", KIND_F32))
    plan.append((np_pfx + "post_attention_layernorm.weight", "aux", gp + "post_attention_norm.weight", "", KIND_F32))
    plan.append((np_pfx + "mlp.gate_proj", "ptq1", gp + "ffn_gate.weight", "ffn_gate.weight"))
    plan.append((np_pfx + "mlp.up_proj", "ptq1", gp + "ffn_up.weight", "ffn_up.weight"))
    plan.append((np_pfx + "mlp.down_proj", "ptq1", gp + "ffn_down.weight", "ffn_down.weight"))
    if L % 4 != 3:
        plan.append((np_pfx + "linear_attn.in_proj_qkv", "ptq1", gp + "attn_qkv.weight", "attn_qkv.weight"))
        plan.append((np_pfx + "linear_attn.in_proj_z", "ptq1", gp + "attn_gate.weight", "attn_gate.weight"))
        plan.append((np_pfx + "linear_attn.in_proj_a", "aux", gp + "ssm_alpha.weight", "ssm_alpha.weight", KIND_F32))
        plan.append((np_pfx + "linear_attn.in_proj_b", "aux", gp + "ssm_beta.weight", "ssm_beta.weight", KIND_F32))
        plan.append((np_pfx + "linear_attn.conv1d.weight", "aux", gp + "ssm_conv1d.weight", "ssm_conv1d.weight", KIND_F16))
        plan.append((np_pfx + "linear_attn.A_log", "aux", gp + "ssm_a", "ssm_a", KIND_F32))
        plan.append((np_pfx + "linear_attn.dt_bias", "aux", gp + "ssm_dt.bias", "ssm_dt.bias", KIND_F32))
        plan.append((np_pfx + "linear_attn.norm.weight", "aux", gp + "ssm_norm.weight", "", KIND_F32))
        plan.append((np_pfx + "linear_attn.out_proj", "ptq1", gp + "ssm_out.weight", "ssm_out.weight"))
    else:
        plan.append((np_pfx + "self_attn.q_proj", "ptq1", gp + "attn_q.weight", "attn_q.weight"))
        plan.append((np_pfx + "self_attn.k_proj", "ptq1", gp + "attn_k.weight", "attn_k.weight"))
        plan.append((np_pfx + "self_attn.v_proj", "ptq1", gp + "attn_v.weight", "attn_v.weight"))
        plan.append((np_pfx + "self_attn.o_proj", "ptq1", gp + "attn_output.weight", "attn_output.weight"))
        plan.append((np_pfx + "self_attn.q_norm.weight", "aux", gp + "attn_q_norm.weight", "", KIND_F32))
        plan.append((np_pfx + "self_attn.k_norm.weight", "aux", gp + "attn_k_norm.weight", "", KIND_F32))

print(f"[repack] Total tensors in unified monolith plan: {len(plan)}")

# Compute exact header size and 128-byte aligned offsets first
header_size = 12
meta = []
for item in plan:
    name = item[0]
    mode = item[1]
    nb = name.encode("utf-8")
    header_size += 2 + len(nb) + 1 + 4 + 4 + 8 + 8
    if mode == "ptq1":
        t = tmap[item[2]]
        rows, width = tuple(int(n) for n in t.shape[::-1])
        ng = width // 128
        prow = 2 * (width // 8)
        byte_len = rows * ng * 2 + rows * prow
        meta.append((name, KIND_TERNARY, rows, width, byte_len))
    elif mode == "aux":
        t = tmap[item[2]]
        shape = tuple(int(n) for n in t.shape[::-1])
        out_dim = shape[0]
        in_dim = shape[1] if len(shape) > 1 else 1
        tkind = item[4]
        elem_sz = 2 if tkind == KIND_F16 else 4
        byte_len = out_dim * in_dim * elem_sz
        meta.append((name, tkind, out_dim, in_dim, byte_len))
    elif mode == "signs":
        w = item[2]
        meta.append((name, KIND_F32, w, 1, w * 4))

# Align data start and each tensor to 128 bytes
cur_off = (header_size + 127) & ~127
offsets = []
for name, tkind, out_dim, in_dim, byte_len in meta:
    cur_off = (cur_off + 127) & ~127
    offsets.append(cur_off)
    cur_off += byte_len

print(f"[repack] Writing single unified monolith {OUT_PATH} ({cur_off / (1024**3):.3f} GiB)...")
with open(OUT_PATH, "wb") as f:
    # Write header
    f.write(b"NPU1")
    f.write(struct.pack("<II", 0, len(plan)))
    for (name, tkind, out_dim, in_dim, byte_len), off in zip(meta, offsets):
        nb = name.encode("utf-8")
        f.write(struct.pack("<H", len(nb)))
        f.write(nb)
        f.write(struct.pack("<BIIQQ", tkind, out_dim, in_dim, off, byte_len))

    # Write tensor payloads
    for idx, (item, (name, tkind, out_dim, in_dim, byte_len), off) in enumerate(zip(plan, meta, offsets)):
        pos = f.tell()
        if pos < off:
            f.write(b"\x00" * (off - pos))
        mode = item[1]
        if mode == "ptq1":
            _, _, _, sc_arr, bits_arr = pack_ptq1_tensor(item[2], item[3])
            f.write(sc_arr.tobytes())
            f.write(bits_arr.tobytes())
        elif mode == "aux":
            _, _, _, raw = encode_aux_bytes(item[2], item[3], item[4])
            assert len(raw) == byte_len
            f.write(raw)
        elif mode == "signs":
            w = item[2]
            raw = np.ascontiguousarray(signs[w], dtype=np.float32).tobytes()
            assert len(raw) == byte_len
            f.write(raw)
        if (idx + 1) % 50 == 0 or idx + 1 == len(plan):
            print(f"  [{idx+1}/{len(plan)}] packed ({f.tell() / (1024**3):.2f} GiB, {time.time() - t0:.1f}s)")

print(f"[repack] DONE in {time.time() - t0:.1f}s! File: {OUT_PATH} ({os.path.getsize(OUT_PATH)} bytes)")
