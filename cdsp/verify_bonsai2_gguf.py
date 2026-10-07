import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from enum import IntEnum
import gguf.constants as gc
import gguf.gguf_reader as gr
from tokenizers import Tokenizer

sys.path.insert(0, r"D:\Download\Bonsai2_NPU\Ternary-Bonsai-2-27B-mlx-2bit\runtime")
from codec import transcode

members = {k: int(v) for k, v in gc.GGMLQuantizationType.__members__.items()}
members["PTQ1_0"] = 143
NewEnum = IntEnum("GGMLQuantizationType", members)
gc.GGMLQuantizationType = NewEnum
gr.GGMLQuantizationType = NewEnum
gc.GGML_QUANT_SIZES[NewEnum.PTQ1_0] = (128, 28)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading Ternary-Bonsai-2-27B-PTQ1_0.gguf on {device}...")
t0 = time.time()
reader = gr.GGUFReader(r"D:\Qualcomm\models\Bonsai-2-27B-PTQ1_0\Ternary-Bonsai-2-27B-PTQ1_0.gguf")
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
    signs[w] = torch.from_numpy(sign_values[cur : cur + w].copy()).to(device)
    cur += w

nv, nk, hk, hd = 48, 16, 128, 128
def vperm(unit):
    return np.arange(nv * unit).reshape(nv // nk, nk, unit).transpose(1, 0, 2).reshape(-1)

def reorder(a, stem):
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

def fwht1024(x, s_vec, inverse=False):
    # x: [..., W]
    orig_shape = x.shape
    W = orig_shape[-1]
    if not inverse:
        x = x * s_vec
    h = x.reshape(-1, W // 1024, 1024)
    # 10-stage Sylvester Walsh-Hadamard butterfly via reshape:
    for stage in range(10):
        half = 1 << stage
        h = h.reshape(-1, 1024 // (2 * half), 2, half)
        u = h[:, :, 0, :]
        v = h[:, :, 1, :]
        h = torch.stack([u + v, u - v], dim=2)
    out = h.reshape(orig_shape) * 0.03125
    if inverse:
        out = out * s_vec
    return out

def get_aux(name, stem=""):
    t = tmap[name]
    tn = t.tensor_type.name
    if tn == "F32":
        a = np.frombuffer(t.data.tobytes(), dtype=np.float32).copy()
    elif tn == "BF16":
        u16 = np.frombuffer(t.data.tobytes(), dtype=np.uint16)
        u32 = u16.astype(np.uint32) << 16
        a = u32.view(np.float32).copy()
    elif tn == "F16":
        a = np.frombuffer(t.data.tobytes(), dtype=np.float16).astype(np.float32).copy()
    shape = tuple(int(n) for n in t.shape[::-1])
    a = a.reshape(shape)
    if stem:
        a = reorder(a, stem)
    if stem == "ssm_a":
        a = np.log(-a)
    return torch.from_numpy(np.ascontiguousarray(a)).to(device)

def decode_ptq1_gpu(raw_u8, rows, width, stem=""):
    blocks = rows * (width // 128)
    u8_2d = np.ascontiguousarray(raw_u8).reshape(blocks, 28)
    data = torch.from_numpy(u8_2d).to(device)
    scales_np = u8_2d[:, 26:28].copy().view("<f2").reshape(rows, width // 128)
    pieces = []
    for lo, hi, count in [(0, 16, 5), (16, 24, 5), (24, 26, 4)]:
        packed = data[:, lo:hi].to(torch.int32)
        for trit in range(count):
            rem = (packed * (3 ** trit)) & 255
            pieces.append((rem * 3) >> 8)
    codes = torch.cat(pieces, dim=1).reshape(rows, width // 128, 128)
    scales = torch.from_numpy(scales_np).to(device, dtype=torch.float32)
    if stem == "attn_qkv.weight":
        qk = 2 * nk * hk
        perm = torch.from_numpy(vperm(hd)).to(device)
        codes = torch.cat([codes[:qk], codes[qk:][perm]], dim=0)
        scales = torch.cat([scales[:qk], scales[qk:][perm]], dim=0)
    elif stem == "attn_gate.weight":
        perm = torch.from_numpy(vperm(hd)).to(device)
        codes = codes[perm]
        scales = scales[perm]
    return ((codes.float() - 1.0) * scales.unsqueeze(-1)).reshape(rows, width)

def dequant_ptq1_rows(name, row_ids):
    t = tmap[name]
    rows, width = tuple(int(n) for n in t.shape[::-1])
    row_bytes = (width // 128) * 28
    u8 = t.data.view(np.uint8).reshape(rows, row_bytes)
    w = decode_ptq1_gpu(u8[row_ids], len(row_ids), width)
    return fwht1024(w, signs[width], inverse=True)

def gemm_ptq1(name, stem, X):
    t = tmap[name]
    rows, width = tuple(int(n) for n in t.shape[::-1])
    W = decode_ptq1_gpu(t.data.view(np.uint8), rows, width, stem)
    X_rot = fwht1024(X, signs[width], inverse=False)
    return torch.matmul(X_rot, W.t())

def rmsnorm(x, w, eps=1e-6):
    inv = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x * inv * w

def apply_rope(x, pos_ids):
    B, H, D = x.shape
    rot = 64
    idx = torch.arange(rot // 2, device=device, dtype=torch.float32)
    freq = 1.0 / (1e7 ** (2.0 * idx / rot))
    angles = pos_ids.unsqueeze(-1).float() * freq.unsqueeze(0)
    c = torch.cos(angles).unsqueeze(1)
    s = torch.sin(angles).unsqueeze(1)
    x0, x1 = x[:, :, :32], x[:, :, 32:64]
    out = x.clone()
    out[:, :, :32] = x0 * c - x1 * s
    out[:, :, 32:64] = x0 * s + x1 * c
    return out

tok = Tokenizer.from_file(r"D:\Download\Bonsai2_NPU\Bonsai-27B-mlx-1bit\tokenizer.json")
prompt = "The capital of France is"
ids = tok.encode(prompt).ids
print(f"Prompt: {prompt!r} -> ids={ids}")

H = dequant_ptq1_rows("token_embd.weight", ids)
B = H.shape[0]
pos_ids = torch.arange(B, device=device)

for L in range(64):
    pfx = f"blk.{L}."
    lw = get_aux(pfx + "attn_norm.weight")
    xn = rmsnorm(H, lw)
    if L % 4 != 3:
        qkv = gemm_ptq1(pfx + "attn_qkv.weight", "attn_qkv.weight", xn)
        z   = gemm_ptq1(pfx + "attn_gate.weight", "attn_gate.weight", xn)
        w_a = get_aux(pfx + "ssm_alpha.weight", "ssm_alpha.weight")
        w_b = get_aux(pfx + "ssm_beta.weight", "ssm_beta.weight")
        av  = torch.matmul(xn, w_a.t())
        bv  = torch.matmul(xn, w_b.t())
        cw  = get_aux(pfx + "ssm_conv1d.weight", "ssm_conv1d.weight").reshape(10240, 4)
        qkv_pad = F.pad(qkv.t().unsqueeze(0), (3, 0))
        conv_out = F.conv1d(qkv_pad, cw.unsqueeze(1), groups=10240)[0].t()
        qkv_act = F.silu(conv_out)
        q = qkv_act[:, :2048].reshape(B, 16, 128)
        k = qkv_act[:, 2048:4096].reshape(B, 16, 128)
        v = qkv_act[:, 4096:].reshape(B, 48, 128)
        q = q * torch.rsqrt(torch.sum(q * q, dim=-1, keepdim=True) + 1e-6) * (128 ** -0.5)
        k = k * torch.rsqrt(torch.sum(k * k, dim=-1, keepdim=True) + 1e-6)
        alog = get_aux(pfx + "ssm_a", "ssm_a")
        dtb  = get_aux(pfx + "ssm_dt.bias", "ssm_dt.bias")
        beta = torch.sigmoid(bv)
        gv   = -torch.exp(alog) * F.softplus(av + dtb)
        eg   = torch.exp(gv)
        S = torch.zeros(48, 128, 128, device=device)
        outs = []
        for t in range(B):
            qt = q[t].repeat_interleave(3, dim=0)
            kt = k[t].repeat_interleave(3, dim=0)
            vt = v[t]
            S = S * eg[t].view(48, 1, 1)
            kv = torch.einsum("hji,hi->hj", S, kt)
            delta = (vt - kv) * beta[t].unsqueeze(-1)
            S = S + torch.einsum("hj,hi->hji", delta, kt)
            ot = torch.einsum("hji,hi->hj", S, qt)
            outs.append(ot)
        out = torch.stack(outs, dim=0)
        nw = get_aux(pfx + "ssm_norm.weight")
        out_norm = rmsnorm(out, nw) * F.silu(z.reshape(B, 48, 128))
        yo = gemm_ptq1(pfx + "ssm_out.weight", "ssm_out.weight", out_norm.reshape(B, 6144))
        H = H + yo
    else:
        qq = gemm_ptq1(pfx + "attn_q.weight", "attn_q.weight", xn)
        kk = gemm_ptq1(pfx + "attn_k.weight", "attn_k.weight", xn)
        vv = gemm_ptq1(pfx + "attn_v.weight", "attn_v.weight", xn)
        qg = qq.reshape(B, 24, 2, 256)
        q, gate = qg[:, :, 0, :], qg[:, :, 1, :]
        qnw = get_aux(pfx + "attn_q_norm.weight")
        knw = get_aux(pfx + "attn_k_norm.weight")
        q = rmsnorm(q, qnw)
        k = rmsnorm(kk.reshape(B, 4, 256), knw)
        q = apply_rope(q, pos_ids)
        k = apply_rope(k, pos_ids)
        v = vv.reshape(B, 4, 256)
        k_rep = k.repeat_interleave(6, dim=1)
        v_rep = v.repeat_interleave(6, dim=1)
        scores = torch.einsum("thd,shd->hts", q, k_rep) / 16.0
        mask = torch.triu(torch.full((B, B), float("-inf"), device=device), diagonal=1)
        probs = torch.softmax(scores + mask.unsqueeze(0), dim=-1)
        ao = torch.einsum("hts,shd->thd", probs, v_rep)
        flat = (ao * torch.sigmoid(gate)).reshape(B, 6144)
        yo = gemm_ptq1(pfx + "attn_output.weight", "attn_output.weight", flat)
        H = H + yo

    pw = get_aux(pfx + "post_attention_norm.weight")
    n2 = rmsnorm(H, pw)
    g8 = gemm_ptq1(pfx + "ffn_gate.weight", "ffn_gate.weight", n2)
    up = gemm_ptq1(pfx + "ffn_up.weight", "ffn_up.weight", n2)
    dn = gemm_ptq1(pfx + "ffn_down.weight", "ffn_down.weight", F.silu(g8) * up)
    H = H + dn

fnw = get_aux("output_norm.weight")
hidden = rmsnorm(H[-1], fnw)
logits = gemm_ptq1("output.weight", "output.weight", hidden.unsqueeze(0))[0]
topv, topi = torch.topk(logits, 5)
print(f"\nBonsai 2 27B (Ternary PTQ1_0) completed in {time.time() - t0:.1f}s! Top-5 predictions:")
for v, i in zip(topv.tolist(), topi.tolist()):
    print(f"  tok={i:6d} logit={v:8.3f} str={tok.decode([i])!r}")
