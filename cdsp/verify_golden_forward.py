import struct
import time
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = r"D:\Download\Bonsai2_NPU\bonsai27b-1bit.npubin"
with open(path, "rb") as f:
    hdr = f.read(250000)
ver, n = struct.unpack_from("<II", hdr, 4)
p = 12
tmap = {}
for _ in range(n):
    nl = struct.unpack_from("<H", hdr, p)[0]; p += 2
    name = hdr[p:p+nl].decode(); p += nl
    kind, od, ind, off, blen = struct.unpack_from("<BIIQQ", hdr, p); p += 25
    tmap[name] = (kind, od, ind, off, blen)

mm = np.memmap(path, dtype=np.uint8, mode='r')
f32_cache = {}
def get_f32(name):
    if name not in f32_cache:
        kind, od, ind, off, blen = tmap[name]
        raw = mm[off:off+blen]
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32) if kind == 1 else np.frombuffer(raw, dtype=np.float32)
        f32_cache[name] = torch.from_numpy(arr.copy()).to(device)
    return f32_cache[name]

shifts = torch.arange(8, device=device, dtype=torch.uint8)

def gemm_q1(name, X):
    # X: [B, ind] -> returns [B, od]
    kind, od, ind, off, blen = tmap[name]
    ng = ind // 128
    prow = (ind // 8) + ((128 - ((ind // 8) % 128)) % 128)
    sc_raw = np.frombuffer(mm[off : off + od * ng * 2], dtype=np.float16)
    bp_raw = mm[off + od * ng * 2 : off + od * ng * 2 + od * prow].reshape(od, prow)[:, :ind // 8]
    sc = torch.from_numpy(sc_raw.copy()).to(device, dtype=torch.float32).reshape(od, ng) * 0.5
    bp = torch.from_numpy(bp_raw.copy()).to(device, dtype=torch.uint8)
    bits = ((bp.unsqueeze(-1) >> shifts) & 1).reshape(od, ng, 128)
    w = torch.where(bits == 1, sc.unsqueeze(-1), -sc.unsqueeze(-1)).reshape(od, ind)
    return torch.matmul(X, w.t())

def dequant_embeds(tok_ids):
    kind, od, ind, off, blen = tmap["language_model.model.embed_tokens"]
    ng = ind // 128
    prow = (ind // 8) + ((128 - ((ind // 8) % 128)) % 128)
    out = []
    for tid in tok_ids:
        sc_raw = np.frombuffer(mm[off + tid * ng * 2 : off + (tid + 1) * ng * 2], dtype=np.float16)
        bp_off = off + od * ng * 2 + tid * prow
        bp_raw = mm[bp_off : bp_off + ind // 8]
        sc = torch.from_numpy(sc_raw.copy()).to(device, dtype=torch.float32) * 0.5
        bp = torch.from_numpy(bp_raw.copy()).to(device, dtype=torch.uint8)
        bits = ((bp.unsqueeze(-1) >> shifts) & 1).reshape(ng, 128)
        w = torch.where(bits == 1, sc.unsqueeze(-1), -sc.unsqueeze(-1)).reshape(ind)
        out.append(w)
    return torch.stack(out, dim=0)

def rmsnorm(x, w, eps=1e-6):
    inv = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x * inv * w

def apply_rope(x, pos_ids):
    # x: [B, H, 256], first 64 dims rotated in NeoX pairs (0..31, 32..63)
    B, H, D = x.shape
    rot = 64
    idx = torch.arange(rot // 2, device=device, dtype=torch.float32)
    freq = 1.0 / (1e7 ** (2.0 * idx / rot))
    angles = pos_ids.unsqueeze(-1).float() * freq.unsqueeze(0) # [B, 32]
    c = torch.cos(angles).unsqueeze(1) # [B, 1, 32]
    s = torch.sin(angles).unsqueeze(1)
    x0 = x[:, :, :32]
    x1 = x[:, :, 32:64]
    out = x.clone()
    out[:, :, :32] = x0 * c - x1 * s
    out[:, :, 32:64] = x0 * s + x1 * c
    return out

tok = Tokenizer.from_file(r"D:\Download\Bonsai2_NPU\Bonsai-27B-mlx-1bit\tokenizer.json")
prompt = "The capital of France is"
ids = tok.encode(prompt).ids
print(f"Prompt: {prompt!r} -> ids={ids}")

# Let's test q_split_mode: 'interleaved' (qq.reshape(B, 24, 2, 256)) vs 'half' (qq[:, :6144], qq[:, 6144:])
for q_mode in ['interleaved', 'half']:
    t0 = time.time()
    H = dequant_embeds(ids) # [B, 5120]
    B = H.shape[0]
    pos_ids = torch.arange(B, device=device)

    for L in range(64):
        pfx = f"language_model.model.layers.{L}."
        lw = get_f32(pfx + "input_layernorm.weight")
        xn = rmsnorm(H, lw)
        if L % 4 != 3:
            qkv = gemm_q1(pfx + "linear_attn.in_proj_qkv", xn) # [B, 10240]
            z   = gemm_q1(pfx + "linear_attn.in_proj_z", xn)   # [B, 6144]
            av  = gemm_q1(pfx + "linear_attn.in_proj_a", xn)   # [B, 48]
            bv  = gemm_q1(pfx + "linear_attn.in_proj_b", xn)   # [B, 48]
            cw  = get_f32(pfx + "linear_attn.conv1d.weight").reshape(10240, 4)
            # causal conv1d of width 4 across sequence B:
            qkv_pad = F.pad(qkv.t().unsqueeze(0), (3, 0)) # [1, 10240, B+3]
            conv_out = F.conv1d(qkv_pad, cw.unsqueeze(1), groups=10240)[0].t() # [B, 10240]
            qkv_act = F.silu(conv_out)
            q = qkv_act[:, :2048].reshape(B, 16, 128)
            k = qkv_act[:, 2048:4096].reshape(B, 16, 128)
            v = qkv_act[:, 4096:].reshape(B, 48, 128)
            q = q * torch.rsqrt(torch.sum(q * q, dim=-1, keepdim=True) + 1e-6) * 0.08838834765
            k = k * torch.rsqrt(torch.sum(k * k, dim=-1, keepdim=True) + 1e-6)
            alog = get_f32(pfx + "linear_attn.A_log")
            dtb  = get_f32(pfx + "linear_attn.dt_bias")
            beta = torch.sigmoid(bv) # [B, 48]
            gv   = -torch.exp(alog) * F.softplus(av + dtb) # [B, 48]
            eg   = torch.exp(gv) # [B, 48]

            S = torch.zeros(48, 128, 128, device=device) # [h, j_v, i_k]
            outs = []
            for t in range(B):
                qt = q[t].repeat_interleave(3, dim=0) # [48, 128]
                kt = k[t].repeat_interleave(3, dim=0) # [48, 128]
                vt = v[t] # [48, 128]
                S = S * eg[t].view(48, 1, 1)
                kv = torch.einsum('hji,hi->hj', S, kt)
                delta = (vt - kv) * beta[t].unsqueeze(-1)
                S = S + torch.einsum('hj,hi->hji', delta, kt)
                ot = torch.einsum('hji,hi->hj', S, qt)
                outs.append(ot)
            out = torch.stack(outs, dim=0) # [B, 48, 128]
            nw = get_f32(pfx + "linear_attn.norm.weight")
            out_norm = rmsnorm(out, nw) * F.silu(z.reshape(B, 48, 128))
            yo = gemm_q1(pfx + "linear_attn.out_proj", out_norm.reshape(B, 6144))
            H = H + yo
        else:
            qq = gemm_q1(pfx + "self_attn.q_proj", xn) # [B, 12288]
            kk = gemm_q1(pfx + "self_attn.k_proj", xn) # [B, 1024]
            vv = gemm_q1(pfx + "self_attn.v_proj", xn) # [B, 1024]
            if q_mode == 'interleaved':
                qg = qq.reshape(B, 24, 2, 256)
                q = qg[:, :, 0, :]
                gate = qg[:, :, 1, :]
            else:
                q = qq[:, :6144].reshape(B, 24, 256)
                gate = qq[:, 6144:].reshape(B, 24, 256)
            qnw = get_f32(pfx + "self_attn.q_norm.weight")
            knw = get_f32(pfx + "self_attn.k_norm.weight")
            q = rmsnorm(q, qnw)
            k = rmsnorm(kk.reshape(B, 4, 256), knw)
            q = apply_rope(q, pos_ids)
            k = apply_rope(k, pos_ids)
            v = vv.reshape(B, 4, 256)
            k_rep = k.repeat_interleave(6, dim=1) # [B, 24, 256]
            v_rep = v.repeat_interleave(6, dim=1) # [B, 24, 256]
            scores = torch.einsum('th d,sh d->hts', q, k_rep) / 16.0 # [24, B, B]
            mask = torch.triu(torch.full((B, B), float('-inf'), device=device), diagonal=1)
            probs = torch.softmax(scores + mask.unsqueeze(0), dim=-1)
            ao = torch.einsum('hts,sh d->thd', probs, v_rep) # [B, 24, 256]
            flat = (ao * torch.sigmoid(gate)).reshape(B, 6144)
            yo = gemm_q1(pfx + "self_attn.o_proj", flat)
            H = H + yo

        pw = get_f32(pfx + "post_attention_layernorm.weight")
        n2 = rmsnorm(H, pw)
        g8 = gemm_q1(pfx + "mlp.gate_proj", n2)
        up = gemm_q1(pfx + "mlp.up_proj", n2)
        dn = gemm_q1(pfx + "mlp.down_proj", F.silu(g8) * up)
        H = H + dn
        if q_mode == 'interleaved' and L in (0, 3, 63):
            print(f"L={L} H[0,:4]={H[0,:4].tolist()} H[-1,:4]={H[-1,:4].tolist()}")

    fnw = get_f32("language_model.model.norm.weight")
    hidden = rmsnorm(H[-1], fnw)
    if q_mode == 'interleaved':
        print(f"final hidden[:4]={hidden[:4].tolist()}")
    logits = gemm_q1("language_model.lm_head", hidden.unsqueeze(0))[0]
    topv, topi = torch.topk(logits, 5)
    print(f"\n=== q_mode={q_mode} ({time.time() - t0:.1f}s) ===")
    for v, idx in zip(topv.tolist(), topi.tolist()):
        print(f"  tok={idx:6d} logit={v:8.3f} str={tok.decode([idx])!r}")
    break
