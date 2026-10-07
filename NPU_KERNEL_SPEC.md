# NPU kernel spec — 1-bit Bonsai 27B on Hexagon v79 (SM8750, OnePlus 13)

Source of truth: `repack_npu.py` (container NPU1) + whitepaper §4.2/§5/App.A.
Math: `w[i,j] = s_g * b`, `b ∈ {−1,+1}`, one FP16 `s_g` per 128 weights.
Activations FP16, accumulation FP32. No Hadamard rotation (unlike Bonsai 2).

## 1. Decode path (batch=1, memory-bound) — HVX GEMV

Per output row `i`, per group of 128:
1. VMEM load 16 B bitplane (one 1024-bit vector holds 8 groups) + 1×FP16 scale.
2. Expand bits → `±s_g` FP16 via nibble LUT (`vlut4`/`vshuff`, 128 bits per cycle class).
3. `vfmpy` with broadcast/streamed `x` chunk (FP16), `vfadd` into FP32 accumulator.
4. Row result claimed once; `x` (5120×2 B = 10 KB) stays in L2/VTCM for the whole layer.

Traffic per token ≈ 3.4 GB weights + scales. Ceiling = LPDDR5X BW / 3.9 GB.
Roofline target: ~10–15 tok/s sustained (iPhone reference 11 tok/s).

## 2. Prefill path (chunk 64–128, compute-bound) — HMX + HVX cooperatively

- HVX unpacks bitplane tiles → FP16 tiles into VTCM (double buffer A/B).
- HMX consumes FP16 tiles: activation tile `M×K` (M = chunk, K streamed), weight-stationary.
- While HMX chews buffer A, DMA fills buffer B. Zero-Copy: weights imported once
  via ION/AHardwareBuffer, never `memcpy`d in the hot loop.
- M < 32 tail rows fall back to HVX GEMV (same kernel as §1).

## 3. Non-GEMM ops (all on CDSP, never ARM)

- RMSNorm / RoPE (partial rotary 0.25, theta 1e7, mrope [11,11,10]) / SiLU×mul (SwiGLU)
  / softmax (only 16 full-attn layers, 4 KV heads × dim 256): scalar + HVX threads.
- Linear-attention SSM recurrence: FP32 state, 48 layers × 16×128×128 ≈ 50 MB,
  pinned resident, updated in place. Conv kernel 4, groups 16.
- Full attention: Q8 KV cache first (32 KB/tok → 16K=512 MB, 32K=1 GB);
  65K mode switches to Q4_0 + mean-centering bias (calibrate later via fork tool
  logic, ported — `llama-kv-mean-center` equivalent, one subtract on write).
- Sampling: temp 0.7, top_p 0.95, top_k 20 (whitepaper thinking-mode).

## 4. Memory map (16 GB LPDDR, target)

- Resident pinned: weights 3.9 GB (npubin) + scales in-container + SSM 50 MB +
  norm/conv/A_log/dt_bias ~100 MB + KV pool (mode-sized: 0.5 / 1.0 / 2.1 GB) +
  runtime ~1.3 GB. 65K/Q8 worst ≈ 9.3 GB — fits with `manage_memory.sh enable`.
- VTCM 8 MB: double-buffered weight tile + activation tile + dequant LUT only.
- FastRPC: blocking serial calls forbidden in hot loop — batch one `invoke` per
  layer (fused dequant+GEMM+norm where possible), async DMA prefetch for next layer.

## 5. Correctness gates (no silent CPU fallback — ever)

- Every `invoke` asserts HTP residency (perf counters); any ARM fallback = fatal log.
- Bit-exactness gate on PC: NPU GEMV vs `repack_npu` dequant on 1000 random
  rows, tolerance FP16 eps. KV gate: forward-KL vs FP16-KV baseline (paper: 0.0009).
- DSpark drafter (6-layer, k=4, τ≈3.6) is STAGE 2 — decode ships without it first.

## 6. Modes (single user)

- 16K: Q8 KV, DSpark later ON. 32K: Q8 KV. 65K: Q4 KV + bias.
- Switching = preallocated pool select at engine start, no recompile.
- Vision tower (HQQ 4-bit, 0.63 GB): demand-paged, text path never pays.
