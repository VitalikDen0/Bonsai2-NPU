# Optimization History: From Scalar C++ to 6.54 tok/s on Mobile NPU

This document traces the step-by-step hardware and algorithmic optimization journey of **Bonsai-NPU** — bringing a 27-billion parameter 1-bit / ternary hybrid language model (**Ternary-Bonsai-2-27B**, 48 Gated DeltaNet Linear Attention layers + 16 Full GQA layers) from a scalar C++ reference baseline to **341–349 ms/token (2.87–2.93 tok/s)** single-token decode and **153.0 ms/token (6.54 tok/s)** multi-token speculative decoding on a **OnePlus 13 (Snapdragon 8 Elite / SM8750, Hexagon v79 DSP, 16 GB LPDDR5X)**.

---

## Summary of Milestones

| Stage | Milestone | Single GEMV (`17408×5120`) | End-to-End Token Latency | Throughput | Speedup vs Stage 0 |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | **Scalar C++ Reference Baseline** | `52,078.00 ms` | `> 3,600,000 ms` (>1 hr) | `< 0.0003 tok/s` | `1.0x` |
| **1** | **Basic HVX 1024-bit SIMD (`32 floats/vec`)** | `180.87 ms` | `~114,000 ms` (114 s) | `0.009 tok/s` | `288x` |
| **2** | **DCVS Turbo Clocks + 4-Row Unroll + In-Reg Reduce** | `164.29 ms` | `~100,000 ms` (100 s) | `0.010 tok/s` | `317x` |
| **3** | **Direct Aligned `uint32_t` Bit-Mask Loads** | `69.44 ms` | `~45,000 ms` (45 s) | `0.022 tok/s` | `750x` |
| **4** | **Multi-Slice 4-Node QNN Graph (`4 HVX threads`)** | `23.00 ms` | `~14,500 ms` | `0.069 tok/s` | `2,264x` |
| **5** | **Native QuRT 4-Thread FastRPC Skeleton** | `8.06 ms` | `3,680 ms` | `0.27 tok/s` | `6,461x` |
| **6** | **QuRT 6-Thread Hardware Sweet Spot (`SM8750`)** | `5.86 ms` | `2,703 ms` | `0.37 tok/s` | `8,887x` |
| **7** | **Bonsai 2 Ternary Q2 + Radix-8/16 FWHT + Ping-Pong** | `4.46 ms` | `1,257 ms` | `0.80 tok/s` | `11,676x` |
| **8** | **8-Group `vmem` + Bit-Plane `vrmpyacc` (`Q8_0`)** | `1.18 ms` | `430 ms` | `2.33 tok/s` | `44,133x` |
| **9** | **Fused MLP (`193 RPCs`) + Hybrid Static SMMU** | `1.08 ms` | `411 ms` | `2.43 tok/s` | `48,220x` |
| **10** | **All-64-Layer DSP Fusion (`66 RPCs`) + Zero Host Math** | `1.00 ms` | **`341–349 ms`** | **`2.87–2.93 tok/s`** | **`52,078x`** |
| **11** | **MTP Batch Verify (`B=2..4`) + Rank-1 Rollback + Turbo4** | **`0.38 ms/tok`** | **`153.0–166.4 ms/tok`** | **`6.01–6.54 tok/s`** | **`> 135,000x`** |

---

## Detailed Chronology

### Stage 0 — Scalar C++ Reference Baseline (`52.08 s` per GEMV)
* **Architecture**: Naive scalar loop running on the Hexagon DSP scalar unit, unpacking 1-bit weights bit-by-bit and converting FP16 group scales in software.
* **Bottleneck**: Complete absence of vectorization (`1 float/cycle` instead of `32 floats/vector`), software FP16 conversion overhead, and byte-by-byte memory loads (`memub`).

### Stage 1–3 — HVX 1024-bit Vectorization & Word-Aligned Bit Unpacking (`52.08 s → 69.44 ms`)
* **1024-bit SIMD**: Replaced scalar loops with Hexagon Vector eXtensions (`HVX_Vector`, 128 bytes = 32 `float32` lanes per register), turning 1-bit weight multiplication into hardware predicate multiplexers (`Q6_V_vmux_QVV`) and `qf32` vector accumulators (`180.87 ms`).
* **In-Register Horizontal Reduction**: Replaced memory-backed horizontal sums with a 5-step in-register cyclic rotation butterfly (`Q6_V_vror_VR` by 64, 32, 16, 8, and 4 bytes) and locked DSP clocks to `DCVS_TURBO_PLUS` (`164.29 ms`).
* **Eliminating Byte Loads**: Replaced 11 million scalar byte loads (`memub`) with aligned 32-bit word loads and `Q6_V_vsplat_R`, cutting GEMV latency to `69.44 ms` (**750x speedup**).

### Stage 4–6 — Bypassing QNN Graph Overhead with Native 6-Thread QuRT (`69.44 ms → 5.86 ms`)
* **QNN Graph Bottleneck**: Running 4 parallel op nodes through Qualcomm QNN graph orchestration reduced GEMV to `23.00 ms`, but incurred massive per-node framework dispatch overhead.
* **Direct FastRPC + QuRT Thread Pool**: Bypassed QNN graph wrappers completely by writing a bare-metal CDSP FastRPC skeleton (`libbonsai_q1_skel.so`) managing a persistent pool of native QuRT worker threads synchronized via lightweight hardware semaphores (`qurt_sem_down` / `qurt_sem_up`).
* **6-Thread Hardware Sweet Spot**: Empirical testing on Snapdragon 8 Elite (Hexagon v79) revealed that 6 hardware HVX threads achieve peak throughput (`5.86 ms` GEMV, `2.70 s/token`), whereas 8 threads degrade performance (`8.25 ms`) due to QuRT hardware context-switching contention.

### Stage 7–8 — Bonsai 2 Ternary Q2, Bit-Plane `vrmpyacc`, and L2 Hardware Prefetch (`2,703 ms → 430 ms`)
* **Ternary Bit-Plane Layout (`Q2`)**: Repacked the 27B model into a single 6.74 GiB monolithic binary (`bonsai2-27b.npubin`) where each row stores contiguous positive (`+1`) and negative (`-1`) 1024-bit bit-planes alongside FP16 scales per 128-element group.
* **8-Bit Per-Group Activation Quantization (`Q8_0`) + `Q6_Vw_vrmpyacc_VwVbVb`**: Instead of expanding 1-bit weights to 32-bit floats, input activations are quantized once per call into 8-bit signed integers (`qx8`) with permuted bit-plane ordering. Eight 128-byte weight bit-planes (1,024 ternary weights) are loaded via aligned `vmem` and accumulated in 8 iterations using `Q6_Vw_vrmpyacc_VwVbVb` (128 int8 MACs per instruction across 4 output rows simultaneously), followed by a 6-instruction `Q6_W_vshuff_VVR` quad-row reduction and hardware L2 DMA prefetching (`Q6_l2fetch_AR`).
* **Compiler Bug Workaround (`HexagonQFloat` & `vextract`)**: Identified and bypassed an LLVM `hexagon-clang` `-O3` bug where `Q6_V_vzero()` initializers on `qf32` accumulators emitted a rogue `vadd.sf` instruction that turned negative sums into `NaN`, and replaced `Q6_R_vextract_VR` with inline assembly (`%0 = vextract(%1,%2)`) to achieve **zero `vmem(r29)` stack spills**.

### Stage 9–10 — Hybrid Static SMMU, All-64-Layer DSP Fusion (`66 RPCs`), and Zero Host Math (`430 ms → 341 ms`)
* **Hybrid Static SMMU + Ring Arena (`trans = 0.0 ms`, `memcpy = 0.0 ms`)**: Probed the 32-bit CDSP SMMU virtual address limit (`~3.68 GiB`) and pinned the first 30 transformer layers (`2.82 GiB`) plus the full `lm_head` (`322 MB`) permanently in CDSP SMMU memory (`FASTRPC_MAP_STATIC`). The remaining 34 layers stream asynchronously from clean Linux pagecache through a 194 MB ring buffer while the NPU computes layers 0..29, eliminating 100% of runtime `fastrpc_mmap` kernel lock contention (`trans = 0 ms`) and hiding `memcpy` latency completely.
* **Full Layer Fusion on Hexagon HVX (`257 RPCs → 66 RPCs`)**: Moved 100% of non-matrix operations — RMSNorm, Fast Walsh-Hadamard Transform (`FWHT-1024`, `FWHT-5120`, `FWHT-6144`, `FWHT-17408`), SwiGLU, Depthwise `Conv1D`, NeoX Partial RoPE, 48-layer Gated DeltaNet state recurrence ($S_h \in \mathbb{R}^{48 \times 128 \times 128}$), 16-layer Full GQA attention, and 2-call `lm_head` HVX lane argmax — directly into the 6-thread QuRT HVX pool (`hvx_lin_layer_fused`).
* **Result**: Host ARM CPU math dropped from `268 ms` to **`0.00 ms`**, FastRPC invocations dropped from `257` to **`66 calls/step`** (64 layers + 2 `lm_head` calls), and single-token latency reached **`341–349 ms/token` (`2.87–2.93 tok/s`)** — **81.1% of the physical LPDDR5X DRAM bandwidth limit (`279.8 ms`)**, even with stochastic sampling (`--temp 0.6 --top-p 0.9`) via 64 HVX lane winners returned directly by the 2-call static LM head.

### Stage 11 — Multi-Token Speculative Decoding (`B = 2..4`), Rank-1 Inverse Rollback & Turbo4 (`349 ms → 153.0 ms/tok = 6.54 tok/s`)
* **TurboQuant / Turbo4 (`16 KiB/token` 4-Bit KV Cache)**: Implemented HVX 4-bit symmetric quantization (`dsp_tq4_quantize_256`) and in-register nibble unpacking for all 16 Full GQA layers, cutting KV-cache memory by 4x (`16 KiB/token`) while accelerating long-context attention and prefill (`160.1–166.5 ms/tok = 6.01–6.25 tok/s`).
* **Register-Blocked Batched Verification (`B = 2..4` in 66 RPC calls)**: Integrated a zero-overhead multi-token prompt-lookup drafter (`1..3` draft tokens) with dual-token register blocking (`process_slice_q2_pair`) and 8-row L1/L2-resident tiling (`process_slice_batch_q2` with continuous `Q6_l2fetch_AR` on `b0 = 0`), streaming the 3.44 GB model weights from DRAM **only once per 4 verified tokens**.
* **Fused Multi-Token DeltaNet Recurrence**: Fused the batch loop `b = 0..batch-1` inside `run_lin_prep_worker` and `run_deltanet_worker` so that each layer's `6.29 MB` DeltaNet state matrix $S_h$ stays in HVX registers `vS0..vS3` across all batch tokens and is read/written to DRAM **only once per layer**, saving `350 MB` of DRAM traffic per extra token in the batch.
* **6-Thread HVX Rank-1 Inverse State Rollback (`TASK_DELTANET_UNDO`)**: Because backing up the full 48-layer DeltaNet state (`144 MiB`) exceeded the 256 MB CDSP heap and caused `166 ms` copy stalls, we derived the exact algebraic inverse of the rank-1 DeltaNet recurrence:
  $$\vec{S}_{\text{old}}[j] = \left(\vec{S}_{\text{new}}[j] - \vec{k} \cdot \delta_j\right) \cdot e^{-g}$$
  Storing only $\left(e^{-g}, \vec{k}, \vec{\delta}\right)$ requires just **10.16 MiB** across 3 draft slots, adds `0 ms` to the forward pass, and rolls back rejected draft tokens in `~3 ms` across 6 HVX threads.
* **Verified On-Device Result**: Accepts up to **4 tokens in a single 612 ms NPU step (`153.0 ms/tok = 6.54 tok/s`)**, achieving **6.01 tok/s (`166.4 ms/tok`, avg `3.33 tok/step`)** sustained end-to-end generation on OnePlus 13 with `--turbo4 --temp 0.6 --top-p 0.9`.
