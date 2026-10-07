# Technical Architecture & Systems Specification: Bonsai-NPU

## Executive Summary

**Bonsai-NPU** is a hardware-optimized inference runtime specifically designed to run the 27-billion parameter **Ternary-Bonsai-2-27B** hybrid language model on Qualcomm Snapdragon mobile silicon. By combining ultra-low-bit quantization (1-bit / ternary representations), monolithic SMMU memory mapping, fused Hexagon Vector Extensions (HVX), batched Multi-Token Prediction (MTP), and Turbo4 4-bit KV caching, the engine breaks the memory-bandwidth wall and enables real-time generation on edge devices.

> **Testing Environment Note:**  
> Reference hardware benchmarks were conducted on a **OnePlus 13 (Snapdragon 8 Elite / SM8750, 16 GB LPDDR5X)** with **Root access enabled**. Root privileges were utilized strictly for low-level engineering benchmarking: manual clock frequency locking via Linux kernel `/sys/` nodes to eliminate thermal throttling variance and direct ADB deployment. For production and end-user deployments, root privileges are **not required**.

---

## 1. Architectural Foundations of Bonsai 2 27B

Bonsai 2 27B employs a state-of-the-art **hybrid transformer architecture** consisting of 64 total layers designed to replace traditional quadratic self-attention with linear recurrent state-spaces wherever possible:

* **Total Layers**: 64 layers.
* **Linear Attention (Delta-Net)**: 32 layers. Uses a recurrent associative state:
  `S_t = S_{t-1} + k_t · v_t^T`  
  This eliminates `O(N)` KV-cache growth and maintains constant `O(1)` computational complexity per generation step.
* **Full Attention (GQA)**: 16 layers. Grouped-Query Attention with 24 query heads and 4 key-value heads, head dimension `d_k = 256`, with NeoX Rotary Positional Embeddings (RoPE).
* **Fused MLP Blocks**: SwiGLU activation (`Swish(x · W_gate) ⊙ (x · W_up)`) combined with Fast Walsh-Hadamard Transforms (FWHT-1024) to eliminate activation outliers prior to low-bit projection.
* **Model Dimensions**: Hidden dimension `d_model = 5120`, intermediate MLP dimension `d_mlp = 17408`, vocabulary size `V = 248320`.

---

## 2. Silicon Constraints & The Physical Memory Wall

### 2.1 Theoretical DRAM Bandwidth Floor
The Snapdragon 8 Elite (SM8750) features a 4-channel LPDDR5X-5300 memory subsystem providing a peak theoretical bandwidth of:
```
BW_peak = 106.7 GB/s
```

Under typical mobile operating conditions, maximum sustainable memory controller bus efficiency is approximately `η ≈ 87%`, yielding an achievable continuous read bandwidth of:
```
BW_eff ≈ 92.8 GB/s
```

With Bonsai 2 27B quantized to 1-bit / ternary weights, the active model weight footprint is:
```
M_weights = 3.405 GB (3.405 × 10^9 bytes)
```

For single-token autoregressive decoding (`B = 1`), every weight parameter must be streamed from DRAM into the NPU compute cores exactly once per step:
```
T_floor = (3.405 × 10^9) / (92.8 × 10^9) ≈ 279.8 ms
```
This represents the absolute theoretical hardware ceiling: **3.57 tok/s**.

### 2.2 Why ≥ 95.8% Utilization is Blocked in Userspace Android
Initial optimization goals targeted `≥ 95.8%` utilization (`≤ 292 ms`). Rigorous empirical probing on hardware revealed why this boundary cannot be crossed in userspace Android:

1. **Kernel Driver Mutex Lockout (`fl->map_mutex`)**:
   Qualcomm FastRPC driver (`drivers/soc/qcom/fastrpc.c`) guards all `fastrpc_mmap` calls with a per-session kernel mutex `fl->map_mutex`. When a background CPU thread attempts to asynchronously map or unmap memory while the NPU thread executes GEMV, the kernel blocks the NPU ioctl, causing single-layer latency to spike from `1.5 ms` to `10.92 ms`.
2. **LPDDR5X Read/Write Bus Contention**:
   The shared memory controller must arbitrate between the NPU streaming reads (`90 GB/s`) and CPU cache writebacks/DMA transfers. High-frequency write bursts penalize LPDDR5X read turnaround cycles (`tWTR` / `tRTW`), introducing a non-negotiable `60 ms` overhead across 64 layers.

As a result, **345 ms (81.1% hardware ceiling)** represents the true physical limit for single-token decode (`B = 1`) in Linux/Android userspace.

---

## 3. Monolithic SMMU Architecture (DMA-BUF Zero-Copy)

Rather than segmenting the model into small sub-graphs or dynamically paging weights in and out of SMMU context:
* The entire 64-layer graph (3.405 GB) is mapped into a single contiguous DMA-BUF SMMU address range at startup (`stage_init`).
* The memory layout consists of:
  * **Group 0 (Layers 0–31)**: 1446 MB
  * **Group 1 (Layers 32–63)**: 1443 MB
  * **Static LM Head**: 322 MB
  * **Ring & Scratch Arenas**: 194 MB
  * **Total SMMU Footprint**: **3405 MB**, well below the Qualcomm SMMU 3680 MB limit.
* During autoregressive decoding, zero `mprotect`, `mmap`, or `munmap` syscalls occur. The NPU accesses weights directly via hardware physical pointers without kernel involvement.

---

## 4. Hexagon DSP HVX Co-Processor Pipeline

The Qualcomm Hexagon v79 DSP serves as a specialized vector co-processor handling all non-GEMV operations across all 64 layers.

```
       Host (ARM64)                    Hexagon CDSP (HVX v79)
┌─────────────────────────┐         ┌──────────────────────────────┐
│  Tokenizer & Sampling   │         │ 4x QuRT Worker Threads Pool  │
│  Orchestration Loop     │◄───────►│ 1024-bit Vector Intrinsics   │
│  FastRPC RPC Dispatch   │         │ (SwiGLU, RMSNorm, RoPE, FWHT)│
└─────────────────────────┘         └──────────────────────────────┘
            │                                      │
            ▼                                      ▼
┌──────────────────────────────────────────────────────────────────┐
│                   Qualcomm NPU Compute Cores                     │
│               Monolithic 1-Bit / Ternary GEMV                    │
└──────────────────────────────────────────────────────────────────┘
```

### 4.1 Vector Register Allocation
Hexagon HVX provides 32 vector registers of 1024 bits (128 bytes) each (`v0` through `v31`). All activation operations are written in native Hexagon C intrinsics (`Q6_V_...`):
* **RMSNorm**: Evaluated via 8-way vector dot-product reduction (`Q6_Vqf32_vadd_Vqf32Vqf32`), computing the reciprocal square root via Newton-Raphson approximation.
* **NeoX RoPE**: Performed directly on vector registers for the first 64 head dimensions using complex rotation intrinsics.
* **FWHT-1024**: Fused Fast Walsh-Hadamard Transform executed in L1 cache on 1024-float blocks before MLP projection, completely eliminating outlier quantization errors.
* **Parallel GQA Attention**: 4 worker threads managed by QuRT (`qurt_sem_down` / `qurt_sem_up`) parallelize the 24 query heads and 4 key-value heads across DSP execution slices.

---

## 5. Multi-Token Prediction (MTP) Speculative Decoding

Because single-token autoregressive decoding is strictly memory-bandwidth bound, the only path to exceeding `3.5 tok/s` is **amortizing the 3.4 GB memory read across multiple tokens**.

### 5.1 Hardware Batched Verification
Bonsai-NPU implements hardware-level batching (`forward_tokens_batch` and `cdsp_lmhead_batch`):
* For batch size `B = 4`, the NPU reads the 3.4 GB model weights from DRAM **once**.
* Computes GEMV for 4 tokens concurrently within the NPU execution units.
* Hardware benchmark on OnePlus 13:
  ```
  T_verify(B = 4) = 380 ms  ==>  95 ms/token  (10.52 tok/s)
  ```

### 5.2 Speculative Draft Loop
```
   Step t:  Context Tokens
                │
                ▼
   ┌──────────────────────────┐
   │ CPU Prompt Lookup Draft  │ ──► Proposes 3 candidate tokens (dt < 2 ms)
   └──────────────────────────┘
                │
                ▼
   ┌──────────────────────────┐
   │ NPU Batch-4 Verification │ ──► Computes logits for all 4 tokens (380 ms)
   └──────────────────────────┘
                │
                ▼
   ┌──────────────────────────┐
   │ Greedy Match Validation  │ ──► Accepts k tokens (k in [1, 4])
   └──────────────────────────┘
```

* **Candidate Draft**: A CPU-based Prompt Lookup / n-gram proposer extracts pattern continuations from recent context in `< 2 ms` (0 additional parameters).
* **Verification**: The batch of 4 candidates is dispatched to `forward_tokens_batch`.
* **Throughput**:
  * If 3 candidates are accepted: 4 tokens emitted in `380 ms` (effective `10.5 tok/s`).
  * Under typical text entropy (70–75% acceptance rate): Effective throughput is **5.5–8.0 tok/s**.

---

## 6. Turbo4 Dynamic Long-Context KV-Cache

### 6.1 Demand-Paged Overcommit
Traditional inference engines allocate contiguous physical RAM upfront for the entire maximum context window, triggering OOM crashes on mobile. Bonsai-NPU solves this via **demand-paged overcommit**:
```c
kvk = (int8_t*)mmap(NULL, kv_sz, PROT_READ | PROT_WRITE,
                    MAP_ANONYMOUS | MAP_PRIVATE | MAP_NORESERVE, -1, 0);
```
* Virtual address space is reserved for up to **262,144 tokens**.
* Physical RAM is committed on-demand by the Linux kernel page-fault handler only as tokens are actually written.
* Context memory consumption is strictly proportional to active conversation length.

### 6.2 Quantization & Memory Footprint Comparison

| Context Length | Format: FP16 (Standard) | Format: Q8_0 / BF16 (Current) | Format: Turbo4 (4-Bit KV) |
| :--- | :---: | :---: | :---: |
| **Bytes / Token** | 128 KiB | 64 KiB | **16 KiB** |
| **4,096 tokens** | 512 MB | 256 MB | **64 MB** |
| **65,536 tokens** | 8.0 GB (OOM) | 4.2 GB (Unstable) | **1.05 GB** (Fits easily) |
| **262,144 tokens** | 32.0 GB (Fatal) | 16.8 GB (Fatal) | **4.1 GB** (Fits in RAM) |

Turbo4 packs each 256-dimensional head into 128 bytes (two 4-bit nibbles per byte) with per-group scales, enabling true 65k–262k long-context execution without exhausting phone memory.

---

## 7. Security Model, Directory Permissions & Non-Root Execution

### 7.1 Accessing `/data/local/tmp/`
Can users access and run binaries in `/data/local/tmp/` without root?

* **Via ADB Shell (from PC)**: **YES, FULL ACCESS**.  
  In Android, `/data/local/tmp` has filesystem permissions `0777` / `1777` (`rwxrwxrwt`). Commands executed over ADB run as the `shell` user (`UID 2000`), which has full read, write, and execute permissions inside `/data/local/tmp` on every retail, non-rooted Android device. Any user can run `adb push` and execute `./bonsai_fwd` without root.
* **Locally on the Phone (without PC, via Termux or Android Apps)**:  
  Since Android 10, SELinux policy enforces W^X execution restrictions preventing untrusted third-party app processes (`untrusted_app`) from executing binaries located in `/data/local/tmp`.  
  In this standalone scenario, execution takes place within the **application's private sandbox**:
  * In **Termux**: Run binaries directly in the user's home directory:
    ```bash
    cd /data/data/com.termux/files/home
    ./bonsai_fwd bonsai27b-1bit.npubin tok.bin --server 8080
    ```
  * In **Android App (APK)**: Place `libbonsai_q1_skel.so` inside `lib/arm64-v8a` and load it natively using `System.loadLibrary()`.

### 7.2 FastRPC Architecture & Performance Voting Without Root
1. **Userspace Accessibility**:  
   The FastRPC device node (`/dev/fastrpc-cdsp`) and vendor dynamic libraries (`/vendor/lib64/libcdsprpc.so`) are exposed to all Android userspace processes by default. Non-root apps have native access to Hexagon DSP and NPU acceleration.
2. **Performance Scaling Without Root**:  
   Production deployments utilize the official FastRPC QoS and Power voting API:
   ```c
   HAP_power_request_t req;
   req.type = HAP_power_req_perf;
   HAP_power_set(NULL, &req);
   ```
   This directs the Qualcomm Hexagon Power Manager to scale both the DSP and memory bus to peak burst frequencies entirely through userspace requests without touching `sysfs`.

---

## 8. Forward Compatibility with Newer Chip Generations

Can users deploy this codebase on newer or future Snapdragon chips?

**Yes, with complete forward compatibility and higher performance.**

1. **Hexagon ISA Forward & Backward Compatibility**:  
   Qualcomm preserves compatibility for the Hexagon Vector eXtensions (HVX) instruction set. Binaries compiled for **Hexagon v79** (Snapdragon 8 Elite) will execute seamlessly on future microarchitectures (**Hexagon v81, v83 and newer** in Snapdragon 8 Elite Gen 2, Snapdragon 8 Gen 5, etc.).
2. **Stable FastRPC & QNN HAL Interfaces**:  
   Qualcomm's FastRPC IPC mechanism (`libcdsprpc.so`) and QNN hardware abstraction layers remain standardized across Android versions and hardware iterations.
3. **Automatic Scaling with Memory Bandwidth**:  
   Because the monolithic 1-bit GEMV engine is bounded by DRAM bandwidth, future Snapdragon platforms featuring **LPDDR5X-9600** or **LPDDR6** (>130–150 GB/s) will immediately yield higher generation speeds without code changes:
   * **106.7 GB/s (Snapdragon 8 Elite)**: `345 ms/token` (`2.90 tok/s`).
   * **~135 GB/s (Future SoCs)**: Projected `~260 ms/token` (`~3.85 tok/s` at B=1, and `~10–12 tok/s` with MTP).
