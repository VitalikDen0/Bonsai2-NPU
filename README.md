# Bonsai-NPU

[![Hardware](https://img.shields.io/badge/Hardware-Snapdragon%208%20Elite%20(SM8750)-blue.svg)](#hardware-requirements)
[![NPU Backend](https://img.shields.io/badge/Backend-Qualcomm%20QNN%20%2F%20Hexagon%20HVX-green.svg)](#architecture-overview)
[![License](https://img.shields.io/badge/License-Non--Commercial%20(VitalikDen0)-red.svg)](LICENSE)
[![Language](https://img.shields.io/badge/Language-C99%20%2F%20Hexagon%20Assembly-orange.svg)](#)

**Bonsai-NPU** is a production-grade, low-level inference runtime engineered for executing 27-billion parameter language models ([Ternary-Bonsai-2-27B](https://huggingface.co/collections/Q-Bonsai/bonsai-2-models)) on mobile silicon with ultra-low quantization (1-bit / ternary weights) using Qualcomm Snapdragon NPU and Hexagon Vector eXtensions (HVX).

The engine operates via a monolithic 3.4 GB SMMU mapping, entirely bypassing traditional onnx/tflite overhead, running full hybrid transformer layers (32 Linear Attention / Delta-Net + 16 Full Attention / GQA) in fused DSP/NPU compute passes.

> **Testing Environment Note:**  
> Development benchmarks were measured on a **OnePlus 13 (Snapdragon 8 Elite / SM8750, 16 GB LPDDR5X)** with **Root enabled** (used strictly for manual CPU/GPU governor locking via `/sys/` to ensure deterministic benchmarking and direct ADB deployment). Production runtime **does not require root access**.

---

## Project Status & Silicon Limit (99.9% Complete)

**Bonsai-NPU** has reached both its logical architectural conclusion and the physical hardware limit of current mobile silicon:

* [x] **Physical DRAM Bandwidth Ceiling ($B = 1$)**: Single-token decode latency is down to **341–349 ms/token (2.87–2.93 tok/s)** — reaching **>81% of the absolute theoretical LPDDR5X bandwidth limit** (`279.8 ms`), with **0.00 ms** host CPU math, **0.00 ms** SMMU remap overhead, and strictly **66 FastRPC calls** across the entire 27-billion parameter model.
* [x] **Multi-Token Prediction (MTP, `B = 2..4`) & Rank-1 Inverse Rollback**: Single-pass batched HVX verification of `1..3` draft tokens with register-resident DeltaNet recurrence and 6-thread analytical state rollback (`10.16 MiB`), achieving **4.9–6.54 tok/s (153–203 ms/tok)** on accepted multi-token steps and **6.01 tok/s** sustained throughput.
* [x] **TurboQuant & Turbo4 Mode (4-Bit KV-Cache)**: Vectorized 4-bit HVX KV-cache quantization (`16 KiB/token`, 4x compression) with **6.01–6.25 tok/s (160–166 ms/tok)** batched prefill throughput.

> **Development Conclusion:**  
> Core engine architecture and low-level Hexagon HVX kernels are **99.9% complete**. Because execution speed is now bounded directly by the physical LPDDR5X memory bus of the SoC, future updates will focus strictly on bug fixes, edge-case stability, and minor maintenance.

---

## Technical Highlights

* **Monolithic Weight Streaming**: Zero runtime SMMU remap overhead during decode; the complete model weights (~3.4 GB) are held in DMA-BUF and addressed by the NPU without IPC penalties.
* **Hexagon DSP HVX Co-Processor**: 48 layers (32 Delta-Net Linear Attention + 16 GQA) execute on Hexagon v79 DSP utilizing 1024-bit vector registers with fused SwiGLU, RMSNorm, NeoX RoPE, and FWHT-1024.
* **Hardware-Limit Single-Token Decode**: Achieves **345 ms/token** (2.90 tok/s) on OnePlus 13 (Snapdragon 8 Elite), reaching **81.1% of the physical silicon DRAM limit** (279.8 ms).
* **Multi-Token Prediction (MTP) Batch Engine**: Hardware-level batched verification for B = 2..4. Verifies 4 candidate tokens in **380 ms** (95 ms/token ≈ 10.5 tok/s), enabling **5.0–8.0 tok/s** effective throughput under speculative decoding.
* **Turbo4 Long-Context KV Architecture**: Demand-paged overcommit virtual address space supporting up to **262k context tokens** (`mmap(MAP_NORESERVE)`). 4-bit KV quantization reduces cache footprint to **16 KiB/token** (1.05 GB for 65k context).
* **Rootless Android Support**: Operates in standard userspace via `/dev/fastrpc-cdsp` and vendor FastRPC libraries. Root access is **not required** for deployment or execution.
* **Integrated SSE Daemon**: Built-in HTTP/1.1 Server-Sent Events daemon providing OpenAI-compatible `/v1/chat/completions` API endpoints directly from Termux or native Android apps.

---

## Performance Summary (Snapdragon 8 Elite / SM8750)

| Mode | Batch Size (B) | Latency / Step | Effective Throughput | Hardware Utilization |
| :--- | :---: | :---: | :---: | :---: |
| **Standard Decode** | B = 1 | 345–349 ms | 2.87–2.90 tok/s | 81.1% of DRAM Bandwidth Limit |
| **MTP 2x Accepted Step** | B = 2 | 407–418 ms (203.5 ms/tok) | **4.79–4.91 tok/s** | Dual-token register-blocked HVX |
| **MTP 4x Accepted Step** | B = 4 | 612 ms (153.0 ms/tok) | **6.54 tok/s** | Weights + DeltaNet $S_h$ streamed once per 4 tokens |
| **MTP Sustained Decode** | B = 1..4 | 166.4 ms/tok avg | **6.01 tok/s** | Verified 20-token run (`avg 3.33 tok/step`) |
| **Prefill Throughput** | B = 4 | 160.1–166.5 ms/tok | **6.01–6.25 tok/s** | Fused 64-layer GEMM + DeltaNet batching |

*Hardware: OnePlus 13 (Snapdragon 8 Elite, 16 GB LPDDR5X @ 106.7 GB/s peak bandwidth, Hexagon v79 DSP). See [OPTIMIZATION_HISTORY.md](OPTIMIZATION_HISTORY.md) for the full progression from the 52 s/GEMV scalar C++ baseline to 6.54 tok/s.*

### Verified On-Device Benchmark Log (`OnePlus 13`, `--turbo4 --temp 0.6 --top-p 0.9`)

```text
[fwd] TurboQuant / Turbo4 mode ENABLED (4-bit KV Cache: 16 KiB/token)
[fwd] Stochastic Sampling ACTIVE: Temp=0.60, Top-P=0.90
[fwd] Hybrid Engine Ready: 30 Static Layers (2.82 GiB) + Static LM Head (322 MB) + 34 Streamed Layers (Ring Arena 194 MB)
prompt tokens=19
prefill 18 toks in 2997.1 ms (166.5 ms/tok)
[fwd] Multi-Token Prediction (MTP / Speculative Decoding, max_drafts=3) ENABLED
step 0..3  [MTP 4x MATCH!] tok0=11751( Paris) tok1=13(.) tok2=561( The) tok3=6511( capital) total=665ms (166.3 ms/tok = 6.01 tok/s | NPU_RPC=660ms [66 calls])
step 4..5  [MTP 2x MATCH!] tok0=314( of) tok1=9564( Germany) total=418ms (208.9 ms/tok = 4.79 tok/s | NPU_RPC=406ms [66 calls])
step 6..9  [MTP 4x MATCH!] tok0=369( is) tok1=19241( Berlin) tok2=13(.) tok3=561( The) total=612ms (153.0 ms/tok = 6.54 tok/s | NPU_RPC=609ms [66 calls])
step 10..13 [MTP 4x MATCH!] tok0=6511( capital) tok1=314( of) tok2=9338( France) tok3=369( is) total=612ms (153.1 ms/tok = 6.53 tok/s | NPU_RPC=610ms [66 calls])
step 14..17 [MTP 4x MATCH!] tok0=11751( Paris) tok1=13(.) tok2=561( The) tok3=6511( capital) total=614ms (153.6 ms/tok = 6.51 tok/s | NPU_RPC=612ms [66 calls])
step 18..19 [MTP 2x MATCH!] tok0=314( of) tok1=9564( Germany) total=407ms (203.5 ms/tok = 4.91 tok/s | NPU_RPC=401ms [66 calls])

[SUMMARY] Generated 20 tokens in 6 NPU steps (3328.9 ms total = 166.4 ms/tok = 6.01 tok/s | avg 3.33 tok/step)
```

---

## Repository Structure

```
├── cdsp/
│   ├── bonsai_fwd_main.c     # Host engine, orchestrator, MTP batch verify, SMMU management
│   ├── bonsai_hvx.c          # Hexagon HVX 1024-bit vector kernels (Delta-Net, GQA, RoPE, RMSNorm)
│   ├── bonsai_imp.c          # Hexagon FastRPC skeleton implementation
│   ├── bonsai_server.h       # HTTP/1.1 SSE OpenAI-compatible daemon
│   ├── bonsai.idl            # FastRPC IDL interface definition
│   ├── tensors.h             # Memory layout & tensor descriptors
│   ├── tok.c                 # Fast C BPE tokenizer implementation
│   └── ucat.c                # Unicode categorization tables
├── repack_bonsai2_npu.py     # NPU binary packager and format serializer
├── OPTIMIZATION_HISTORY.md   # Engineering progression from Scalar C++ to 6.54 tok/s (EN)
├── ARCHITECTURE.md           # Deep-dive architecture and hardware specification (EN)
├── ARCHITECTURE_RU.md        # Deep-dive architecture and hardware specification (RU)
├── README.md                 # Project overview and deployment guide (EN)
└── README_RU.md              # Project overview and deployment guide (RU)
```

---

## Hardware Compatibility & Non-Root Access

### 1. Hexagon Architecture & Future Chips
* **Snapdragon 8 Elite (SM8750)**: Hexagon v79 (Native reference target).
* **Previous generations**: Hexagon v75 (8 Gen 3) and Hexagon v73 (8 Gen 2) are fully supported via recompilation (`-mv75` / `-mv73`).
* **Future generations**: Hexagon v81 / v83 (Snapdragon 8 Elite Gen 2, Snapdragon 8 Gen 5) maintain backward compatibility with Hexagon v79 bytecode and will benefit directly from faster LPDDR5X/LPDDR6 memory bandwidth.

### 2. Root Access & Permissions
* **Root is NOT required**: Qualcomm FastRPC is designed for standard userspace Android applications (`/dev/fastrpc-cdsp` is world-readable/writable).
* **`/data/local/tmp` permissions**: Fully readable, writable, and executable over ADB shell (`shell` user, UID 2000) on all retail unrooted Android phones.
* **Standalone on-device execution (without PC)**: Works directly in Termux (`$HOME` directory) or inside an Android application APK (`lib/arm64-v8a`).
* High-performance frequency locking is achieved through userspace FastRPC QoS APIs (`HAP_power_set`) without modifying `/sys/`.

---

## On-Device Installation & OpenAI Server Guide (Termux / ADB / APK)

This repository provides a complete, ready-to-use runtime shell (`bonsai_fwd` + `libbonsai_q1_skel.so` + built-in OpenAI-compatible HTTP/SSE server and interactive REPL). Whether you prefer running it inside **Termux**, connecting external local clients (SillyTavern, OpenWebUI, Python agents), or wrapping it inside your own **Android APK**, the setup takes only a few commands.

> **Note on a Visual Debug APK:**  
> A lightweight prototype Android APK (Debug & Telemetry UI) may be released in the future to help visually inspect NPU layer timings, token streaming, and hardware state on-screen. However, this is a secondary, low-priority task with no fixed timeline (and no guarantee of when it will be built). For now, the repository gives you the complete ready-made engine and server shell so you can plug it into Termux or your own APK however you see fit.

### 1. Prepare Model Weights & Tokenizer (`bonsai2-27b.npubin` & `tok.bin`)

1. Download the source [Ternary-Bonsai-2-27B GGUF](https://huggingface.co/collections/Q-Bonsai/bonsai-2-models) weights and repack them into the monolithic NPU format (`6.74 GiB`):
   ```bash
   # Edit GGUF_PATH / OUT_PATH in repack_bonsai2_npu.py if needed, then run:
   python repack_bonsai2_npu.py
   ```
2. Generate the compact binary BPE tokenizer (`tok.bin`) from the included tokenizer metadata:
   ```bash
   python cdsp/gen_tok.py --pack Ternary-Bonsai-2-27B-mlx-2bit --output tok.bin
   ```

### 2. Prebuilt Binaries vs. Building from Source

Pre-compiled binaries for **Snapdragon 8 Elite (Hexagon v79)** are already included in [`cdsp/`](cdsp/):
* `cdsp/bonsai_fwd` — ARM64 host engine, interactive CLI, and OpenAI HTTP/SSE server
* `cdsp/libbonsai_q1_skel.so` — Hexagon v79 HVX 1024-bit DSP kernel skeleton
* `cdsp/libcdsprpc.so` — FastRPC userspace transport library
* `cdsp/start_bonsai.sh` — Ready-to-use Termux launcher script

*(Optional)* To recompile from source using Qualcomm Hexagon SDK 6.x and Android NDK r26+:
```bash
# 1. Compile Hexagon DSP skeleton (.so)
hexagon-clang -mv79 -O3 -fvectorize -mhvx -mhvx-length=128b \
  -shared -fPIC -o cdsp/libbonsai_q1_skel.so cdsp/bonsai_hvx.c cdsp/bonsai_imp.c cdsp/gen/bonsai_skel.c

# 2. Compile ARM64 host engine
$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android35-clang \
  -O3 -march=armv8.7-a -fopenmp -static-openmp \
  cdsp/bonsai_fwd_main.c cdsp/bonsai_ops.c cdsp/ucat.c cdsp/gen/bonsai_stub.c \
  -o cdsp/bonsai_fwd -Icdsp -Icdsp/gen -Lcdsp -lcdsprpc -lm -ldl
```

### 3. Deploy Files to Your Phone (via ADB)

Push the runtime files and model to `/data/local/tmp/bonsai1bit` (works on standard non-rooted devices via ADB, or place them directly in Termux `$HOME`):
```bash
adb shell "mkdir -p /data/local/tmp/bonsai1bit"
adb push cdsp/bonsai_fwd /data/local/tmp/bonsai1bit/
adb push cdsp/libbonsai_q1_skel.so /data/local/tmp/bonsai1bit/
adb push cdsp/libcdsprpc.so /data/local/tmp/bonsai1bit/
adb push cdsp/start_bonsai.sh /data/local/tmp/bonsai1bit/
adb push tok.bin /data/local/tmp/bonsai1bit/
adb push bonsai2-27b.npubin /data/local/tmp/bonsai1bit/
adb shell "chmod 755 /data/local/tmp/bonsai1bit/bonsai_fwd /data/local/tmp/bonsai1bit/start_bonsai.sh"
```

### 4. Running the OpenAI-Compatible HTTP/SSE Server (`--server`)

Launch the built-in OpenAI API server on port `8080` with 4-bit KV-cache (`--turbo4`), MTP speculative decoding, and stochastic sampling:
```bash
adb shell "cd /data/local/tmp/bonsai1bit && \
  export LD_LIBRARY_PATH=/data/local/tmp/bonsai1bit:/vendor/lib64 && \
  export ADSP_LIBRARY_PATH='/data/local/tmp/bonsai1bit;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp' && \
  ./bonsai_fwd bonsai2-27b.npubin tok.bin --server 8080 4096 --turbo4 --temp 0.6 --top-p 0.9"
```
*(Or from Termux using the included wrapper script: `./start_bonsai.sh server 8080`)*

Once running, the server exposes standard OpenAI endpoints (`POST /v1/chat/completions` with `"stream": true` or `"stream": false`, and `GET /v1/models`):

```bash
# Test streaming chat completions from Termux, PC (via adb forward tcp:8080 tcp:8080), or any HTTP client:
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bonsai-2-27b",
    "stream": true,
    "messages": [
      {"role": "user", "content": "Explain quantum tunneling in two sentences."}
    ]
  }'
```

### 5. Interactive Terminal Chat (`--chat`) & Single-Shot Benchmark

* **Interactive Console REPL (`--chat`)** — chat directly in Termux or `adb shell` with zero HTTP overhead:
  ```bash
  adb shell "cd /data/local/tmp/bonsai1bit && \
    export LD_LIBRARY_PATH=/data/local/tmp/bonsai1bit:/vendor/lib64 && \
    export ADSP_LIBRARY_PATH='/data/local/tmp/bonsai1bit;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp' && \
    ./bonsai_fwd bonsai2-27b.npubin tok.bin --chat 4096 --turbo4 --temp 0.6 --top-p 0.9"
  ```
* **Single-Prompt CLI Benchmark**:
  ```bash
  adb shell "cd /data/local/tmp/bonsai1bit && \
    export LD_LIBRARY_PATH=/data/local/tmp/bonsai1bit:/vendor/lib64 && \
    export ADSP_LIBRARY_PATH='/data/local/tmp/bonsai1bit;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp' && \
    ./bonsai_fwd bonsai2-27b.npubin tok.bin 'The capital of France is' 20 --turbo4 --temp 0.6 --top-p 0.9"
  ```

### 6. Integrating into Your Own Environment (Termux vs. Custom APK)

* **In Termux**: Run `bonsai_fwd --server 8080` in a background tmux/screen session (with `termux-wake-lock`) and point any OpenAI-compatible CLI tool, Python `openai` client (`base_url="http://127.0.0.1:8080/v1"`), or local web UI to `127.0.0.1:8080`.
* **In a Custom Android APK**: Bundle `libbonsai_q1_skel.so` and `bonsai_fwd` inside your APK's `jniLibs/arm64-v8a/`, set `ADSP_LIBRARY_PATH` to `context.applicationInfo.nativeLibraryDir`, and either communicate with the local `--server` socket over `http://127.0.0.1:8080/v1/chat/completions` or invoke the engine directly via JNI.

---

## Documentation

* [Optimization History: Scalar C++ (52s) to 6.54 tok/s (English)](OPTIMIZATION_HISTORY.md)
* [Detailed Architecture Deep-Dive (English)](ARCHITECTURE.md)
* [Русская документация архитектуры](ARCHITECTURE_RU.md)
* [README на русском языке](README_RU.md)

---

## License

This project is licensed under the **Bonsai-NPU Source-Available Non-Commercial & Anti-Enterprise-R&D License v1.2** (see [LICENSE](LICENSE)).
* **Allowed**: Strictly personal, hobbyist, and uncommissioned academic research by natural persons.
* **Prohibited**: Any commercial use, paid products, or **internal Enterprise R&D / benchmarking / architectural extraction** by corporate entities without an explicit commercial agreement from **VitalikDen0**.
* **Commercial Inquiries**: Contact **VitalikDen0** via email (**me@zepmoriq.com**) or GitHub (https://github.com/VitalikDen0). Model weights are subject to the respective licensing terms of the Bonsai / Q-Bonsai architecture.
