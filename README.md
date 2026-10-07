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
| **Standard Decode** | B = 1 | 345 ms | 2.90 tok/s | 81.1% of DRAM Bandwidth Limit |
| **MTP Batch Verify** | B = 4 | 380 ms | 10.52 tok/s | Weights loaded once per 4 tokens |
| **MTP Speculative Decode** | Variable | — | **5.5–8.0 tok/s** | Expected acceptance rate 70–80% |
| **Prefill Throughput** | B = 8 | 88 ms/tok | 11.36 tok/s | Fused GEMV batching |

*Hardware: OnePlus 13 (Snapdragon 8 Elite, 16 GB LPDDR5X @ 106.7 GB/s peak bandwidth, Hexagon v79 DSP).*

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

## Quick Start (Termux / ADB)

### Build Prerequisites
* Qualcomm Hexagon SDK 5.x / 6.x (or Hexagon LLVM Clang toolchain)
* Android NDK (r25c or newer, `aarch64-linux-android`)
* Python 3.10+ (for weight serialization and model repacking)

### Compilation
```bash
# 1. Compile Hexagon DSP skeleton (.so)
hexagon-clang -mv79 -O3 -fvectorize -mhvx -mhvx-length=128b \
  -shared -fPIC -o cdsp/libbonsai_q1_skel.so cdsp/bonsai_hvx.c cdsp/bonsai_imp.c cdsp/gen/bonsai_skel.c

# 2. Compile ARM64 host engine
$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang \
  -O3 -march=armv8.7-a -pthread -o cdsp/bonsai_fwd cdsp/bonsai_fwd_main.c cdsp/tok.c cdsp/ucat.c cdsp/gen/bonsai_stub.c -ldl
```

### Execution on Device
```bash
# Push binaries and skeleton to device (works on non-rooted phones via ADB)
adb push cdsp/bonsai_fwd /data/local/tmp/
adb push cdsp/libbonsai_q1_skel.so /data/local/tmp/

# Start interactive generation
adb shell "export ADSP_LIBRARY_PATH=/data/local/tmp; cd /data/local/tmp && ./bonsai_fwd bonsai27b-1bit.npubin tok.bin \"Hello, tell me about yourself\" 32"
```

For HTTP server mode:
```bash
adb shell "export ADSP_LIBRARY_PATH=/data/local/tmp; cd /data/local/tmp && ./bonsai_fwd bonsai27b-1bit.npubin tok.bin --server 8080"
```

---

## Roadmap & Upcoming Enhancements

* [ ] **Multi-Token Prediction (MTP) Speculative Decoding**: Integrating CPU prompt-lookup n-gram proposer with the batched NPU verification kernel (`forward_tokens_batch` B = 4) directly into the main decode loop to achieve real-world **5.0–8.0 tok/s**.
* [ ] **TurboQuant & Turbo4 Mode (4-Bit KV-Cache)**: Implementing 4-bit nibble packing (`dsp_q4_enc`) and fast vectorized dot-product in HVX to slash KV-cache footprint to **16 KiB/token**, enabling **65k–262k context** within mobile RAM limits.
* [ ] **Android JNI & Standalone APK**: Providing ready-to-run JNI wrappers and an on-device UI application for one-click installation without ADB or terminal setup.

---

## Documentation

* [Detailed Architecture Deep-Dive (English)](ARCHITECTURE.md)
* [Русская документация архитектуры](ARCHITECTURE_RU.md)
* [README на русском языке](README_RU.md)

---

## License

This project is licensed under the **Bonsai-NPU Source-Available Non-Commercial & Anti-Enterprise-R&D License v1.2** (see [LICENSE](LICENSE)).
* **Allowed**: Strictly personal, hobbyist, and uncommissioned academic research by natural persons.
* **Prohibited**: Any commercial use, paid products, or **internal Enterprise R&D / benchmarking / architectural extraction** by corporate entities without an explicit commercial agreement from **VitalikDen0**.
* **Commercial Inquiries**: Contact **VitalikDen0** via email (**me@zepmoriq.com**) or GitHub (https://github.com/VitalikDen0). Model weights are subject to the respective licensing terms of the Bonsai / Q-Bonsai architecture.
