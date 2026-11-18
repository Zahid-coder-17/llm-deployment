# LLM Deployment: Embeddra Edge Runtime (EER)

<p align="center">
  <b>A Zero-Cloud, High-Performance On-Device LLM Deployment & Inference Platform</b>
  <br>
  <i>Native C++20 Core • ARM Mali OpenCL 3.0 FP16 UMA • 8-Core ARM NEON SIMD • Android NDK</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Under%20Active%20Development-orange?style=flat-square&logo=git" alt="Status">
  <img src="https://img.shields.io/badge/C%2B%2B-20-blue.svg?style=flat-square&logo=cplusplus" alt="C++20">
  <img src="https://img.shields.io/badge/Android-NDK%20r26-green.svg?style=flat-square&logo=android" alt="Android NDK">
  <img src="https://img.shields.io/badge/GPU-Mali%20OpenCL%20FP16-red.svg?style=flat-square" alt="Mali OpenCL">
  <img src="https://img.shields.io/badge/CPU-ARM%20NEON%208--Core-orange.svg?style=flat-square" alt="ARM NEON">
  <img src="https://img.shields.io/badge/Cloud%20Dependency-0%25%20(100%25%20Offline)-brightgreen.svg?style=flat-square" alt="Offline">
  <img src="https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg?style=flat-square" alt="License">
</p>

---

> [!WARNING]
> ### 🚧 Project Status: Under Active Development
> This project is currently undergoing active engineering, optimization, and research. Core inference kernels, alignment datasets, mobile integration layers, and memory serialization specs are iterating rapidly. Features, CLI arguments, and internal APIs may change across commits.

---

## 📖 Overview

**LLM Deployment (Embeddra Edge Runtime - EER)** is an end-to-end framework designed to train, optimize, serialize, and deploy Large Language Models directly on consumer mobile devices and edge hardware with **zero cloud dependencies** and **100% privacy preservation**.

Rather than relying on generic, memory-heavy runtimes (such as full PyTorch, ONNX, or Ollama) that introduce latency and excessive overhead on mobile devices, EER provides an optimized bare-metal C++20 engine that interfaces directly with mobile silicon:
* **ARM Mali GPUs**: Custom OpenCL 3.0 vector-8 half-precision (`half8`) compute kernels utilizing unified memory architecture (UMA).
* **8-Core ARM CPUs**: Multi-threaded thread pool with native ARMv8.2-A NEON SIMD instructions (`float16x8_t`).
* **Zero-Copy Memory-Mapped I/O**: Custom 4KB page-aligned binary serialization format (`.eer`) allowing instant startup without loading entire model weights into RAM.
* **On-Device Agentic Capabilities**: Full Android Accessibility Service integration (`Yori`) and Desktop Agent bridge for autonomous UI perception and multi-step action execution.

---

## 🚦 Development Status & Feature Roadmap

| Module | Feature | Status | Notes |
| :--- | :--- | :---: | :--- |
| **Inference Engine** | Native C++20 HyperGQE Core | ✅ Completed | Custom dynamic routing & SDPA kernels |
| | ARM Mali OpenCL 3.0 FP16 Acceleration | ✅ Completed | Vectorized `half8` kernels with UMA zero-copy |
| | ARM NEON 8-Core Multi-threaded SIMD | ✅ Completed | Static thread pool with core affinity binding |
| | 4KB Page-Aligned `.eer` Binary Serializer | ✅ Completed | `mmap` zero-copy flash weights loading |
| | INT4 / INT8 Mixed Quantization | 🔄 In Progress | Calibration and Per-Channel AWQ kernels |
| **Android Platform** | Native JNI C++ Bridge | ✅ Completed | Low-latency streaming token callbacks |
| | Standalone APK with Jetpack UI | ✅ Completed | Real-time chat & token generation telemetry |
| | Yori Accessibility Automation Service | 🔄 In Progress | Screen parsing, UI hierarchy inspection & touch injection |
| **Training & Alignment** | Dynamic Router Hypernetwork SFT | ✅ Completed | HyperGQE multi-expert group routing |
| | Online RL / GRPO Multi-Step UI Policy | 🔄 In Progress | Reinforcement learning on mobile navigation tasks |
| | Black-Box Self-Correction / Polish | 🔄 In Progress | Synthetic data refinement & recovery pipelines |
| **Desktop & Edge** | Windows / Linux Desktop Agent (`yori_desktop`)| 🔄 In Progress | Python/C++ local execution bridge |
| | Apple Silicon / Metal Performance Shaders | ⏳ Planned | Future exploration for iOS / macOS execution |

---

## 🏛️ Model Architecture: Maria-1B (HyperGQE)

The reference architecture optimized in this repository is **Maria-1B**, featuring **HyperGQE (Hypernetwork-based Grouped Query Experts)** attention:

```
+-----------------------------------------------------------------------------------+
|                               Maria-1B Geometry                                   |
+-----------------------------------------------------------------------------------+
|  Parameters:           1,074,481,152 (~1.074 Billion unique FP16 parameters)      |
|  Layers:               18 Transformer Decoder Blocks                              |
|  Hidden Dimension:     d_model = 2048                                             |
|  Attention Heads:      16 Query Heads (d_head = 128), 4 KV Heads (Grouped Query)  |
|  Attention Mode:       HyperGQE Dynamic Routing + Expert Hypernetworks            |
|  Dynamic Router:       Linear(2048 -> 16), Softmax over 4 Expert Groups           |
|  Expert Embeddings:    4 Groups x 4 Experts x 64 Dimensions                       |
|  Hypernetworks:        W_D: (32768, 64) -> (2048, 16), W_U: (2048, 64) -> (16, 128)|
|  FFN Intermediate:     d_ffn = 5632 (SwiGLU: gate_proj, up_proj, down_proj)       |
|  Compressed Output:    o_proj: 768 -> 2048 (62.5% reduction vs standard dense)   |
|  Normalization:        RMSNorm (eps = 1e-5)                                       |
|  Vocabulary:           128,256 tokens (Byte-Pair Encoding)                        |
|  Weight Tying:         lm_head.weight == embed_tokens.weight                      |
+-----------------------------------------------------------------------------------+
```

---

## ⚡ Empirical On-Device Benchmarks

Tested and verified on a physical **Samsung Galaxy A35 5G (`SM-A356E`, Exynos 1380 SoC, 8 CPU cores, Mali-G68 MP5 GPU, Android 16)**:

### Dual-Hardware Comparison (8-Core CPU vs. Mali-G68 GPU)

| Benchmark Metric | **8-Core CPU (ARM NEON)** | **Mobile GPU (Mali-G68 MP5)** | Speedup / Efficiency |
| :--- | :--- | :--- | :--- |
| **Compute Technology** | **4x Cortex-A78 + 4x Cortex-A55** | **5 Valhall Execution Engines** | Hardware Native |
| **Active Memory Overhead** | 2,148 MB | **1,980 MB** | 168 MB less RAM (GPU zero-copy buffer) |
| **Prefill Speed (Prompt)** | 14.8 tokens/sec | **28.6 tokens/sec** | **1.93x faster** prefill on GPU |
| **Autoregressive Generation** | 4.1 tokens/sec | **9.2 tokens/sec** | **2.24x faster** token generation |
| **Thermal Throttle (15 min)** | Mild frequency drop after 8m | **Stable** sustained clock rate | GPU distributes heat across die |
| **RAM Utilization Type** | CPU Thread Stack Copies | **Zero-Copy Mmap (UMA)** | Direct buffer pointers without duplicate copies |

---

## 📁 Repository Structure

```
llm-deployment/
├── embeddra-edge/                         # Core Edge Inference & Deployment Runtime
│   ├── android/                           # Native Android Application & Build Toolchain
│   │   ├── app/                           # Kotlin Jetpack frontend UI & Accessibility Services
│   │   │   ├── src/main/java/com/embeddra/edge/
│   │   │   │   ├── MainActivity.kt        # Interactive UI & Streaming Console
│   │   │   │   ├── EmbeddraRuntime.kt     # Android Kotlin SDK wrapper
│   │   │   │   ├── ActionParser.kt        # LLM UI Action parser & executor
│   │   │   │   ├── YoriAccessibilityService.kt # Accessibility driver
│   │   │   │   └── VocabularyDecoder.kt   # Native 128k BPE Tokenizer
│   │   │   └── src/main/res/              # Android layouts, themes, drawables
│   │   ├── jni/                           # JNI C++ <-> Java Bridge
│   │   └── build_apk.ps1                  # Automated headless NDK build & deploy script
│   ├── backends/
│   │   ├── custom_cpp/                    # Native C++20 HyperGQE Core Engine
│   │   │   ├── hypergqe_engine.h          # RoPE, SDPA, SwiGLU, RMSNorm headers
│   │   │   └── hypergqe_engine.cpp        # 8-Core ARM NEON FP16 SIMD kernels
│   │   └── gpu/                           # Mobile GPU OpenCL Acceleration
│   │       ├── gpu_engine.h               # OpenCL 3.0 command queues & buffers
│   │       └── gpu_engine.cpp             # Vectorized half8 compute kernels
│   ├── core/                              # Hardware detection, memory pools, stats
│   ├── export/                            # Weight serialization & quantization scripts
│   │   ├── serialize_to_eer.py            # Converts PyTorch weights to 4KB page-aligned .eer
│   │   └── quantize_maria.py              # Calibration & mixed-precision quantization
│   ├── hypergqe/                          # PyTorch reference implementations
│   └── CMakeLists.txt                     # Multi-architecture CMake build definition
├── yori_desktop/                          # Local desktop companion agent
│   ├── desktop_agent.py                   # PC agent execution engine
│   ├── windows_inspector.py               # Windows UI Automation inspector
│   └── server.py                          # Local HTTP/WebSocket bridge
├── train_*.py                             # SFT, RL, GRPO, and UI-tuning scripts
├── eval_*.py                              # Perplexity, accuracy, and benchmark scripts
├── benchmark_100_scenarios.py             # 100 Real-world mobile scenario validation suite
└── README.md                              # Project documentation
```

---

## 🚀 Getting Started

### 1. Serializing Weights to `.eer` Format
Convert standard PyTorch checkpoints into 4KB page-aligned binary files for instant zero-copy loading:
```bash
python embeddra-edge/export/serialize_to_eer.py \
    --input checkpoints/maria_1b_weights.pt \
    --output checkpoints/maria_1b.eer
```

### 2. Native C++20 Inference (Desktop / Linux Edge)
```cpp
#include "embeddra_runtime.h"

int main() {
    embeddra::runtime::EmbeddraRuntime runtime;
    runtime.initialize();

    // Select compute backend: 0 = Mali GPU OpenCL, 1 = 8-Core CPU NEON
    runtime.setBackendMode(0);

    // Memory-map local model weights with zero memory duplication
    runtime.loadModel("maria_1b.eer");

    // Execute generation with streaming callback
    runtime.generate("Clinical Assessment: Patient has persistent cough...", 64,
        [](const std::string& token, int token_id, bool is_final) {
            std::cout << token << std::flush;
        }
    );

    runtime.unloadModel();
    return 0;
}
```

### 3. Build & Deploy Android APK
Build the C++20 NDK native libraries and assemble the Android APK:
```powershell
powershell -ExecutionPolicy Bypass -File .\embeddra-edge\android\build_apk.ps1
```

### 4. Running the Evaluation Suite
Validate token generation accuracy and perplexity across benchmarks:
```bash
python benchmark_100_scenarios.py
python eval_current_ppl_and_accuracy.py
```

---

## 🔒 Privacy & Edge Security

* **100% On-Device & Air-Gapped**: Runs entirely offline with zero network requests or telemetry.
* **Unified Memory Security**: Tensors are mapped strictly read-only (`PROT_READ`), avoiding state corruption.
* **Ephemeral Context**: Session contexts remain resident in device RAM only for the duration of the process.

---

## 🤝 Contributing & Feedback

Because this project is **under active development**, feedback, bug reports, and suggestions are welcome!
Feel free to open an issue or pull request on GitHub to report edge device performance, compiler compatibility, or architectural questions.

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).
