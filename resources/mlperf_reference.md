# MLPerf Reference: Benchmarks, Participation, and Strategy

Reference doc for TinyHVM. Researched 2026-03-25 from mlcommons.org and GitHub.

---

## 1. What is MLPerf

**MLPerf** is the industry-standard benchmark suite for ML systems, run by **MLCommons** — a 501(c)(6) non-profit with 125+ member organizations. Created in 2018 by Google, Intel, Harvard, Stanford, UC Berkeley, and others. Analogous to what SPEC does for CPUs.

Results are peer-reviewed by fellow submitters before publication. Every major ML hardware vendor participates. Cited in product launches and procurement decisions. The gold standard.

---

## 2. Benchmark Suites

MLCommons maintains **10 benchmark categories**:

| Suite | Focus | Cadence |
|-------|-------|---------|
| **Training** | Time to train models to target quality | 2x/year (~Jun, ~Nov) |
| **Training: HPC** | Training on supercomputers | With Training |
| **Inference: Datacenter** | Inference throughput/latency in DCs | 2x/year (~Apr, ~Sep) |
| **Inference: Edge** | Inference on edge devices | With Inference DC |
| **Inference: Tiny** | Inference on MCUs (<50mW) | ~1x/year |
| **Inference: Mobile** | Inference on phones | Periodic |
| **Client** | LLMs on PCs/laptops | ~1x/year |
| **Storage** | Storage I/O for ML training | ~1x/year |
| **Automotive** | ADAS/AD embedded systems | Early stage |
| **Endpoints** | GenAI serving via API | Rolling submissions |

### Training (v5.1 — Nov 2025)

Wall-clock time to train to target quality. Seven benchmarks:

| Benchmark | Model | Dataset | Quality Target |
|-----------|-------|---------|---------------|
| LLM Pretrain (large) | Llama 3.1 405B | C4 v3.0.1 | 5.6 log perplexity |
| LLM Pretrain (small) | Llama 3.1 8B | C4 | TBD (new v5.1) |
| LLM Fine-tune | Llama 2 70B LoRA | SCROLLS GovReport | 0.925 CE loss |
| Image Generation | FLUX.1 (11.9B) | CC12M / COCO 2014 | TBD (new v5.1) |
| Object Detection | RetinaNet | Open Images | 34.0% mAP |
| Recommendation | DLRM-dcnv2 | Criteo 4TB | 0.8032 AUC |
| Graph Neural Net | R-GAT | IGBH-Full | 72% accuracy |

Retired: ResNet-50 (after v4.1), BERT-large (after v5.0), GPT-3 175B (after v4.1), Stable Diffusion v2 (after v5.0).

### Inference (v5.1 — Sep 2025)

| Benchmark | Model | Dataset | Category |
|-----------|-------|---------|----------|
| Image Classification | ResNet-50 v1.5 | ImageNet 50K | DC + Edge |
| Object Detection | RetinaNet 800x800 | OpenImages 24.8K | DC + Edge |
| Medical Imaging | 3D-UNet | KiTS2019 | DC + Edge |
| Language (QA) | BERT-Large | SQuAD v1.1 | Edge only |
| Summarization | Llama 3.1 8B | CNN-DailyMail | DC + Edge |
| LLM | Llama 2 70B | OpenORCA 24.6K | DC only |
| LLM (large) | Llama 3.1 405B | various | DC only |
| MoE LLM | Mixtral 8x7B | mixed 15K | DC only |
| Reasoning | DeepSeek-R1 (671B) | math/QA/code | DC only |
| Text-to-Image | SDXL | COCO 2014 5K | DC + Edge |
| Speech | Whisper Large V3 | LibriSpeech | DC + Edge |
| Recommendation | DLRM-v2 | Synthetic Criteo | DC only |
| GNN | R-GAT | IGBH 788K | DC only |
| Automotive | PointPainting | Waymo 40K | Edge |

**Four scenarios:**

| Scenario | What it models | Metric |
|----------|---------------|--------|
| Offline | Bulk processing | samples/sec (or tokens/sec) |
| Server | Poisson queries | queries/sec under latency SLA |
| Single-Stream | One at a time | 90th-percentile latency (ms) |
| Multi-Stream | Parallel streams | streams meeting latency target |
| Interactive (new) | Agentic LLM | TTFT + TPOT |

DC = Server + Offline. Edge = Single-Stream + Offline.

### Tiny (v1.3 — Sep 2025)

Targets 10–250 MHz MCUs, <50mW, models under 100KB:

| Benchmark | Model | Dataset | Accuracy |
|-----------|-------|---------|----------|
| Keyword Spotting | DS-CNN | Speech Commands | 90% |
| Visual Wake Words | MobileNet | VWW | 80% |
| Image Classification | ResNet (small) | CIFAR-10 | 85% |
| Anomaly Detection | AutoEncoder | ToyADMOS | 0.85 AUC |
| Streaming Wakeword | 1D DS-CNN | Custom audio | <=8 FP/FN |

### Training: HPC

| Benchmark | Model | Dataset |
|-----------|-------|---------|
| CosmoFlow | 3D CNN | N-body cosmological simulation |
| DeepCAM | Encoder-decoder CNN | CAM5+TECA climate sim |
| OpenCatalyst | DimeNet++ (GNN) | OC20 catalyst dataset |

### Storage, Client, Automotive

- **Storage (v2.0):** DLIO-based I/O throughput. Emulates compute to isolate storage. Does NOT require expensive accelerators.
- **Client (v1.5):** Llama 2 7B, Llama 3.1 8B, Phi 3.5 Mini on PCs. Metrics: TTFT and tokens/sec. Weights quantized to int4.
- **Automotive (v0.5):** 2D recognition, 2D segmentation, 3D recognition. Still early (2 submitters in v0.5).

---

## 3. How to Participate

### Step by step

1. **Join MLCommons** — contact info@mlcommons.org. Individual and academic membership is **free**. Very small orgs (<10 employees) can get free Observer membership.

2. **Join the working group** for your target benchmark (Training, Inference, Tiny, etc.).

3. **Register** for the submission round (deadlines are several weeks before submission).

4. **Run benchmarks** using reference implementations from GitHub:
   - `github.com/mlcommons/inference` — Inference
   - `github.com/mlcommons/training` — Training
   - `github.com/mlcommons/tiny` — Tiny

5. **Submit** code, logs, system descriptions by deadline. Results undergo peer review.

6. **Results published** on mlcommons.org.

### Upcoming deadlines

| Round | Submission Deadline | Results Published |
|-------|-------------------|------------------|
| Inference v6.0 | **Feb 13, 2026** | Apr 1, 2026 |
| Training v6.0 | **May 15, 2026** | TBD |

### What you submit

- Source code (must be open-sourced for Closed division)
- Run logs with timing and accuracy
- System description (hardware, software, config)
- Compliance checker output (automated validation)

### Closed vs Open division

| | Closed | Open |
|---|--------|------|
| Model | Must use reference (or equivalent) | Any model |
| Accuracy | Within 99% of FP32 reference | Report only, no minimum |
| Scenarios | Must submit ALL required scenarios | Any single scenario |
| Pre/post-processing | Must match reference | Any |
| Purpose | Apples-to-apples comparison | Showcase innovation |
| Code | Must be open-sourced | Must be open-sourced |

### Can a solo developer participate?

**Yes.** Individual membership is free. In MLPerf Inference v5.1, an individual ("Amitash Nanda") submitted as a solo. GATEOverflow (small community group) regularly submits. Academic labs submit routinely.

### Cost

| Item | Cost |
|------|------|
| Individual membership | Free |
| Academic membership | Free |
| Small org (<10 employees) | Free (board approval) |
| Mid org (10–499 employees) | $18K/year + $18K initiation |
| Large org (500+) | $90K/year + $90K initiation |
| Hardware | Whatever you already have |
| ImageNet dataset | ~150 GB storage |
| Large datasets (Criteo, C4) | 800 GB – 4 TB |

---

## 4. Leaderboards and Results

### Where to view

| Suite | URL |
|-------|-----|
| Training | mlcommons.org/benchmarks/training/ |
| Inference DC | mlcommons.org/benchmarks/inference-datacenter/ |
| Inference Edge | mlcommons.org/benchmarks/inference-edge/ |
| Inference Tiny | mlcommons.org/benchmarks/inference-tiny/ |
| Storage | mlcommons.org/benchmarks/storage/ |
| Client | mlcommons.org/benchmarks/client/ |
| Automotive | mlcommons.org/benchmarks/mlperf-automotive/ |
| Raw results | github.com/mlcommons/inference_results_v5.1 etc. |

Interactive tables with filters by submitter, accelerator, system config, scenario, division.

### How results are categorized

- By **accelerator** (NVIDIA B200, Google Trillium TPU, AMD MI300X, Intel Gaudi, etc.)
- By **system configuration** (number of accelerators, nodes)
- By **submitter** (NVIDIA, Google, Lambda, CoreWeave, Nebius, etc.)
- By **availability**: Available (purchasable), Preview (announced), Research/Development

### Metrics

| Benchmark Type | Primary Metric |
|---------------|---------------|
| Training | Wall-clock **time-to-train** (minutes) |
| Inference Offline | **samples/sec** or **tokens/sec** |
| Inference Server | **queries/sec** under latency SLA |
| Inference Single-Stream | **90th-percentile latency** (ms) |
| Inference Interactive | **TTFT** + **TPOT** |
| Tiny | **latency** (ms) + **energy/inference** (uJ) |
| Storage | **throughput** (samples/sec) + **accelerator utilization** |

### Notable record holders (late 2025)

- **NVIDIA** — swept all 7 Training v5.1 benchmarks with Blackwell Ultra (GB300 NVL72). Trained Llama 3.1 405B in 10 minutes with 5000+ GPUs. Dominates both Training and Inference.
- **Google** — Trillium TPU debuted in v5.0 with 3.5x throughput gain over TPU v5e on Stable Diffusion.
- **AMD** — first Training submission with MI300X in v5.0.
- **Lambda, Nebius, CoreWeave** — cloud providers showcasing NVIDIA hardware.
- **Syntiant** — leads Tiny for energy efficiency.
- **Qualcomm, Broadcom** — active in Inference.

Inference v5.1: 27 organizations, 65 unique systems, 12 different accelerators.

---

## 5. Rules and Compliance

### Accuracy targets (Closed division)

- Must reach **99% of FP32 reference accuracy** (99.9% for high-accuracy variants)
- Open division: no accuracy requirement, but must report accuracy

### Timing rules

- **Training:** wall-clock from start to target quality. Multiple runs required (benchmark-specific), highest/lowest discarded, rest averaged.
- **Inference:** measured by **LoadGen** (standard C++ load generator from MLCommons). Pre/post-processing included. Minimum 600-second duration. 1 valid run per scenario.

### Reproducibility

- All source code open-sourced (Closed)
- Results must be replicable
- Fixed random seeds (Mersenne Twister mt19937)
- Quantization methods publicly described
- Automated compliance suite validates submissions
- Peer review by fellow submitters

### System classification

- **Available** — commercially purchasable hardware and software
- **Preview** — announced but not yet available
- **Research/Development** — prototype or research systems

---

## 6. Strategy for TinyHVM

### Target: MLPerf Inference Edge, Open Division, ResNet-50 Single-Stream

This is the most accessible entry point:

| Aspect | Details |
|--------|---------|
| Model | ResNet-50 v1.5 (conv, BN, ReLU, pool, FC) |
| Dataset | ImageNet validation set (~150 GB, 50K images) |
| Scenario | Single-Stream (process one image, report latency) |
| Division | Open (any framework, any quantization) |
| Accuracy | Report only — no minimum in Open |
| Hardware | Single Mac with Apple Silicon |
| Cost | $0 (free individual membership) |

### What TinyHVM already has

- Conv2d, BatchNorm, ReLU, MaxPool, Linear — all the layers ResNet-50 needs
- Metal backend on Apple Silicon with compute shaders and MPS matmul
- Strided view algebra for reshape/permute/expand

### What TinyHVM needs to build

1. **LoadGen integration** — MLCommons provides `loadgen` as a C++ library. It generates queries and measures timing. We write a C callback that runs inference and returns results. This is the main integration work.

2. **Weight loading** — load pre-trained ResNet-50 weights. Options:
   - Convert from ONNX/PyTorch checkpoint to raw float arrays
   - Write a simple loader that reads weight files into `thvm_tensor_from_host`

3. **ImageNet preprocessing** — resize to 224x224, normalize with ImageNet mean/std. Can use stb_image or a simple JPEG decoder.

4. **Full ResNet-50 inference graph** — build the 50-layer forward pass using our existing ops. No autograd needed (inference only).

5. **Accuracy validation** — run all 50K ImageNet validation images, report top-1 accuracy.

### What it buys us

Even with modest numbers compared to NVIDIA datacenter GPUs, a submission from a novel small C/Metal framework would be:
- **Legitimate MLPerf-certified benchmarks** for Apple Silicon via a non-PyTorch/non-TensorFlow stack
- **Proof that TinyHVM is real** — a working ML framework, not just a research prototype
- **Visibility** — results are published alongside NVIDIA, Google, AMD, etc.
- **Baseline** — gives us a quantitative starting point to optimize against

### Not realistic for TinyHVM

- **Training benchmarks** — require 1000s of GPUs and massive datasets
- **Datacenter inference** — requires server-grade hardware, all scenarios
- **Closed division** — requires exact model equivalence and full scenario coverage
- **LLM benchmarks** — require transformer attention, KV-cache, large model weight management

### Timeline estimate

Given existing ops, a minimum viable ResNet-50 Inference Edge submission could target:
- **Inference v6.0** submission deadline: Feb 13, 2026 (tight but possible if focused)
- **Inference v6.1** (likely ~Sep 2026): more realistic target with polish time

---

## 7. Software-Focused Benchmarks and Landscape

MLPerf is a **system** benchmark (hardware + software together). There is no dedicated software-only leaderboard. But there are ways to showcase framework quality.

### MLPerf framework tracking

Every MLPerf submission lists its ML framework. The interactive dashboards allow filtering by framework. But it's metadata, not a controlled variable — comparing frameworks requires finding submissions on identical hardware with different software, which is rare.

### Precedent: tinygrad MLPerf submissions

**tinygrad has made official MLPerf submissions** — Training v5.0 (BERT on tinybox) and Training v6.0 (LLaMA on AMD MI300X/MI350X). This proves a small framework (~8K LoC) can compete on a recognized leaderboard. Directly relevant precedent for TinyHVM.

### Other benchmarks

| Benchmark | What it measures | Metal? | Status |
|-----------|-----------------|--------|--------|
| **KernelBench** (Stanford, 2025) | AI-generated kernel quality vs PyTorch. 250+ workloads. | No (CUDA/Triton/HIP) | Active, leaderboard planned |
| **Gimlet Labs Metal Kernels** (2025) | AI-generated Metal kernels on Apple Silicon. 1.2–1.9x speedup over PyTorch baseline. | **Yes** | One-off study |
| **MultiKernelBench** (2025) | Kernel generation across 6 platforms (CUDA, Triton, AscendC, Pallas, SYCL, TileLang). | No (Metal not supported) | Active |
| **TritonBench** (2025) | 184 real-world Triton operators from GitHub. | No | Active |
| **TritonGym** (2025) | Agentic Triton kernel generation workflow. | No | Active, has leaderboard |
| **FlashInfer-Bench** (2025) | LLM serving kernels (MLSys 2026 competition). | No (B200 only) | Competition |
| **KernelCraft** (2026) | Kernel generation for emerging/novel accelerator ISAs. | No | Active |
| **DAWNBench** (Stanford) | Compared frameworks on same tasks. | No | **Retired** (2020) |
| **metal-benchmarks** (philipturner) | Raw Apple GPU FLOPS. M4: ~2.9 TFLOPS FP32 via MPS. | **Yes** | Active |
| **llama.cpp Discussion #4167** | Crowdsourced Apple Silicon throughput table across M-series chips. | **Yes** | Community-maintained |

### Apple Silicon framework comparison (arXiv:2511.05502, Nov 2025)

LLM inference throughput on Apple Silicon:

```
MLX (~230 tok/s) > MLC-LLM (~190 tok/s) > llama.cpp (~150 tok/s) > Ollama (20-40 tok/s) > PyTorch MPS (~7-9 tok/s)
```

### Gaps and opportunities

- **No Metal kernel benchmark leaderboard** — CUDA/Triton dominated
- **No Apple Silicon ML framework comparison leaderboard** — only one-off papers
- **No minimal-framework benchmark** — nothing measures LOC vs performance
- **No MultiKernelBench Metal backend** — could be a contribution opportunity

### Bottom line for TinyHVM

The most credible path to a recognized leaderboard remains **MLPerf Inference Edge, Open Division**. tinygrad proved a small framework can do this. For Apple Silicon specifically, the landscape is sparse — there's room to be the first Metal-native C framework with MLPerf numbers.

---

## 8. Key GitHub Repos

| Repo | Purpose |
|------|---------|
| `github.com/mlcommons/inference` | Inference reference implementations + LoadGen |
| `github.com/mlcommons/training` | Training reference implementations |
| `github.com/mlcommons/tiny` | Tiny reference implementations |
| `github.com/mlcommons/inference_policies` | Inference rules document |
| `github.com/mlcommons/training_policies` | Training rules document |
| `github.com/mlcommons/policies` | General submission rules |
| `github.com/mlcommons/inference_results_v5.1` | Raw v5.1 Inference results |

---

## 9. Key Datasets

| Dataset | Size | Used By |
|---------|------|---------|
| ImageNet 2012 (validation) | ~150 GB | Inference: ResNet-50 |
| OpenImages | ~500 GB | Inference: RetinaNet; Training: RetinaNet |
| SQuAD v1.1 | ~35 MB | Inference: BERT |
| COCO 2014 | ~20 GB | Inference: SDXL eval; Training: FLUX eval |
| CC12M | ~100 GB | Training: FLUX |
| C4 v3.0.1 | ~800 GB | Training: Llama pretraining |
| Criteo 4TB | ~4 TB | Training: DLRM |
| CNN-DailyMail | ~600 MB | Inference: Llama summarization |
| OpenORCA | ~2 GB | Inference: Llama 2 70B |
| KiTS2019 | ~30 GB | Inference: 3D-UNet |
| LibriSpeech | ~60 GB | Inference: Whisper |
| IGBH-Full | ~1 TB | Training/Inference: R-GAT |
| CIFAR-10 | ~170 MB | Tiny: image classification |
| Speech Commands | ~2.3 GB | Tiny: keyword spotting |
