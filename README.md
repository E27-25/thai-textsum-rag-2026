# thai-textsum-rag-2026

> **Query-based Thai Meeting-Minutes Summarization with Evidence Retrieval**
> RAG + LLM pipeline submitted to the **2026 TextSum** competition (Thai Parliament meeting minutes).
>
> **Final best score: 0.678** (Qwen3-32B-SFT-v1-AWQ + thinking mode) — submitted 2026-06-03.
>
> Stack: BGE-M3 retriever → bge-reranker-v2-m3 → Qwen3 (LoRA SFT/DPO) → AWQ INT4 → vLLM 0.8.5 on H100.

> ⚠️ **Deadline closed 2026-06-05.** Final submission (`32b-thinking-v2`) crashed with `Exit Status 1` and admin never returned the traceback in time to debug. Best run remains `32b-thinking-v1` at 0.678. See [`docs/FINAL_STATUS.md`](docs/FINAL_STATUS.md) for the full post-mortem.

---

## Table of Contents

1. [The Competition](#the-competition)
2. [Final Pipeline](#final-pipeline)
3. [Results Timeline](#results-timeline)
4. [EDA Findings](#eda-findings)
5. [Repository Layout](#repository-layout)
6. [How to Build & Submit](#how-to-build--submit)
7. [Hardware Constraints & Tricks](#hardware-constraints--tricks)
8. [Lessons Learned](#lessons-learned)

---

## The Competition

**Task.** Given a Thai parliamentary meeting transcript split into numbered paragraphs (`P1`, `P2`, …) and a set of questions, the system must produce, for each question:

1. **`abstractive`** — a Thai short-form answer (≤ ~30 words)
2. **`refs`** — the list of paragraph IDs that support the answer

**Score.**

```
Score = 0.45·SS + 0.35·RougeL + 0.20·IoU
```

| Metric | Meaning |
| --- | --- |
| **SS**  (Semantic Similarity) | `bge-m3` cosine similarity between candidate and reference summary. |
| **RougeL** | Longest common subsequence overlap of the summary text. |
| **IoU**   | Intersection-over-Union of paragraph IDs in `refs`. |

**Hard constraints.**

| Constraint | Value |
| --- | --- |
| GPU | 1 × H100 PCIe (40 GB VRAM) |
| Image size | ≤ 60 GB |
| Per-dataset timeout | 30 minutes |
| Submissions / day | 2 (failures count *only if* they complete) |
| Output language | Thai |

**Data.**

| Split | Meetings | Q/A pairs |
| --- | ---: | ---: |
| Train | 50 | ≈ 1,200 |
| Sample test | 5  | 50 |
| Hidden test | — | — |

---

## Final Pipeline

```
                  ┌────────────────────────────────────┐
   question ─────▶│  BGE-M3 dense retrieval (top-K)    │
                  └──────────────────┬─────────────────┘
                                     │ top-15 paragraphs
                  ┌──────────────────▼─────────────────┐
                  │  bge-reranker-v2-m3 cross-encoder   │
                  └──────────────────┬─────────────────┘
                                     │ top-5 + ± neighbours (cluster expansion)
                  ┌──────────────────▼─────────────────┐
                  │  Qwen3-14B-SFT-v5  (AWQ INT4, vLLM) │
                  │  prompt → JSON  {abstractive, refs} │
                  └──────────────────┬─────────────────┘
                                     │
                          robust 3-stage JSON parser
                                     │
                  ┌──────────────────▼─────────────────┐
                  │  contiguous-ref expansion          │
                  │  (P3,P4,P5 ← merged from P3 + P5)  │
                  └──────────────────┬─────────────────┘
                                     ▼
                            { abstractive, refs }
```

**Components in detail.**

- **Retriever** — local [`bge-m3`](https://huggingface.co/BAAI/bge-m3) dense + sparse + ColBERT. We use the dense head for ranking and discard sparse/ColBERT in the speed-tuned variants to stay inside the 30-minute budget.
- **Reranker** — local [`bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3). Cross-encoder, max_length=384, truncated paragraphs to 500 chars to keep latency low.
- **Generator** — Qwen3 (14B SFT-v5 or 32B SFT-v1). LoRA SFT fine-tuned on the train set, then AWQ-quantised to 4-bit so the whole thing fits next to BGE-M3 on a single H100. Served via **vLLM 0.8.5** with `enforce_eager=False`, `max_num_seqs ≤ 16`, `max_model_len = 6144`, `MAX_NEW_TOKENS = 120`.
- **Neighbour clustering** — after the LLM picks refs, we expand any near-miss IDs by `NEIGHBOR_THRESHOLD = 0.20` to merge contiguous evidence paragraphs (helped IoU by ~3 pts in EDA-replay).
- **Robust JSON parser** — three fallback strategies (full JSON → regex extraction of `abstractive`/`refs` from truncated output → plaintext cleanup). Fixed **51 / 51** broken outputs in a held-out replay with 0 regressions over 200 clean outputs.

---

## Results Timeline

| # | Submission | Model | Score | SS | RougeL | IoU | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | v4 | Qwen3-14B | 0.182 | – | – | ~0 | Bug: hardcoded `P1` ref |
| 2 | v5 / v6 | Qwen3-14B | 0.644 | 0.838 | 0.476 | 0.501 | IoU fix |
| 3 | 14B SFT-v2 + rerank | Qwen3-14B-SFT-v2 | **0.671** | 0.851 | 0.518 | 0.519 | First fine-tune |
| 4 | 35B v15 | Qwen3.6-35B-A3B-FP8 | 0.634 | 0.805 | 0.410 | 0.638 | FP8 MoE; lost RougeL |
| 5 | 32B SFT v1 (AWQ) | Qwen3-32B-SFT-AWQ | **0.678** | – | – | – | **Best so far** |
| 6 | Refboost v1–v4 | 32B-SFT-AWQ + ref-verify | timeout | – | – | – | Eval server slower in late May; all variants > 30 min |
| 7 | 14B AWQ v1 | Qwen3-14B-SFT-v5-AWQ | *pending* | – | – | – | + robust JSON parser + speed optims (final pivot) |

**Why 14B at the end.** Repeated 32B timeouts on the eval server (even a 1-line diff from the working 0.678 submission) forced a downsize. 14B AWQ runs in ~40 % of the wall-time at <2-pt expected score loss.

---

## EDA Findings

A 1239-query analysis of the train set surfaced the failure modes driving the 0.20–0.30 gap from the leaderboard:

| Finding | Number | Impact |
| --- | --- | --- |
| Gold answers are **80 % abstractive** (LCS = 0.18) | 1239 | The original "ใช้ถ้อยคำโดยตรง" prompt was *anti-aligned* with the gold. |
| 50 % of gold answers **restate the question** | ≈ 620 | Adding a restatement skeleton to the prompt clawed back ~2 pts. |
| Median answer length | 7 words | Anything over ~30 words penalised RougeL. |
| Predictions with only 1 ref | 94 % | Gold has 2+ refs in 28 % of queries → under-prediction is the dominant IoU loss. |
| Single-ref preds **partially matching** gold | 67 % | Cluster-expansion (`NEIGHBOR_THRESHOLD = 0.20`) is the cheapest fix. |
| IoU = 0 failures **adjacent (±1)** to gold | 28 % of the 25.3 % IoU=0 cohort | Same cluster-expansion catches these. |
| Truncated / wrapped JSON outputs | 51 / 1239 (4.1 %) | Each costs **−0.257 RougeL** → fixed by the 3-stage parser. |
| Worst documents | `doc_001` (IoU 0.289), `doc_050` (0.342) | Very long, sparsely-distributed evidence — out-of-scope for cheap fixes. |
| Best documents | `doc_048` (IoU 0.861) | Validates the pipeline ceiling. |

---

## Repository Layout

The tree is grouped by *role*, not chronology. The final inference path lives in **`inference/`**, all the LoRA training experiments in **`training/`**, AWQ quantisers in **`quantization/`**, container recipes in **`docker/`**, and the Lanta orchestration scripts in **`scripts/`**.

```
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── docs/                               ← project narrative
│   ├── RECAP.md                            • day-by-day competition log
│   └── STATUS.md                           • raw status snapshots
│
├── inference/                          ← runtime / submission code
│   ├── run_vllm_14b_awq.py                 ★ FINAL — 14B AWQ + 3-stage JSON parser
│   ├── run_vllm_32b.py                     • prior best (0.678, 32B-SFT-AWQ)
│   ├── run_vllm_35b.py                     • Qwen3.6-35B-A3B-FP8 variant
│   ├── run_vllm.py                         • original 14B BF16 path
│   ├── run.py                              • thin /model/run.py CMD shim
│   ├── run_vllm_32b_bestofn.py             • best-of-N sampling ablation
│   ├── run_vllm_32b_hybrid.py              • dense + sparse hybrid retrieval
│   ├── run_vllm_32b_thinking.py            • chain-of-thought ablation
│   ├── run_vllm_32b_twostage.py            • retrieve → answer → re-retrieve
│   ├── run_vllm_32b_refboost.py            • reranker-heavy (timed out)
│   ├── run_vllm_32b_refboost_v3.py         • minimal-diff retry (timed out)
│   ├── run_vllm_32b_refboost_v4.py         • speed-tuned retry (timed out)
│   ├── entrypoint.sh                       • 14B CUDA bootstrap
│   ├── entrypoint_v2.sh                    • 32B variant
│   └── entrypoint_35b.sh                   • 35B / CUDA 13 path
│
├── training/                           ← LoRA fine-tunes
│   ├── sft/                                supervised fine-tuning
│   │   ├── train_sft.py                    • 14B baseline
│   │   ├── train_sft_v5.py                 ★ final 14B SFT
│   │   ├── train_sft_v4_oracle.py          • Oracle-style (gold refs as input)
│   │   ├── train_sft_32b.py / _v2 / _v3    • 32B iterations
│   │   └── train_sft_32b_oracle.py         • 32B Oracle
│   ├── dpo/                                direct preference optimisation
│   │   ├── train_dpo.py                    • 14B DPO
│   │   ├── train_dpo_32b.py / _v2          • 32B DPO
│   │   └── gen_dpo_rejected{,_v2}.py       • rejected-sample synth
│   ├── augmentation/
│   │   └── gen_paraphrases_v3.py           • paraphrase data aug
│   └── merge_lora/                         LoRA → full-weights merge utilities
│       ├── merge_lora{,_v4,_v5}.py             — 14B
│       ├── merge_lora_32b{,_v2,_v3,_oracle}.py — 32B
│       └── merge_lora_dpo{,_32b,_32b_v2}.py    — DPO heads
│
├── quantization/                       ← AWQ INT4 drivers (autoawq + GEMM)
│   ├── quantize_14b_v5_awq.py              ★ final 14B AWQ
│   ├── quantize_32b_awq.py / _v2 / _v3     • 32B AWQ iterations
│   ├── quantize_32b_oracle_awq.py
│   └── quantize_dpo_32b{,_v2}_awq.py
│
├── evaluation/                         ← offline scoring & ablations
│   ├── eval_train.py                       • mirrors the org's eval.py on train set
│   ├── eval_hf.py                          • HuggingFace-side eval
│   ├── eval_vllm.py                        • vLLM-side eval
│   ├── inference.py                        • legacy HF inference
│   ├── inference_train.py                  • HF inference on train split
│   ├── inference_moe.py                    • 35B-MoE inference
│   ├── inference_moe_train.py              • 35B-MoE on train
│   ├── infer_train.py                      • thin wrapper
│   ├── test_pipeline.py                    • end-to-end smoke test
│   └── test_fp8_quick.py                   • FP8 sanity check
│
├── docker/                             ← container recipes
│   ├── Dockerfile.14b_awq                  ★ FINAL submission image
│   ├── Dockerfile                          • original 14B
│   ├── Dockerfile.9b / .9b_v2              • Qwen3.5-9B trials
│   ├── Dockerfile.32b                      • 32B baseline
│   ├── Dockerfile.32b_sft_v2 / _sft_v3     • SFT iterations
│   ├── Dockerfile.32b_dpo / _dpo_v2        • DPO iterations
│   ├── Dockerfile.32b_oracle               • Oracle SFT
│   ├── Dockerfile.32b_refboost{,_v3,_v4}   • reranker-heavy variants
│   ├── Dockerfile.32b_bestofn / _thinking / _twostage / _hybrid
│   ├── Dockerfile.35b / .35b.test          • Qwen3.6-35B-A3B
│   ├── Dockerfile.fp8                      • FP8 quantised
│   ├── Dockerfile.dpo                      • 14B DPO
│   ├── Dockerfile.oracle_v4 / _v4_p2       • Oracle v4 multi-stage
│   ├── Dockerfile.sft_v5                   • 14B SFT-v5 (BF16)
│   ├── docker-compose.yml                  • local smoke-test harness
│   └── dockerignore/                       per-image build-context filters
│       ├── dockerignore                        (rename of .dockerignore — GitHub
│       └── dockerignore.<variant>               hides dotfiles in the tree view)
│
├── scripts/                            ← orchestration
│   ├── build/                              rootless podman build+push (Lanta)
│   │   └── lanta_build_push_*.sh           • one per image variant
│   └── slurm/                              SLURM batch wrappers
│       ├── script_train_*.sh               • LoRA training jobs
│       ├── script_eval_*.sh                • eval jobs
│       ├── script_quantize_*.sh            • AWQ jobs
│       ├── script_dpo*.sh                  • DPO jobs
│       ├── script_build_push.sh            • image build job
│       ├── script_podman_build.sh          • podman variant
│       ├── script_moe_eval.sh              • 35B MoE eval
│       ├── start_*.sh                      • chained-job launchers
│       ├── swait.sh                        • squeue watcher utility
│       ├── download_qwen35_9b.sh           • model fetch helper
│       └── rebuild_oracle_after_35b.sh     • cascade rebuild script
│
└── data/
    └── evaluate_sample/                ← official competition scorer
        ├── eval.py                         • bge-m3 + ROUGE + IoU implementation
        ├── requirements.txt
        └── submission.csv                  • format example
```

**Legend.**
- ★ = file you actually want to read first.
- Large artefacts (model checkpoints, LoRA outputs, raw build logs, the > 800-MB train-set zip) are **not** committed — they live on the Lanta Lustre scratch and in the AI-Singapore registry.
- The hardcoded paths inside the bash scripts assume the original flat layout on Lanta; this repo's directory structure is reorganised for *readability*. Adjust working directories if you re-run the scripts from this layout.

---

## How to Build & Submit

The repo assumes you have:
- A Lanta HPC account (or any single-H100 box) with **rootless podman**.
- Read access to the local `bge-m3/` and `bge-reranker-v2-m3/` checkpoints.
- A merged + AWQ-quantised model (see `quantize_14b_v5_awq.py`).

### 1. Quantise the LoRA-merged model

```bash
sbatch scripts/slurm/script_quantize_14b_v5.sh
# → produces ./Qwen3-14B-SFT-v5-AWQ/ (~9.3 GB)
```

### 2. Build the inference image (rootless podman, Lanta-tuned)

```bash
# Storage on /dev/shm (126 GB tmpfs) — /tmp is far too small for 40 GB images.
nohup bash scripts/build/lanta_build_push_14b_awq.sh > build_14b_awq.log 2>&1 &
tail -f build_14b_awq.log
```

The script:
1. Cleans any leftover `podman-root-*` directories on `/dev/shm`.
2. Builds `textsum-14b-awq:v1` from `docker/Dockerfile.14b_awq`.
3. Logs in to `registry.ai.in.th` and pushes (with up to 5 retries) the tag
   `registry.ai.in.th/2026-textsum/48f0b4ab/watin-promfiy.tme5:AI-Benchmark-Programs-2026-14b-awq-v1`.

### 3. Smoke-test locally before submission

```bash
docker compose -f docker/docker-compose.yml up
# mounts ./test_data and ./result
```

### 4. Submit

Submit the pushed tag via the competition portal. **One submission per build** — the daily quota is two.

---

## Hardware Constraints & Tricks

- **Why AWQ INT4, not FP8 / bitsandbytes / NF4?**
  - bitsandbytes needs internet on the compute node — Lanta nodes are air-gapped.
  - FP8 (35B MoE) lost too much RougeL on the abstractive head.
  - AWQ ships precompiled kernels and slots cleanly into vLLM 0.8.5.

- **Why `/dev/shm` for podman roots?**
  - Lustre doesn't support the xattrs overlayfs needs.
  - `/tmp` is < 10 GB → blows up on the 40-GB AWQ image.
  - `/dev/shm` is a 126 GB tmpfs; clean it between builds: `podman unshare rm -rf /dev/shm/podman-root-*-$USER`.

- **vLLM tuning that mattered.**
  - `max_num_seqs = 16` (14B) / `4` (32B) — anything higher OOM'd.
  - `max_model_len = 6144` — covers the longest train doc + question + slack.
  - `MAX_NEW_TOKENS = 120` — empirically the 99-th percentile of gold answer length.
  - `VLLM_USE_V1=0` — the v1 scheduler was unstable on long single-doc batches in 0.8.5.

- **Where time goes in a 30-minute budget.**
  - vLLM startup + weights load: ~70 s
  - BGE-M3 retrieval (1239 queries × top-15): ~3 min
  - Reranker (1239 × 15): ~5 min
  - LLM generation: ~17 min (14B) / ~26 min (32B)
  - JSON parse + write: < 5 s
  - The eval-server timeout pushed 32B over the edge in late May — hence the 14B pivot.

---

## Lessons Learned

1. **Read the data before tuning the model.** EDA caught the abstractive-vs-extractive mismatch and the 4 % JSON wrap bug that *no amount* of fine-tuning would have fixed. A 1-day EDA paid off more than a week of LoRA runs.
2. **Don't trust the eval server's wall-clock.** A passing build can time out a week later. Always keep one rung lower in model size as a safety net (this saved us on day-of-deadline).
3. **Per-image `.dockerignore` is mandatory.** Forgetting one excluded a 200 GB checkpoint dir into the build context and OOM'd `/dev/shm`. The `.dockerignore.<variant>` pattern + `--ignorefile` flag made parallel image builds safe.
4. **Cluster-expand refs, don't over-train.** A 1-line constant tweak (`NEIGHBOR_THRESHOLD 0.30 → 0.20`) caught 28 % of the IoU=0 failures — more than any reranker change.
5. **Have a robust output parser before fancy decoding.** 4.1 % truncated-JSON outputs were costing 0.257 RougeL each — the parser fix recovered ~0.01 to ~0.02 final-score across the whole eval set with zero model-side work.

---

## Acknowledgements

- 2026 TextSum organisers (Thai Parliament meeting-minutes dataset).
- BAAI for [BGE-M3](https://huggingface.co/BAAI/bge-m3) and [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3).
- Qwen team for the Qwen3 family.
- vLLM, AutoAWQ, PEFT, TRL — open-source plumbing that made this run.
- ThaiSC / Lanta HPC for the H100 hours.
