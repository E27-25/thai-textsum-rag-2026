# 2026 TextSum Competition — Recap

## โจทย์

Thai meeting-minutes summarization + evidence retrieval

**Score formula:** `0.45×SS + 0.35×RougeL + 0.20×IoU`

| Component | คืออะไร |
|-----------|---------|
| SS (Semantic Similarity) | bge-m3 embedding similarity ระหว่าง output กับ reference |
| RougeL | Longest common subsequence overlap ของ summary |
| IoU | Intersection-over-Union ของ paragraph refs ที่เลือก |

**ข้อจำกัดการ submit:**
- ภาษาไทย เท่านั้น
- H100 PCIe 40GB, CUDA 13
- Timeout 30 นาที/dataset
- Image size < 60GB
- 2 submissions/วัน
- Baseline score: 0.58

---

## Model Performance (ล่าสุด)

| Submission | Model | Score | SS | RougeL | IoU | หมายเหตุ |
|------------|-------|-------|----|--------|-----|---------|
| v4 (เก่า) | Qwen3-14B | 0.182 | - | - | ~0 | Bug: IoU=0 เพราะ hardcode "P1" ใน refs |
| v5/v6 | Qwen3-14B | 0.644 | 0.838 | 0.476 | 0.501 | IoU fix |
| **SFT-v2+rerank** | **Qwen3-14B-SFT-v2** | **0.649** | **0.817** | **0.453** | **0.616** | **Previous best** |
| SFT-v3 | Qwen3-14B-SFT-v3 | 0.644 | 0.820 | 0.440 | 0.606 | ส่งแล้ว 23 May |
| 35B v15 | Qwen3.6-35B-A3B-FP8 | 0.634 | 0.805 | 0.410 | 0.638 | dense-only (no rerank), แพ้ 14B |
| Oracle SFT-v4 | Qwen3-14B-Oracle-v4 | pending | - | - | - | train เสร็จ, กำลัง build |
| DPO v1 | Qwen3-14B-DPO-v1 | pending | - | - | - | กำลัง train |

---

## Architecture (ปัจจุบัน)

```
Input query
    │
    ▼
BGE-M3 dense embed → top-20 candidates (cosine)
    │
    ▼
ColBERT late-interaction rerank → top-7 paragraphs
    │
    ▼
Context ordered (most relevant LAST — recency bias)
    │
    ▼
Qwen3 LLM (vllm) + system prompt
    │
    ▼
JSON: { "abstractive": "...", "refs": ["P3", "P7"] }
    │
    ▼
expand_contiguous() — เติม gap 1 para ระหว่าง refs
```

---

## Training Pipeline

### SFT-v3 (base ของ DPO)
- Base: `Qwen3-14B`
- LoRA r=64, alpha=128, 3 epochs
- Context: ColBERT-retrieved top-7 paragraphs (same as inference)
- Output: `Qwen3-14B-SFT-v3`

### Oracle SFT-v4 ✅ เสร็จแล้ว
- Base: `Qwen3-14B`
- LoRA r=64, alpha=128, 3 epochs
- Context: ground-truth refs + 3 hard negatives (dense-retrieved non-refs)
- ไม่มี retrieval noise → model เรียนรู้ abstractive quality โดยตรง
- Script: `train_sft_v4_oracle.py`, `script_train_v4.sh`
- SLURM job 5785250 → เสร็จ 22 May 19:34
- Output: `Qwen3-14B-Oracle-v4`

### DPO v1 🔄 กำลัง train (79%)
- Base: `Qwen3-14B-SFT-v3`
- Generate N=6 answers per query (temp=0.7, top_p=0.9)
- Score by RougeL vs ground-truth abstractive
- Pair: chosen=best RougeL, rejected=worst (diff threshold 0.08)
- DPO beta=0.1, 1 epoch, LoRA r=32
- Script: `train_dpo.py`, `script_dpo.sh`
- SLURM job 5785251 → 980/1239 generating pairs
- Output: `Qwen3-14B-DPO-v1`

---

## Inference Improvements (ทุก version นับจาก SFT-v2)

### ColBERT Reranking
```python
TOP_K_RETRIEVE = 20   # dense cosine first pass
TOP_K_FINAL    = 7    # colbert late-interaction rerank
```

### Contiguous Ref Expansion
ถ้า model เลือก P51 + P53 แต่ไม่เลือก P52 → เติม P52 อัตโนมัติ (83.4% multi-ref adjacent)
```python
def expand_contiguous(refs, valid_ids):
    # fills 1-paragraph gaps
```

### Context Ordering
reversed retrieved list → most relevant paragraph อยู่ท้ายสุด (LLM recency bias)

### System Prompt
```python
SYSTEM_PROMPT = (
    "คุณเป็นผู้ช่วยตอบคำถามจากบันทึกการประชุมภาษาไทย "
    "ตอบให้ครบถ้วน ชัดเจน และอ้างอิงเฉพาะข้อมูลจากย่อหน้าที่ให้มาเท่านั้น"
)
```

---

## 35B Issues & Fixes

| Error | สาเหตุ | วิธีแก้ |
|-------|--------|--------|
| `open /run/nvidia-persistenced/socket` | HW issue บน node นั้น (admin investigating) | รอ admin แก้ |
| `nvcc: not found` | pip `nvidia-cuda-nvcc-cu12` binary ไม่ execute ได้ | เปลี่ยน base → `nvidia/cuda:12.4.1-devel-ubuntu22.04` |
| `ninja build failed` (FlashInfer JIT) | nvcc CUDA 12.4 ≠ torch `cu130` → header mismatch | เปลี่ยน base → `nvidia/cuda:13.0.0-devel-ubuntu22.04` + `VLLM_ATTENTION_BACKEND=FLASH_ATTN` |

**v15 (ล่าสุด, ส่งแล้ว):**
- Base: `nvidia/cuda:13.0.0-devel-ubuntu22.04` (nvcc 13.0.48 ตรงกับ torch cu130)
- `VLLM_ATTENTION_BACKEND=FLASH_ATTN` ป้องกัน FlashInfer JIT
- Speed: ไม่มี ColBERT rerank, dense-only top-7, max_tokens 200, enforce_eager=False
- Result: **0.634** (SS=0.805, RougeL=0.410, IoU=0.638) — แพ้ 14B SFT-v2+rerank
- **วิเคราะห์:** RougeL ต่ำ (0.410) เพราะไม่มี ColBERT rerank → retrieval quality แย่ลง; IoU ดีกว่า (0.638) เพราะ model ใหญ่กว่า
- Registry tag: `:AI-Benchmark-Programs-2026-35b`

---

## Registry Tags

| Image | Tag |
|-------|-----|
| 14B (current) | `registry.ai.in.th/.../watin-promfiy.tme5:AI-Benchmark-Programs-2026` |
| 35B v14 | `registry.ai.in.th/.../watin-promfiy.tme5:AI-Benchmark-Programs-2026-35b` |

---

## Build Issues & Solutions

| ปัญหา | สาเหตุ | วิธีแก้ |
|-------|--------|--------|
| `no space left on device` | `/tmp` ผันผวน | PODMAN_ROOT → `/dev/shm` (126GB) |
| Lustre overlay fail | Lustre ไม่รองรับ xattr | ใช้ `/dev/shm` (tmpfs) |
| Build context 250GB | ไม่มี `.dockerignore` | เพิ่ม `.dockerignore` แยก 14B/35B |
| Parallel builds ทับกัน | PODMAN_ROOT เดียวกัน | 14B ใช้ `podman-root2`, 35B ใช้ `podman-root` |
| `/dev/shm` เต็ม (parallel) | 14B storage ค้างหลัง push (52GB) | ล้าง `podman-root2` ก่อน start 35B |

### Build Commands

```bash
# 14B SFT-v3
nohup bash lanta_build_push_14b_sft_v3.sh > build_14b_sftv3.log 2>&1 &

# 35B v14
nohup bash lanta_build_push_35b.sh > build_35b_v14.log 2>&1 &

# ดู log
tail -f build_14b_sftv3.log
tail -f build_35b_v14.log
```

---

## SLURM Jobs

| Job ID | Name | Status | หมายเหตุ |
|--------|------|--------|---------|
| 5785250 | train_sft_v4_oracle | ✅ DONE | เสร็จ 22 May 19:34 |
| 5785251 | train_dpo_v1 | 🔄 RUNNING | 980/1239 generating pairs |

```bash
squeue -u $USER   # check status
tail -f train_sft_v4_oracle-5785250.out
tail -f train_dpo_v1-5785251.out
```

---

## Files สำคัญ

```
Arther/
├── Dockerfile                    # 14B image (SFT-v3)
├── Dockerfile.35b                # 35B-FP8 image (v14, CUDA 13.0 devel)
├── Dockerfile.35b.test           # test build ไม่มีโมเดล
├── run_vllm.py                   # 14B inference (ColBERT + expand_contiguous)
├── run_vllm_35b.py               # 35B inference
├── entrypoint.sh                 # 14B CUDA path
├── entrypoint_35b.sh             # 35B CUDA 13.0 path (dynamic PYSITE)
├── requirements.txt              # 14B deps
├── .dockerignore                 # exclude 14B build
├── .dockerignore.35b             # exclude 35B build
├── lanta_build_push.sh           # build+push 14B (original)
├── lanta_build_push_14b_sft_v3.sh # build+push 14B SFT-v3 (podman-root2)
├── lanta_build_push_35b.sh       # build+push 35B (podman-root, tag -35b)
├── train_sft_v4_oracle.py        # Oracle SFT-v4 training
├── train_dpo.py                  # DPO pair generation + training
├── merge_lora_v4.py              # merge Oracle SFT-v4 LoRA
├── merge_lora_dpo.py             # merge DPO LoRA
├── script_train_v4.sh            # SLURM: Oracle SFT-v4 (120h)
├── script_dpo.sh                 # SLURM: DPO (120h)
├── Qwen3-14B/                    # base model 28GB
├── Qwen3-14B-SFT-v3/             # SFT-v3 merged 28GB
├── Qwen3-14B-Oracle-v4/          # Oracle SFT-v4 merged ✅ ready
├── Qwen3.6-35B-A3B-FP8/          # 35B FP8 model 35GB
└── bge-m3/                       # retrieval model 2GB
```

---

## Next Steps

1. **ทันที**: build + push `Qwen3-14B-Oracle-v4` → submit
2. **~1.5h**: DPO train เสร็จ → merge → build + push `Qwen3-14B-DPO-v1` → submit
3. **รอผล**: 35B v14 benchmark result
