# 2026-TextSum — Project Status

## เป้าหมาย
แข่งขัน Query-based Abstractive Summarization จากบันทึกการประชุมรัฐสภา  
Score = `0.45×SS + 0.35×RougeL + 0.20×IoU`

---

## สิ่งที่ทำแล้ว

### Pipeline
- **Retriever**: BGE-M3 (local `./bge-m3`) — dense retrieval TOP_K=7 paragraphs
- **Generator**: RAG-style → build prompt → LLM → parse JSON `{abstractive, refs}`
- **Eval**: ลอก logic จาก `evaluate_sample/eval.py` ตรงๆ → `eval_train.py`

### Model iterations
| รอบ | Model | Mode | หมายเหตุ |
|-----|-------|------|---------|
| 1 | Qwen3-14B | BF16, 2-GPU | baseline, 50 test queries ✅ |
| 2 | Qwen3.6-35B-A3B | INT4 NF4 | ล้มเหลว — bitsandbytes ติดตั้งไม่ได้บน compute node (no internet) |
| 3 | Qwen3.6-35B-A3B | BF16 4-GPU (Lanta) / INT4 (submit) | pandas ไม่อยู่ใน env |
| 4 | **Qwen3.6-35B-A3B-FP8** | BF16 4-GPU (Lanta) / FP8 (submit) | **กำลังรันอยู่ — Job 5769946** |

### Packages ที่ install เพิ่มใน env
```
pandas, bitsandbytes, rouge-score, pythainlp, sentence-transformers
```
(ติดตั้งจาก login node ที่มี internet — compute node ไม่มี internet)

### ไฟล์หลัก
| ไฟล์ | หน้าที่ |
|------|---------|
| `inference_moe.py` | inference หลัก — FP8 single GPU (submit) / BF16 multi-GPU (Lanta test) |
| `inference_moe_train.py` | ชี้ไปที่ train_set.json |
| `eval_train.py` | eval เทียบกับ ground truth (ลอกจาก evaluate_sample/eval.py) |
| `script_moe_eval.sh` | SLURM: inference train → eval |
| `Dockerfile` | image สำหรับ submit — ใช้ FP8 model, install bitsandbytes |
| `script_build_push.sh` | SLURM: podman build + push |

---

## กำลังจะทำต่อ

### 1. ดู eval score จาก Job 5769946 (train set)
```bash
tail -30 moe_eval-5769946.out
```
ได้ค่า RougeL, SS-score, IoU, Final Score บน 1239 queries

### 2. ทดสอบ FP8 mode บน submission จริง
ตอนนี้ Lanta job ใช้ BF16 4-GPU เพื่อทดสอบ quality  
รอบหน้า: รัน `inference_moe.py` โดยไม่ตั้ง `LANTA_MULTIGPU=1` เพื่อจำลอง H100 submission (FP8 single-GPU)

### 3. Build & Push Docker image
```bash
# login node
podman login registry.ai.in.th
sbatch script_build_push.sh    # build (ใช้เวลานานเพราะ 35GB model)
podman push 'registry.ai.in.th/2026-textsum/48f0b4ab/watin promfiy.tme5:v1'
podman logout registry.ai.in.th
```

### 4. Optimize ถ้าคะแนนยังไม่ดีพอ
- เพิ่ม TOP_K (ตอนนี้ 7) — อาจช่วย IoU
- ปรับ prompt ให้ explicit มากขึ้นเรื่อง refs format
- ลอง Qwen3-32B BF16 เป็น fallback ถ้า FP8 มีปัญหา

---

## Submission Info
| | |
|---|---|
| Registry | `registry.ai.in.th` |
| Image path | `registry.ai.in.th/2026-textsum/48f0b4ab/watin promfiy.tme5:<tag>` |
| Container tool | `podman` (ใช้แทน docker บน Lanta) |
| Limit | 2 ครั้ง/วัน/ทีม |
| Inference time | 30 นาที/submission |
| GPU | H100 40GB |

## Memory layout บน H100 40GB (FP8 mode)
```
[BGE-M3 4.3GB] → retrieve ทุก query → del + empty_cache()
[FP8 LLM 35GB] → generate ทั้งหมด batch_size=4
Total peak: ~36GB / 40GB ✅
```
