# 2026 TextSum — Final Status (Post-Deadline)

**Deadline:** 2026-06-05
**Best score achieved:** **0.678** — `Qwen3-32B-SFT-v1-AWQ` (32b-thinking-v1)
**Baseline to beat:** 0.58 ✅

---

## Submission Timeline (Final 2 Days)

| Date | Tag | Model | Score | Note |
|------|-----|-------|-------|------|
| 2026-06-03 | `32b-thinking-v1` | Qwen3-32B-SFT-v1-AWQ + thinking | **0.678** | Best run |
| 2026-06-04 | `14b-awq-v1` (refboost v5) | Qwen3-14B-SFT-v5-AWQ + reranker + verify_refs | **Timeout** | 5th consecutive timeout from cumulative reranker overhead |
| 2026-06-04 | `14b-dense-v2` | Qwen3-14B-SFT-v5-AWQ, no reranker, abstractive prompt | 0.593 | RougeL=0.466, SS=0.800, IoU=0.352. Worse than baseline — prompt deviation hurt SS+RougeL, dropping reranker collapsed IoU |
| 2026-06-04 | `32b-thinking-v2` | Qwen3-32B-SFT-v1-AWQ + thinking + parser fix | **Exit Status 1** | Unresolved crash; admin never returned a traceback for our tag specifically |

---

## Known Issues / Unresolved

### Mystery #1 — `32b-thinking-v2` Exit Status 1

**Symptom:** Container ran but Python crashed; portal returned only `Error Exit StatusCode 1` (no stdout/stderr surfaced before deadline).

**What we verified:**
- Local image `localhost/textsum-32b-thinking:v2` contains the **correct** `run.py` (def main(), no `BitsAndBytesConfig` — verified via `podman run --entrypoint=cat`)
- Push succeeded twice (initial + retry after "no data" error from registry)
- Dockerfile/entrypoint/model/deps **identical** to thinking-v1 (which scored 0.678)
- Diff v1→v2 is purely additive (parser robustness) + removal (verify_refs)
- Unit-tested parser + expand_contiguous against edge cases — no crash

**Error traces shared by admin chat (none were ours):**
- `BitsAndBytesConfig` not defined — wrong script (someone else's old image)
- `decoder prompt 24625 > max_model_len 21000` — not us (we use 8192)
- `gemma4_unified` unrecognized — not us (we use Qwen3)

**Theories (ranked, untested):**
1. Edge case in eval test data triggered a path v1 never hit (parser strategy 3 with empty/binary text)
2. Registry served stale image despite successful push (digest mismatch)
3. `del retriever` + `torch.cuda.empty_cache()` interaction with vLLM init
4. `/result/` mount or `/benchmark_lib/progress` subprocess interaction

### Mystery #2 — 5 consecutive timeouts (May 28 – Jun 4)

`refboost v1/v2/v3/v4` + `14b-awq-v1` all timed out at 30 min. Common factor: **reranker + `verify_refs`** (2 cross-encoder calls per query × ~50 queries × ~3s/call ≈ 5 min overhead alone).

Removing `verify_refs` was the v2-thinking fix that never got verified.

---

## Lessons Learned

### What worked
- **AWQ 4-bit on 32B** fits H100 40GB with room for context (max_model_len=8192)
- **SFT-aligned prompts** outperformed EDA-derived "improvements" by 0.05+ on SS/RougeL
- **Reranker (bge-reranker-v2-m3)** was load-bearing for IoU — removing it dropped IoU by 0.17
- **Hybrid retrieval** (dense + sparse + BM25) > dense-only

### What didn't work
- **Changing prompts based on EDA** — drift from SFT distribution hurts all metrics
- **NEIGHBOR ±2 expansion** — over-predicted refs, IoU dropped
- **verify_refs with 2 cross-encoder calls** — too expensive, caused timeouts
- **Local environment testing** — env/ had torch 2.6.0+cu124 vs ~/.local torch 2.12.0+cpu, fundamental dependency hell

### Operational
- **Podman build on Lanta requires PODMAN_ROOT on Lustre via short symlink** (`/tmp/pr`) — `/tmp` too small for >40GB images, `runroot` path has <50 char hard limit
- **Registry pushes can be flaky** — "no data" error required retry with re-login
- **2 submissions/day is brutal with 30-min eval timeout** — debug iterations cost a full day each time

---

## Final Container Tags on Registry

```
registry.ai.in.th/2026-textsum/48f0b4ab/watin-promfiy.tme5:
  AI-Benchmark-Programs-2026-32b-thinking-v1    ← best (0.678)
  AI-Benchmark-Programs-2026-32b-thinking-v2    ← crashed (Exit 1)
  AI-Benchmark-Programs-2026-14b-dense-v2       ← 0.593
  AI-Benchmark-Programs-2026-14b-awq-v1         ← timeout
```

---

## What's in This Repo

- `inference/run_vllm_32b_thinking_v2.py` — the unresolved-crash script (preserved for post-mortem)
- `inference/run_vllm_32b_thinking.py` — best-scoring script (0.678)
- `inference/run_vllm_14b_dense_v2.py` — dense-only abstractive attempt (0.593)
- `docker/Dockerfile.32b_thinking_v2`, `docker/Dockerfile.14b_dense` — matching build files
- `scripts/build/lanta_build_push_*.sh` — SLURM build+push helpers
- `docs/RECAP.md` — earlier-period recap (pre-32B AWQ)
- `docs/STATUS.md` — earlier-period status
