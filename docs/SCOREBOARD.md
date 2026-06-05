# Submission Scoreboard

All known submissions to the **2026 TextSum** eval portal.
Score formula: `0.45·SS + 0.35·RougeL + 0.20·IoU` | Baseline to beat: **0.58**.

---

## 🏆 Final Standing

> **Best: 0.678** — `32b-thinking-v1` (Qwen3-32B-SFT-v1-AWQ + thinking mode), submitted 2026-06-03.

---

## All Submissions

| # | Date | Tag / Script | Model | Score | SS | RougeL | IoU | Notes |
|---|------|--------------|-------|-------|----|--------|-----|-------|
| 1 | early-May | v4 | Qwen3-14B (BF16) | **0.182** | 0.364 | 0.051 | ~0 | Bug: prompt hard-coded `"refs":["P1"]` → IoU=0 |
| 2 | mid-May | v5 / v6 | Qwen3-14B (BF16) | **0.644** | 0.838 | 0.476 | 0.501 | IoU bug fix |
| 3 | 2026-05-22 | SFT-v2 + rerank | Qwen3-14B-SFT-v2 | **0.649** | 0.817 | 0.453 | 0.616 | Reranker added, was previous best |
| 4 | 2026-05-23 | SFT-v3 | Qwen3-14B-SFT-v3 | **0.644** | 0.820 | 0.440 | 0.606 | Similar pipeline, ~same score |
| 5 | ~2026-05-24 | 35B v15 | Qwen3.6-35B-A3B-FP8 | **0.634** | 0.805 | 0.410 | 0.638 | Dense-only (no rerank), 35B lost to 14B+rerank |
| 6 | 2026-05-28 | refboost v1 | Qwen3-14B-SFT-v5-AWQ | **Timeout** | — | — | — | Reranker + verify_refs too slow |
| 7 | 2026-05-29 | refboost v2 | Qwen3-14B-SFT-v5-AWQ | **Timeout** | — | — | — | |
| 8 | 2026-05-30 | refboost v3 | Qwen3-14B-SFT-v5-AWQ | **Timeout** | — | — | — | |
| 9 | 2026-05-31 | refboost v4 | Qwen3-14B-SFT-v5-AWQ | **Timeout** | — | — | — | |
| 10 | 2026-06-02 | 32b-thinking-v1 | Qwen3-32B-SFT-v1-AWQ + thinking | **🏆 0.678** | — | — | — | **Best run** (full SS/RougeL/IoU not preserved) |
| 11 | 2026-06-04 | 14b-awq-v1 | Qwen3-14B-SFT-v5-AWQ + verify_refs | **Timeout** | — | — | — | 5th consecutive timeout |
| 12 | 2026-06-04 | 14b-dense-v2 (EDA Strategy A) | Qwen3-14B-SFT-v5-AWQ, no reranker, abstractive prompt | **0.593** | 0.800 | 0.466 | 0.352 | EDA-derived; dropped reranker → IoU collapsed −0.17 |
| 13 | 2026-06-04 | 32b-thinking-v2 | Qwen3-32B-SFT-v1-AWQ + thinking + parser fix | **Exit 1** | — | — | — | Mystery crash; admin never returned traceback before deadline |

---

## Score Progression

```
0.182 ─────┐
           │ (IoU bug: hardcoded P1)
0.644 ─────│──────────────┐
           │              │ (v5/v6 IoU fix → +0.46)
0.649 ─────│──────────────│─┐
           │              │ │ (SFT-v2 + reranker → previous best)
0.634 ─────│──────────────│─│┐
0.644 ─────│──────────────│─││ (35B dense, SFT-v3)
           │              │ ││
           │              │ ││ ── many timeouts (refboost v1-v4) ──
           │              │ ││
0.593 ─────│──────────────│─││────┐
           │              │ ││    │ (14b-dense-v2: prompt drift hurt)
0.678 ─────│──────────────│─││────│──── 🏆 FINAL BEST
           │                          ↑
           │                          32b-thinking-v1 (06-03)
           │
   Exit 1 ─┴── 32b-thinking-v2 (unresolved, deadline closed)
```

---

## What Drove the Wins

| Change | Impact | Note |
|---|---|---|
| IoU bug fix (don't hardcode "P1") | +0.46 | Single biggest gain ever |
| Add `bge-reranker-v2-m3` after BGE-M3 retrieval | +0.07 IoU | SFT-v2: 0.616 IoU vs ~0.50 baseline |
| Switch 14B → 32B-SFT-v1-AWQ | +0.03 vs 14B baseline | Bigger model + AWQ fits 40GB |
| Enable thinking mode on 32B-AWQ | +0.03 above 0.649 | Final best — 0.678 |

## What Hurt Us

| Change | Impact | Lesson |
|---|---|---|
| Prompt drift from SFT distribution (14b-dense-v2) | −0.05 SS, −0.05 RougeL | Don't optimize prompt away from training distribution |
| Drop reranker (14b-dense-v2) | −0.17 IoU | Reranker was load-bearing for ref selection, not just multi-ref |
| `verify_refs` cross-encoder × 2 calls/query | Caused 5 timeouts | Cumulative ~5min overhead exceeded 30min budget |
| Local-env timing tests | Burned ~2h | env/ vs ~/.local torch clash; should have benchmarked in container |

---

See [`FINAL_STATUS.md`](FINAL_STATUS.md) for the unresolved Exit-1 post-mortem.
