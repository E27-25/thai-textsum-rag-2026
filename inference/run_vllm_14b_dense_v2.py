#!/usr/bin/env python3
"""14B AWQ — dense-only (no reranker) + EDA-driven prompt/tuning.

Why this variant:
  • 5 timeouts in a row with reranker+verify_refs pipeline → cumulative cross-encoder
    overhead (load + per-query rerank + per-query verify×2) eats ~8-10 min of budget.
  • Dropping reranker brings the wall-clock back in line with the 0.678 baseline.

EDA findings applied:
  • Gold answers are 80% abstractive (LCS=0.18) — system prompt no longer says
    "ใช้ถ้อยคำโดยตรง"; encourages reformulation.
  • Only 13.4% of answers restate the question — no restate cue in prompt.
  • 95% of predictions under-predict refs vs gold (28% multi-ref) — prompt now
    explicitly invites multiple refs; NEIGHBOR expansion widened to ±2.
  • 89% of answers < 20 words (median 7) — MAX_NEW_TOKENS=80 is plenty.
  • para_id formats are mixed (P1, "(น...", bare digits) — robust regex.
  • Robust JSON parser (3-strategy fallback) ports over from v1, fixes the 4.1%
    truncated/wrapped outputs.
"""
import time; time.sleep(10)  # let Loki collect startup logs
import sys, os
print("[BOOT] Python starting", flush=True)
print(f"[BOOT] VLLM_USE_V1={os.environ.get('VLLM_USE_V1','<unset>')}", flush=True)

import glob
import json
import re
import csv
import subprocess

try:
    import torch
    print(f"[BOOT] torch {torch.__version__} OK | CUDA={torch.cuda.is_available()}", flush=True)
except Exception as _e:
    print(f"[BOOT] torch FAILED: {_e}", flush=True); sys.exit(1)

try:
    from transformers import AutoTokenizer
    from FlagEmbedding import BGEM3FlagModel
    print("[BOOT] transformers + FlagEmbedding OK", flush=True)
except Exception as _e:
    print(f"[BOOT] transformers/FlagEmbedding FAILED: {_e}", flush=True); sys.exit(1)

try:
    from vllm import LLM, SamplingParams
    print("[BOOT] vllm OK", flush=True)
except Exception as _e:
    print(f"[BOOT] vllm FAILED: {_e}", flush=True); sys.exit(1)

from tqdm import tqdm

# ── Paths & knobs ─────────────────────────────────────────────────────────────
BGE_PATH    = "/model/bge-m3"
LLM_PATH    = os.environ.get("LLM_PATH", "/model/Qwen3-14B-SFT-v5-AWQ")
TEST_DIR    = "/model/test/"
RESULT_DIR  = "/result/"
PROGRESS_LIB = "/benchmark_lib/progress"
OUTPUT_CSV  = os.path.join(RESULT_DIR, "submission.csv")

TOP_K_FINAL    = 7           # dense top-7 → prompt context
MAX_NEW_TOKENS = 80          # EDA: 95% < 30 words ≈ ~70 tokens
GPU            = "cuda:0"
BGE_BATCH      = 128

# Neighbor expansion (replaces reranker-based verify_refs)
NEIGHBOR_RADIUS    = 2          # ±2 paragraphs — EDA: p95 multi-ref gap = 11
NEIGHBOR_DENSE_THR = 0.50       # min dense-cos similarity for a neighbor to count
DROP_REF_THR       = 0.30       # drop a model-predicted ref if dense score below this

# EDA-aligned prompt: abstractive + multi-ref encouragement, no question restate
SYSTEM_PROMPT = (
    "คุณเป็นผู้ช่วยตอบคำถามจากบันทึกการประชุมรัฐสภาภาษาไทยแบบ abstractive "
    "ให้สรุปคำตอบสั้น กระชับ 5-20 คำ โดยเรียบเรียงประโยคใหม่ (อย่าลอกประโยคจากบริบทตรงๆ ทั้งประโยค) "
    "หากคำตอบกระจายอยู่หลายย่อหน้า ให้ระบุ ID ของทุกย่อหน้าที่เกี่ยวข้อง อย่าใส่แค่ย่อหน้าเดียว"
)


# ── Robust para_id parsing ────────────────────────────────────────────────────
_PARA_NUM_RE = re.compile(r'(\d+)')
def _para_num(pid: str) -> int | None:
    m = _PARA_NUM_RE.search(str(pid))
    return int(m.group(1)) if m else None

def _para_prefix(pid: str) -> str:
    """Return everything before the trailing number (e.g. 'P' for 'P12', '(น' for '(น45')."""
    s = str(pid)
    m = re.search(r'\d+\s*$', s)
    return s[:m.start()] if m else s


# ── Retriever: dense-only ─────────────────────────────────────────────────────
class DenseRetriever:
    def __init__(self, model_path: str, device: str):
        self.device = device
        self.model = BGEM3FlagModel(model_path, use_fp16=(device != "cpu"), device=device)

    def encode_corpus(self, paragraphs: list[dict]) -> torch.Tensor:
        texts = [p["text"] for p in paragraphs]
        out = self.model.encode(
            texts, batch_size=BGE_BATCH,
            return_dense=True, return_sparse=False, return_colbert_vecs=False,
            max_length=512,
        )
        return torch.tensor(out["dense_vecs"])    # (N, dim)

    def encode_query(self, query: str) -> torch.Tensor:
        out = self.model.encode(
            [query], return_dense=True, return_sparse=False, return_colbert_vecs=False,
            max_length=512,
        )
        return torch.tensor(out["dense_vecs"][0])  # (dim,)


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_prompt(query: str, retrieved: list[dict], max_para_chars: int = 800) -> str:
    # Most relevant last → LLM recency bias helps recall
    ordered = list(reversed(retrieved))
    context = "\n".join(f"[{p['para_id']}] {p['text'][:max_para_chars]}" for p in ordered)
    ids = ", ".join(f'"{p["para_id"]}"' for p in ordered)
    return (
        f"ย่อหน้าจากบันทึกการประชุม:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"ตอบเป็น JSON เท่านั้น:\n"
        f'{{"abstractive": "สรุปคำตอบภาษาไทย 5-20 คำ", "refs": ["id1", "id2", ...]}}\n'
        f"ID ที่เลือกได้: {ids}\n"
        f"กฎ: refs ต้องระบุทุกย่อหน้าที่สนับสนุนคำตอบ (อาจมีมากกว่า 1)"
    )


# ── Ref post-processing: cheap, BGE-based ─────────────────────────────────────
def expand_contiguous(refs: list[str], valid_ids: set[str]) -> list[str]:
    """Fill single-paragraph gaps: refs={P51,P53} ∧ P52∈valid → add P52."""
    if len(refs) <= 1:
        return refs
    parsed = [(p, _para_prefix(p), _para_num(p)) for p in refs]
    parsed = [(p, pfx, n) for p, pfx, n in parsed if n is not None]
    if not parsed:
        return refs
    parsed.sort(key=lambda x: x[2])
    result = set(refs)
    for i in range(len(parsed) - 1):
        _, pfx1, n1 = parsed[i]
        _,  _,  n2 = parsed[i + 1]
        gap = n2 - n1
        if 2 <= gap <= 3:
            for step in range(1, gap):
                mid = f"{pfx1}{n1 + step}"
                if mid in valid_ids:
                    result.add(mid)
    return sorted(result, key=lambda x: _para_num(x) or 0)


def verify_and_expand(
    refs: list[str],
    valid_ids: set[str],
    para_dense_map: dict[str, dict],   # doc_id → {para_id → idx}
    dense_corpus: dict[str, torch.Tensor],   # doc_id → (N, dim) tensor
    q_dense: torch.Tensor,
    doc_id: str,
    fallback: str,
) -> list[str]:
    """Cheap reranker-free verification using pre-computed BGE dense scores.
    1) drop refs with dense score < DROP_REF_THR (likely hallucinations)
    2) add ±NEIGHBOR_RADIUS neighbors whose dense score >= NEIGHBOR_DENSE_THR
    3) contiguous-gap fill (P51,P53 → P52)
    """
    corpus = dense_corpus.get(doc_id)
    idx_map = para_dense_map.get(doc_id, {})
    if corpus is None or not idx_map:
        return refs or [fallback]

    sims = (corpus @ q_dense).tolist()    # (N,)

    # 1) drop hallucinated refs
    kept = []
    for r in refs:
        idx = idx_map.get(r)
        if idx is None:
            continue  # ref not in doc → drop
        if sims[idx] >= DROP_REF_THR:
            kept.append(r)
    if not kept:
        # nothing survived → keep highest-scoring of originals if any, else fallback
        in_doc = [(r, sims[idx_map[r]]) for r in refs if r in idx_map]
        if in_doc:
            in_doc.sort(key=lambda x: -x[1])
            kept = [in_doc[0][0]]
        else:
            kept = [fallback]

    # 2) neighbor expansion ±NEIGHBOR_RADIUS, gated by dense score
    out = set(kept)
    for r in list(kept):
        pfx, num = _para_prefix(r), _para_num(r)
        if num is None:
            continue
        for d in range(1, NEIGHBOR_RADIUS + 1):
            for nbr in (f"{pfx}{num - d}", f"{pfx}{num + d}"):
                if nbr in out or nbr not in valid_ids:
                    continue
                idx = idx_map.get(nbr)
                if idx is not None and sims[idx] >= NEIGHBOR_DENSE_THR:
                    out.add(nbr)

    # 3) fill contiguous gaps
    return expand_contiguous(sorted(out, key=lambda x: _para_num(x) or 0), valid_ids)


# ── Robust JSON parser (ported from 14b_awq v1) ───────────────────────────────
def parse_output(text: str, fallback_para: str, valid_para_ids: set[str] | None = None) -> tuple[str, list[str]]:
    # Strategy 1: try complete JSON
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            obj = json.loads(match.group())
            abstractive = str(obj.get("abstractive", "")).strip()
            refs = [str(r) for r in obj.get("refs", [])]
            if abstractive and refs:
                return abstractive, refs
    except Exception:
        pass

    # Strategy 2: truncated JSON — regex extract
    abstractive = ""
    refs: list[str] = []
    am = re.search(r'"abstractive"\s*:\s*"((?:[^"\\]|\\.)*)"?', text, re.DOTALL)
    if am:
        raw = am.group(1)
        abstractive = (raw.replace('\\n', '\n').replace('\\t', '\t')
                          .replace('\\"', '"').replace('\\\\', '\\').strip())
    rm = re.search(r'"refs"\s*:\s*\[([^\]]*)', text, re.DOTALL)
    if rm:
        for m in re.finditer(r'"([^"]+)"', rm.group(1)):
            refs.append(m.group(1))

    if abstractive:
        if not refs:
            refs = [fallback_para]
        return abstractive, refs

    # Strategy 3: plaintext cleanup
    cleaned = re.sub(r'^\s*\{?\s*"?abstractive"?\s*:?\s*"?', '', text).strip()
    cleaned = re.sub(r'"\s*,?\s*"?refs"?.*$', '', cleaned, flags=re.DOTALL).strip()
    return cleaned or text.strip(), [fallback_para]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"[ENV] LLM_PATH={LLM_PATH} | TEST_DIR={TEST_DIR} | RESULT_DIR={RESULT_DIR}", flush=True)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"[CUDA] {torch.cuda.get_device_name(0)} | {p.total_memory/1e9:.1f}GB", flush=True)

    print(f"[STEP 1] Scanning {TEST_DIR}", flush=True)
    json_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.json")))
    if not json_files:
        raise FileNotFoundError(f"No JSON file found in {TEST_DIR}")
    test_path = json_files[0]
    print(f"Test file: {test_path}", flush=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    with open(test_path, encoding="utf-8") as f:
        test_data = json.load(f)
    doc_map = {d["doc_id"]: d for d in test_data["docs"]}
    queries  = test_data["queries"]
    print(f"Loaded {len(queries)} queries across {len(doc_map)} docs", flush=True)

    print(f"[STEP 2] BGE-M3 on {GPU}", flush=True)
    retriever = DenseRetriever(BGE_PATH, GPU)

    dense_corpus: dict[str, torch.Tensor] = {}
    para_dense_map: dict[str, dict[str, int]] = {}
    para_text_map: dict[str, str] = {}
    para_list_map: dict[str, list[dict]] = {}
    for doc_id, doc in tqdm(doc_map.items(), desc="Encode corpora", dynamic_ncols=False):
        paras = doc["paragraphs"]
        para_list_map[doc_id] = paras
        dense_corpus[doc_id] = retriever.encode_corpus(paras)
        para_dense_map[doc_id] = {p["para_id"]: i for i, p in enumerate(paras)}
        for p in paras:
            para_text_map[p["para_id"]] = p["text"]

    print(f"Retrieve top-{TOP_K_FINAL} for {len(queries)} queries (dense only)", flush=True)
    retrieved_all: list[list[dict]] = []
    q_dense_all: list[torch.Tensor] = []
    for q in tqdm(queries, desc="Retrieve", dynamic_ncols=False):
        paras = para_list_map[q["doc_id"]]
        qd = retriever.encode_query(q["query"])
        q_dense_all.append(qd)
        sims = (dense_corpus[q["doc_id"]] @ qd).tolist()
        top_idx = sorted(range(len(paras)), key=lambda i: sims[i], reverse=True)[:TOP_K_FINAL]
        retrieved_all.append([paras[i] for i in top_idx])

    # Free BGE model (keep tensors for ref verification)
    del retriever
    torch.cuda.empty_cache()
    print(f"BGE freed | VRAM: {torch.cuda.memory_allocated(0)/1e9:.1f} GB", flush=True)

    print(f"[STEP 3] Build prompts", flush=True)
    tok = AutoTokenizer.from_pretrained(LLM_PATH)
    prompts = []
    for q, retrieved in zip(queries, retrieved_all):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_prompt(q["query"], retrieved)},
        ]
        prompts.append(tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        ))
    del tok

    print(f"[STEP 4] Loading {LLM_PATH} via vLLM", flush=True)
    llm = LLM(
        model=LLM_PATH,
        dtype="auto",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.92,
        max_model_len=6144,
        max_num_seqs=32,        # no reranker → more VRAM → bigger batch
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS)

    print(f"Generating {len(prompts)} answers", flush=True)
    outputs = llm.generate(prompts, sampling_params)

    print(f"[STEP 5] Parse + verify refs", flush=True)
    rows = []
    for q, retrieved, qd, output in zip(queries, retrieved_all, q_dense_all, outputs):
        gen_text = output.outputs[0].text
        fallback = retrieved[0]["para_id"] if retrieved else "P1"
        valid_ids = set(para_dense_map.get(q["doc_id"], {}).keys())
        abstractive, refs = parse_output(gen_text, fallback, valid_ids)
        refs = verify_and_expand(refs, valid_ids, para_dense_map, dense_corpus, qd, q["doc_id"], fallback)
        rows.append({"ID": q["ID"], "abstractive": abstractive, "refs": ",".join(refs)})

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "abstractive", "refs"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows → {OUTPUT_CSV}", flush=True)
    try:
        subprocess.run([PROGRESS_LIB, str(len(rows))], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"Progress update skipped: {e}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise
