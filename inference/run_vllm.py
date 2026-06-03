#!/usr/bin/env python3
import time; time.sleep(10)  # wait for Loki to collect startup logs
import sys, os
print("[BOOT] Python starting", flush=True)
print(f"[BOOT] VLLM_USE_V1={os.environ.get('VLLM_USE_V1','<unset>')}", flush=True)
print(f"[BOOT] LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH','<unset>')[:300]}", flush=True)

import glob
import json
import re
import csv
import subprocess

try:
    import torch
    import torch.nn.functional as F
    print(f"[BOOT] torch {torch.__version__} OK | CUDA={torch.cuda.is_available()} | devices={torch.cuda.device_count()}", flush=True)
    if torch.cuda.is_available():
        print(f"[BOOT] {torch.cuda.get_device_name(0)} {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB", flush=True)
except Exception as _e:
    print(f"[BOOT] torch FAILED: {_e}", flush=True)
    sys.exit(1)

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from FlagEmbedding import BGEM3FlagModel
    print("[BOOT] transformers + FlagEmbedding OK", flush=True)
except Exception as _e:
    print(f"[BOOT] transformers/FlagEmbedding FAILED: {_e}", flush=True)
    sys.exit(1)

try:
    from vllm import LLM, SamplingParams
    print("[BOOT] vllm OK", flush=True)
except Exception as _e:
    print(f"[BOOT] vllm FAILED: {_e}", flush=True)
    sys.exit(1)

from tqdm import tqdm
from collections import Counter as _Counter

try:
    from pythainlp.tokenize import word_tokenize as _wt
    _THAI_TOK = True
except Exception:
    _THAI_TOK = False

BGE_PATH      = "/model/bge-m3"
RERANKER_PATH = "/model/bge-reranker-v2-m3"
LLM_PATH      = os.environ.get("LLM_PATH", "/model/Qwen3-14B-SFT-v3")
TEST_DIR     = "/model/test/"
RESULT_DIR   = "/result/"
PROGRESS_LIB = "/benchmark_lib/progress"
OUTPUT_CSV   = os.path.join(RESULT_DIR, "submission.csv")
TOP_K_RETRIEVE = 30   # first-stage hybrid retrieval candidates
TOP_K_FINAL    = 10   # after cross-encoder rerank
SPARSE_WEIGHT  = 0.3  # α for hybrid: score = dense + α×sparse + β×bm25
BM25_WEIGHT    = 0.2  # β for BM25 component
MAX_NEW_TOKENS = 256
GPU          = "cuda:0"
BGE_BATCH    = 128

SYSTEM_PROMPT = (
    "คุณเป็นผู้ช่วยตอบคำถามจากบันทึกการประชุมภาษาไทย "
    "ตอบให้ครบถ้วนและชัดเจน โดยใช้ถ้อยคำและสำนวนจากย่อหน้าที่ให้มาโดยตรง "
    "อ้างอิงเฉพาะข้อมูลจากย่อหน้าที่ให้มาเท่านั้น "
    "หากคำตอบกระจายอยู่หลายย่อหน้าให้ระบุทุก ID ที่เกี่ยวข้อง"
)


class BM25:
    """Lightweight BM25 with pythainlp word tokenization."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1; self.b = b
        self.corpus_tfs: list[dict] = []
        self.df: dict = {}
        self.avgdl: float = 0.0
        self.N: int = 0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        # regex-based: Thai character runs + ASCII alphanumeric — fast enough for BM25
        return re.findall(r'[฀-๿]+|[A-Za-z0-9]+', text)

    def build(self, texts: list[str]) -> None:
        self.N = len(texts)
        tfs = []
        for t in texts:
            tokens = self._tokenize(t)
            tf = _Counter(tokens)
            tfs.append((len(tokens), tf))
        self.avgdl = sum(l for l, _ in tfs) / max(self.N, 1)
        self.df = {}
        for _, tf in tfs:
            for tok in tf:
                self.df[tok] = self.df.get(tok, 0) + 1
        self.corpus_tfs = tfs

    def score(self, query: str, idx: int) -> float:
        dl, tf = self.corpus_tfs[idx]
        norm = 1 - self.b + self.b * dl / max(self.avgdl, 1)
        s = 0.0
        for tok in self._tokenize(query):
            if tok not in tf:
                continue
            idf = max(0.0, ((self.N - self.df.get(tok, 0) + 0.5) /
                            (self.df.get(tok, 0) + 0.5)))
            tf_val = tf[tok] * (self.k1 + 1) / (tf[tok] + self.k1 * norm)
            s += idf * tf_val
        return s


class BGERetriever:
    def __init__(self, model_path: str, reranker_path: str, device: str):
        self.device = device
        use_fp16 = (device != "cpu")
        self.model = BGEM3FlagModel(model_path, use_fp16=use_fp16, device=device)
        dtype = torch.float16 if use_fp16 else torch.float32
        self.reranker_tok = AutoTokenizer.from_pretrained(reranker_path)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
            reranker_path, torch_dtype=dtype
        ).to(device)
        self.reranker_model.eval()

    def build_index(self, paragraphs: list[dict], doc_id: str = "") -> dict:
        """Returns dict with dense tensor + sparse list + BM25 for hybrid retrieval."""
        texts = [p["text"] for p in paragraphs]
        out = self.model.encode(
            texts, batch_size=BGE_BATCH,
            return_dense=True, return_sparse=True, return_colbert_vecs=False,
            max_length=512,
        )
        dense = torch.tensor(out["dense_vecs"])           # (N, dim) float32
        sparse = out["lexical_weights"]                    # list of {token_id: weight}
        bm25 = BM25()
        bm25.build(texts)
        return {"dense": dense, "sparse": sparse, "bm25": bm25}

    def _sparse_score(self, q_sparse: dict, p_sparse: dict) -> float:
        """Dot product over shared token ids."""
        score = 0.0
        for tid, w in q_sparse.items():
            if tid in p_sparse:
                score += w * p_sparse[tid]
        return score

    def retrieve(self, query: str, paragraphs: list[dict], index: dict, top_k: int) -> list[dict]:
        """Hybrid first-stage: dense + SPARSE_WEIGHT×sparse + BM25_WEIGHT×bm25."""
        q_out = self.model.encode(
            [query], return_dense=True, return_sparse=True, return_colbert_vecs=False,
            max_length=512,
        )
        q_dense  = torch.tensor(q_out["dense_vecs"][0])   # (dim,)
        q_sparse = q_out["lexical_weights"][0]

        n = len(paragraphs)
        dense_scores = (index["dense"] @ q_dense).tolist()
        bm25_raw     = [index["bm25"].score(query, i) for i in range(n)]
        bm25_max     = max(bm25_raw) if bm25_raw else 1.0
        bm25_scores  = [s / max(bm25_max, 1e-9) for s in bm25_raw]  # normalize to [0,1]

        hybrid_scores = [
            dense_scores[i]
            + SPARSE_WEIGHT * self._sparse_score(q_sparse, index["sparse"][i])
            + BM25_WEIGHT   * bm25_scores[i]
            for i in range(n)
        ]
        top_idx = sorted(range(n), key=lambda i: hybrid_scores[i], reverse=True)[:top_k]
        return [paragraphs[i] for i in top_idx]

    def rerank(self, query: str, passages: list[dict], top_k: int) -> list[dict]:
        """Cross-encoder rerank via bge-reranker-v2-m3 (batch)."""
        queries  = [query] * len(passages)
        texts    = [p["text"][:800] for p in passages]
        enc = self.reranker_tok(
            queries, texts, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            logits = self.reranker_model(**enc).logits.squeeze(-1)
        scores = torch.sigmoid(logits).tolist()
        if isinstance(scores, float):
            scores = [scores]
        ranked = sorted(range(len(passages)), key=lambda i: scores[i], reverse=True)
        return [passages[i] for i in ranked[:top_k]]


def build_prompt(query: str, retrieved: list[dict], max_para_chars: int = 800) -> str:
    # Most relevant last — LLM recency bias helps recall
    ordered = list(reversed(retrieved))
    context = "\n".join(f"[{p['para_id']}] {p['text'][:max_para_chars]}" for p in ordered)
    ids = ", ".join(f'"{p["para_id"]}"' for p in ordered)
    return (
        f"ย่อหน้าจากบันทึกการประชุม:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"ให้ตอบคำถามโดยสรุปคำตอบจากย่อหน้าข้างต้น และระบุ ID ย่อหน้าที่เกี่ยวข้อง\n"
        f"ID ที่เลือกได้: {ids}\n\n"
        f"ตอบเป็น JSON เท่านั้น:\n"
        f'{{"abstractive": "สรุปคำตอบภาษาไทย", "refs": ["id1", "id2"]}}'
    )


def expand_contiguous(refs: list[str], valid_ids: set[str]) -> list[str]:
    """Fill single-paragraph gaps: if P51+P53 both cited and P52 exists → add P52."""
    if len(refs) <= 1:
        return refs
    nums = []
    for r in refs:
        m = re.match(r'([A-Za-z]+)(\d+)$', r)
        if m:
            nums.append((int(m.group(2)), m.group(1), r))
    if not nums:
        return refs
    nums.sort()
    result_ids = set(refs)
    for i in range(len(nums) - 1):
        n1, pfx, _ = nums[i]
        n2, _, _   = nums[i + 1]
        if n2 - n1 == 2:
            mid = f"{pfx}{n1 + 1}"
            if mid in valid_ids:
                result_ids.add(mid)
        elif n2 - n1 == 3:
            for step in (1, 2):
                mid = f"{pfx}{n1 + step}"
                if mid in valid_ids:
                    result_ids.add(mid)
    return sorted(result_ids, key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)


def parse_output(text: str, fallback_para: str, valid_para_ids: set[str] | None = None) -> tuple[str, list[str]]:
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            obj = json.loads(match.group())
            abstractive = str(obj.get("abstractive", "")).strip()
            refs = [str(r) for r in obj.get("refs", [])]
            if abstractive and refs:
                if valid_para_ids:
                    refs = expand_contiguous(refs, valid_para_ids)
                return abstractive, refs
    except Exception:
        pass
    return text.strip(), [fallback_para]


def main():
    import platform as _pl
    print(f"[START] Python {sys.version} | {_pl.platform()}", flush=True)
    print(f"[ENV] LLM_PATH={LLM_PATH}", flush=True)
    print(f"[ENV] TEST_DIR={TEST_DIR} | RESULT_DIR={RESULT_DIR}", flush=True)
    print(f"[ENV] PROGRESS_LIB={PROGRESS_LIB}", flush=True)
    print(f"[CUDA] available={torch.cuda.is_available()} | device_count={torch.cuda.device_count()}", flush=True)
    if torch.cuda.is_available():
        print(f"[CUDA] {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB", flush=True)

    print(f"[STEP 1] Scanning {TEST_DIR} ...", flush=True)
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

    print(f"[STEP 2] Loading BGE-M3 + reranker on {GPU} ...", flush=True)
    retriever = BGERetriever(BGE_PATH, RERANKER_PATH, GPU)

    index_map: dict[str, tuple[list[dict], dict]] = {}
    for doc_id, doc in tqdm(doc_map.items(), desc="Building BGE-M3 indexes", dynamic_ncols=False):
        paras = doc["paragraphs"]
        index = retriever.build_index(paras, doc_id=doc_id)
        index_map[doc_id] = (paras, index)

    print("Hybrid retrieve + cross-encoder rerank for all queries ...", flush=True)
    retrieved_all = []
    for q in tqdm(queries, desc="Retrieve+Rerank", dynamic_ncols=False):
        paras, index = index_map[q["doc_id"]]
        candidates = retriever.retrieve(q["query"], paras, index, top_k=TOP_K_RETRIEVE)
        retrieved = retriever.rerank(q["query"], candidates, top_k=TOP_K_FINAL)
        retrieved_all.append(retrieved)

    # Keep reranker on CPU for post-gen ref verification
    reranker_tok   = retriever.reranker_tok
    reranker_model = retriever.reranker_model.to("cpu")
    reranker_model.eval()

    del retriever
    del index_map
    torch.cuda.empty_cache()
    print(f"BGE-M3 freed | VRAM: {torch.cuda.memory_allocated(0)/1e9:.1f} GB", flush=True)

    # Para lookup for ref verification
    para_text_map: dict[str, str] = {}
    for doc in test_data["docs"]:
        for p in doc["paragraphs"]:
            para_text_map[p["para_id"]] = p["text"]

    print(f"[STEP 3] Building prompts ...", flush=True)
    tok = AutoTokenizer.from_pretrained(LLM_PATH)
    prompts = []
    for q, retrieved in zip(queries, retrieved_all):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(q["query"], retrieved)},
        ]
        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(prompt)
    del tok

    print(f"[STEP 4] Loading {LLM_PATH} via vLLM ...", flush=True)
    llm = LLM(
        model=LLM_PATH,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=12288,
        max_num_seqs=2,
        trust_remote_code=True,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=MAX_NEW_TOKENS,
    )

    print(f"Generating {len(prompts)} answers ...", flush=True)
    outputs = llm.generate(prompts, sampling_params)

    REF_VERIFY_THRESHOLD  = 0.10   # drop original refs below this
    NEIGHBOR_THRESHOLD    = 0.30   # add ±1 neighbors above this

    def _cross_score(query: str, para_ids: list[str]) -> list[float]:
        pairs = [[query, para_text_map[r][:800]] for r in para_ids]
        enc = reranker_tok(
            [p[0] for p in pairs], [p[1] for p in pairs],
            padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        with torch.no_grad():
            logits = reranker_model(**enc).logits.squeeze(-1)
        scores = torch.sigmoid(logits).tolist()
        return [scores] if isinstance(scores, float) else scores

    def verify_refs(query: str, refs: list[str], valid_ids: set[str]) -> list[str]:
        valid = [r for r in refs if r in para_text_map]
        if not valid:
            return refs

        # Step 1: drop low-scoring original refs
        scores = _cross_score(query, valid)
        kept = [r for r, s in zip(valid, scores) if s >= REF_VERIFY_THRESHOLD]
        if not kept:
            kept = [valid[int(torch.tensor(scores).argmax())]]

        # Step 2: neighbor expansion — check ±1 adjacent paras for each kept ref
        candidates = set(kept)
        for r in kept:
            m = re.match(r'([A-Za-z]*)(\d+)$', r)
            if not m:
                continue
            pfx, num = m.group(1), int(m.group(2))
            for nbr in (f"{pfx}{num-1}", f"{pfx}{num+1}"):
                if nbr in valid_ids and nbr not in candidates and nbr in para_text_map:
                    candidates.add(nbr)

        new_neighbors = [r for r in candidates if r not in kept]
        if new_neighbors:
            nbr_scores = _cross_score(query, new_neighbors)
            for r, s in zip(new_neighbors, nbr_scores):
                if s >= NEIGHBOR_THRESHOLD:
                    kept.append(r)

        return sorted(set(kept), key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)

    rows = []
    for q, retrieved, output in zip(queries, retrieved_all, outputs):
        gen_text = output.outputs[0].text
        fallback = retrieved[0]["para_id"] if retrieved else "P1"
        valid_ids = {p["para_id"] for p in doc_map[q["doc_id"]]["paragraphs"]}
        abstractive, refs = parse_output(gen_text, fallback, valid_ids)
        refs = verify_refs(q["query"], refs, valid_ids)
        rows.append({"ID": q["ID"], "abstractive": abstractive, "refs": ",".join(refs)})
        tqdm.write(f"[{q['ID']}] refs={refs} | {abstractive[:70]}...")

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
