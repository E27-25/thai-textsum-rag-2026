#!/usr/bin/env python3
"""Thinking mode enabled. Model reasons before answering."""
import time; time.sleep(10)
import sys, os, glob, json, re, csv, subprocess
print("[BOOT] Python starting", flush=True)

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from FlagEmbedding import BGEM3FlagModel
from vllm import LLM, SamplingParams
from tqdm import tqdm
from collections import Counter as _Counter

BGE_PATH      = "/model/bge-m3"
RERANKER_PATH = "/model/bge-reranker-v2-m3"
LLM_PATH      = os.environ.get("LLM_PATH", "/model/Qwen3-32B-SFT-v1-AWQ")
TEST_DIR     = "/model/test/"
RESULT_DIR   = "/result/"
PROGRESS_LIB = "/benchmark_lib/progress"
OUTPUT_CSV   = os.path.join(RESULT_DIR, "submission.csv")

# Trim retrieval to give thinking room
TOP_K_RETRIEVE = 15
TOP_K_FINAL    = 5
SPARSE_WEIGHT  = 0.3
BM25_WEIGHT    = 0.2
MAX_NEW_TOKENS = 600   # thinking ~400 + answer ~150 + buffer
GPU            = "cuda:0"
BGE_BATCH      = 128

SYSTEM_PROMPT = (
    "คุณเป็นผู้ช่วยตอบคำถามจากบันทึกการประชุมภาษาไทย "
    "ตอบให้ครบถ้วนและชัดเจน โดยใช้ถ้อยคำและสำนวนจากย่อหน้าที่ให้มาโดยตรง "
    "อ้างอิงเฉพาะข้อมูลจากย่อหน้าที่ให้มาเท่านั้น "
    "หากคำตอบกระจายอยู่หลายย่อหน้าให้ระบุทุก ID ที่เกี่ยวข้อง"
)


class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1; self.b = b
        self.corpus_tfs = []; self.df = {}; self.avgdl = 0.0; self.N = 0
    @staticmethod
    def _tokenize(text):
        return re.findall(r'[฀-๿]+|[A-Za-z0-9]+', text)
    def build(self, texts):
        self.N = len(texts); tfs = []
        for t in texts:
            toks = self._tokenize(t); tf = _Counter(toks)
            tfs.append((len(toks), tf))
        self.avgdl = sum(l for l, _ in tfs) / max(self.N, 1)
        self.df = {}
        for _, tf in tfs:
            for tok in tf:
                self.df[tok] = self.df.get(tok, 0) + 1
        self.corpus_tfs = tfs
    def score(self, query, idx):
        dl, tf = self.corpus_tfs[idx]
        norm = 1 - self.b + self.b * dl / max(self.avgdl, 1)
        s = 0.0
        for tok in self._tokenize(query):
            if tok not in tf: continue
            idf = max(0.0, ((self.N - self.df.get(tok, 0) + 0.5) / (self.df.get(tok, 0) + 0.5)))
            tf_val = tf[tok] * (self.k1 + 1) / (tf[tok] + self.k1 * norm)
            s += idf * tf_val
        return s


class BGERetriever:
    def __init__(self, model_path, reranker_path, device):
        self.device = device
        self.model = BGEM3FlagModel(model_path, use_fp16=(device != "cpu"), device=device)
        dtype = torch.float16 if device != "cpu" else torch.float32
        self.reranker_tok = AutoTokenizer.from_pretrained(reranker_path)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
            reranker_path, torch_dtype=dtype).to(device)
        self.reranker_model.eval()
    def build_index(self, paragraphs, doc_id=""):
        texts = [p["text"] for p in paragraphs]
        out = self.model.encode(texts, batch_size=BGE_BATCH,
            return_dense=True, return_sparse=True, return_colbert_vecs=False, max_length=512)
        return {"dense": torch.tensor(out["dense_vecs"]),
                "sparse": out["lexical_weights"],
                "bm25": (lambda b: (b.build(texts), b)[1])(BM25())}
    def _sparse_score(self, q, p):
        return sum(w * p[tid] for tid, w in q.items() if tid in p)
    def retrieve(self, query, paragraphs, index, top_k):
        q_out = self.model.encode([query], return_dense=True, return_sparse=True,
            return_colbert_vecs=False, max_length=512)
        q_dense = torch.tensor(q_out["dense_vecs"][0]); q_sparse = q_out["lexical_weights"][0]
        n = len(paragraphs)
        dense_scores = (index["dense"] @ q_dense).tolist()
        bm25_raw = [index["bm25"].score(query, i) for i in range(n)]
        bm25_max = max(bm25_raw) if bm25_raw else 1.0
        bm25_scores = [s / max(bm25_max, 1e-9) for s in bm25_raw]
        hybrid = [dense_scores[i] + SPARSE_WEIGHT * self._sparse_score(q_sparse, index["sparse"][i])
                  + BM25_WEIGHT * bm25_scores[i] for i in range(n)]
        top_idx = sorted(range(n), key=lambda i: hybrid[i], reverse=True)[:top_k]
        return [paragraphs[i] for i in top_idx]
    def rerank(self, query, passages, top_k):
        enc = self.reranker_tok([query] * len(passages), [p["text"][:800] for p in passages],
            padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.reranker_model(**enc).logits.squeeze(-1)
        scores = torch.sigmoid(logits).tolist()
        if isinstance(scores, float): scores = [scores]
        ranked = sorted(range(len(passages)), key=lambda i: scores[i], reverse=True)
        return [passages[i] for i in ranked[:top_k]]


def build_prompt(query, retrieved, max_para_chars=800):
    ordered = list(reversed(retrieved))
    context = "\n".join(f"[{p['para_id']}] {p['text'][:max_para_chars]}" for p in ordered)
    ids = ", ".join(f'"{p["para_id"]}"' for p in ordered)
    return (f"ย่อหน้าจากบันทึกการประชุม:\n{context}\n\n"
            f"คำถาม: {query}\n\nให้ตอบคำถามโดยสรุปคำตอบจากย่อหน้าข้างต้น และระบุ ID ย่อหน้าที่เกี่ยวข้อง\n"
            f"ID ที่เลือกได้: {ids}\n\nตอบเป็น JSON เท่านั้น:\n"
            f'{{"abstractive": "สรุปคำตอบภาษาไทย", "refs": ["id1", "id2"]}}')


def expand_contiguous(refs, valid_ids):
    if len(refs) <= 1: return refs
    nums = []
    for r in refs:
        m = re.match(r'([A-Za-z]+)(\d+)$', r)
        if m: nums.append((int(m.group(2)), m.group(1), r))
    if not nums: return refs
    nums.sort(); result_ids = set(refs)
    for i in range(len(nums) - 1):
        n1, pfx, _ = nums[i]; n2, _, _ = nums[i + 1]
        if n2 - n1 == 2:
            mid = f"{pfx}{n1 + 1}"
            if mid in valid_ids: result_ids.add(mid)
        elif n2 - n1 == 3:
            for step in (1, 2):
                mid = f"{pfx}{n1 + step}"
                if mid in valid_ids: result_ids.add(mid)
    return sorted(result_ids, key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)


def parse_output(text, fallback_para, valid_para_ids=None):
    # Strip <think>...</think> first so JSON regex finds the right block
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
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
    print(f"[ENV] LLM_PATH={LLM_PATH}", flush=True)
    json_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.json")))
    if not json_files: raise FileNotFoundError(f"No JSON in {TEST_DIR}")
    test_path = json_files[0]; print(f"Test file: {test_path}", flush=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    with open(test_path, encoding="utf-8") as f:
        test_data = json.load(f)
    doc_map = {d["doc_id"]: d for d in test_data["docs"]}
    queries = test_data["queries"]
    print(f"Loaded {len(queries)} queries across {len(doc_map)} docs", flush=True)

    retriever = BGERetriever(BGE_PATH, RERANKER_PATH, GPU)
    index_map = {}
    for doc_id, doc in tqdm(doc_map.items(), desc="Index"):
        index_map[doc_id] = (doc["paragraphs"], retriever.build_index(doc["paragraphs"]))

    retrieved_all = []
    for q in tqdm(queries, desc="Retrieve+Rerank"):
        paras, index = index_map[q["doc_id"]]
        cand = retriever.retrieve(q["query"], paras, index, top_k=TOP_K_RETRIEVE)
        retrieved_all.append(retriever.rerank(q["query"], cand, top_k=TOP_K_FINAL))

    reranker_tok = retriever.reranker_tok
    reranker_model = retriever.reranker_model.to("cpu"); reranker_model.eval()
    del retriever, index_map
    torch.cuda.empty_cache()

    para_text_map = {p["para_id"]: p["text"] for doc in test_data["docs"] for p in doc["paragraphs"]}

    tok = AutoTokenizer.from_pretrained(LLM_PATH)
    prompts = []
    for q, retrieved in zip(queries, retrieved_all):
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(q["query"], retrieved)}]
        # enable_thinking=True — model will reason before answering
        prompts.append(tok.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=True))
    del tok

    print(f"[STEP 4] Loading {LLM_PATH} via vLLM (thinking mode) ...", flush=True)
    llm = LLM(model=LLM_PATH, dtype="auto", tensor_parallel_size=1,
              gpu_memory_utilization=0.92, max_model_len=8192, max_num_seqs=4,
              trust_remote_code=True)

    sampling_params = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS)
    print(f"Generating {len(prompts)} answers (with thinking) ...", flush=True)
    outputs = llm.generate(prompts, sampling_params)

    def _cross_score(query, texts):
        enc = reranker_tok([query] * len(texts), [t[:800] for t in texts],
            padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            logits = reranker_model(**enc).logits.squeeze(-1)
        scores = torch.sigmoid(logits).tolist()
        return [scores] if isinstance(scores, float) else scores

    REF_VERIFY_THRESHOLD = 0.10
    NEIGHBOR_THRESHOLD   = 0.30

    def verify_refs(query, refs, valid_ids):
        valid = [r for r in refs if r in para_text_map]
        if not valid: return refs
        scores = _cross_score(query, [para_text_map[r] for r in valid])
        kept = [r for r, s in zip(valid, scores) if s >= REF_VERIFY_THRESHOLD]
        if not kept: kept = [valid[int(torch.tensor(scores).argmax())]]
        candidates = set(kept)
        for r in kept:
            m = re.match(r'([A-Za-z]*)(\d+)$', r)
            if not m: continue
            pfx, num = m.group(1), int(m.group(2))
            for nbr in (f"{pfx}{num-1}", f"{pfx}{num+1}"):
                if nbr in valid_ids and nbr not in candidates and nbr in para_text_map:
                    candidates.add(nbr)
        new_neighbors = [r for r in candidates if r not in kept]
        if new_neighbors:
            nbr_scores = _cross_score(query, [para_text_map[r] for r in new_neighbors])
            for r, s in zip(new_neighbors, nbr_scores):
                if s >= NEIGHBOR_THRESHOLD: kept.append(r)
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
        writer.writeheader(); writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows → {OUTPUT_CSV}", flush=True)
    try:
        subprocess.run([PROGRESS_LIB, str(len(rows))], check=True)
    except Exception as e:
        print(f"Progress update skipped: {e}", flush=True)


if __name__ == "__main__":
    main()
