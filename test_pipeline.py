#!/usr/bin/env python3
"""Quick sanity test for run_vllm.py logic (no LLM required)."""
import sys, re, json
from collections import Counter as _Counter

BASE = "/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther"
BGE_PATH      = f"{BASE}/bge-m3"
RERANKER_PATH = f"{BASE}/bge-reranker-v2-m3"

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def check(name, expr):
    try:
        result = expr()
        print(f"[{PASS}] {name}: {result}")
        return True
    except Exception as e:
        print(f"[{FAIL}] {name}: {e}")
        return False

# ─── 1. imports ─────────────────────────────────────────────────────────────
print("\n=== 1. Imports ===")
import torch
print(f"[{PASS}] torch {torch.__version__} | CUDA={torch.cuda.is_available()} | devices={torch.cuda.device_count()}")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from FlagEmbedding import BGEM3FlagModel
print(f"[{PASS}] transformers + FlagEmbedding")

try:
    from pythainlp.tokenize import word_tokenize as _wt
    _THAI_TOK = True
    print(f"[{PASS}] pythainlp")
except Exception as e:
    _THAI_TOK = False
    print(f"[WARN] pythainlp: {e}")

# ─── 2. sort guard fix ───────────────────────────────────────────────────────
print("\n=== 2. Sort guard (the crash fix) ===")

def sort_refs(refs):
    return sorted(set(refs), key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)

# Normal case
check("normal refs", lambda: sort_refs(["P3", "P1", "P2"]))
# The crash case — ref without digits
check("ref without digits", lambda: sort_refs(["P3", "P", "P1"]))
# Empty string
check("empty string ref", lambda: sort_refs(["P3", "", "P1"]))
# Mixed garbage
check("garbage refs", lambda: sort_refs(["P10", "para", "P5", "refs"]))

# ─── 3. expand_contiguous ────────────────────────────────────────────────────
print("\n=== 3. expand_contiguous (gap=1 and gap=2) ===")

def expand_contiguous(refs, valid_ids):
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

valid = {f"P{i}" for i in range(1, 200)}
check("gap=1 fill", lambda: expand_contiguous(["P51","P53"], valid))       # expect P51,P52,P53
check("gap=2 fill", lambda: expand_contiguous(["P51","P54"], valid))       # expect P51,P52,P53,P54
check("gap=3 no fill", lambda: expand_contiguous(["P51","P55"], valid))    # expect P51,P55 only
check("single ref", lambda: expand_contiguous(["P51"], valid))             # no change

# ─── 4. BM25 ─────────────────────────────────────────────────────────────────
print("\n=== 4. BM25 ===")

class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1; self.b = b
        self.corpus_tfs = []; self.df = {}; self.avgdl = 0.0; self.N = 0

    @staticmethod
    def _tokenize(text):
        if _THAI_TOK:
            return _wt(text, engine="newmm", keep_whitespace=False)
        return list(text)

    def build(self, texts):
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

    def score(self, query, idx):
        dl, tf = self.corpus_tfs[idx]
        norm = 1 - self.b + self.b * dl / max(self.avgdl, 1)
        s = 0.0
        for tok in self._tokenize(query):
            if tok not in tf:
                continue
            idf = max(0.0, (self.N - self.df.get(tok, 0) + 0.5) / (self.df.get(tok, 0) + 0.5))
            tf_val = tf[tok] * (self.k1 + 1) / (tf[tok] + self.k1 * norm)
            s += idf * tf_val
        return s

docs = [
    "มติคณะกรรมการอ้อยและน้ำตาลทรายออกมาตรการแก้ไขปัญหาอ้อยไฟไหม้",
    "การประชุมคณะกรรมการพิจารณาโครงการก่อสร้างถนน",
    "รายงานผลการดำเนินงานประจำปีงบประมาณ 2567",
    "มาตรการสินเชื่อและการส่งเสริมชาวไร่อ้อย",
]
bm25 = BM25()
bm25.build(docs)
scores = [bm25.score("มาตรการแก้ไขปัญหาอ้อยไฟไหม้", i) for i in range(len(docs))]
best = docs[scores.index(max(scores))][:40]
check("BM25 scores computed", lambda: scores)
check("BM25 top result relevant", lambda: "อ้อย" in best or scores[0] == max(scores))

# ─── 5. Reranker load + score ────────────────────────────────────────────────
print("\n=== 5. Cross-encoder reranker (torch_dtype fix) ===")

dtype = torch.float16

def load_reranker():
    tok = AutoTokenizer.from_pretrained(RERANKER_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        RERANKER_PATH, torch_dtype=dtype   # ← the fixed line
    ).to("cpu")
    model.eval()
    return tok, model

def rerank_score(tok, model, query, passages):
    enc = tok([query]*len(passages), passages,
              padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits.squeeze(-1)
    scores = torch.sigmoid(logits).tolist()
    return [scores] if isinstance(scores, float) else scores

tok, model = None, None
try:
    print("  Loading reranker (CPU, fp16)...", flush=True)
    tok, model = load_reranker()
    print(f"[{PASS}] reranker loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")
except Exception as e:
    print(f"[{FAIL}] reranker load: {e}")

if tok and model:
    query = "มติคณะกรรมการออกมาตรการแก้ไขปัญหาอ้อยไฟไหม้อย่างไร"
    passages = [
        "คณะกรรมการมีมติให้หักเงินชาวไร่อ้อยที่ส่งอ้อยไฟไหม้ 1,000 บาทต่อตัน",
        "การประชุมสภาผู้แทนราษฎรพิจารณาร่างพระราชบัญญัติงบประมาณ",
        "รณรงค์ประชาสัมพันธ์ให้ชาวไร่หยุดเผาอ้อย และมีสินเชื่อดอกเบี้ยต่ำ",
    ]
    check("reranker scores", lambda: rerank_score(tok, model, query, passages))
    scores = rerank_score(tok, model, query, passages)
    best_idx = scores.index(max(scores))
    check("reranker top result relevant", lambda: best_idx in (0, 2))
    print(f"  Scores: {[f'{s:.3f}' for s in scores]}")
    print(f"  Best: [{best_idx}] {passages[best_idx][:60]}")

# ─── 6. BGE-M3 hybrid retrieval (GPU if available) ───────────────────────────
if torch.cuda.is_available():
    print("\n=== 6. BGE-M3 hybrid retrieval (GPU) ===")
    try:
        print("  Loading BGE-M3...", flush=True)
        bge = BGEM3FlagModel(BGE_PATH, use_fp16=True, device="cuda:0")
        paragraphs = [
            {"para_id": "P1", "text": "มติคณะกรรมการอ้อยและน้ำตาลทราย ออกมาตรการหักเงินชาวไร่"},
            {"para_id": "P2", "text": "โครงการก่อสร้างถนนสายหลักในจังหวัดชนบท"},
            {"para_id": "P3", "text": "มาตรการแก้ไขปัญหาอ้อยไฟไหม้ สินเชื่อดอกเบี้ยต่ำ รณรงค์"},
            {"para_id": "P4", "text": "รายงานการเงินประจำปีงบประมาณ 2567"},
        ]
        texts = [p["text"] for p in paragraphs]
        out = bge.encode(texts, batch_size=4, return_dense=True, return_sparse=True,
                         return_colbert_vecs=False, max_length=512)
        dense = torch.tensor(out["dense_vecs"])
        sparse = out["lexical_weights"]

        q_out = bge.encode(["มาตรการแก้ไขปัญหาอ้อย"], return_dense=True, return_sparse=True,
                           return_colbert_vecs=False, max_length=512)
        q_dense = torch.tensor(q_out["dense_vecs"][0])
        scores = (dense @ q_dense).tolist()
        best_idx = scores.index(max(scores))
        print(f"[{PASS}] BGE-M3 hybrid: top={paragraphs[best_idx]['para_id']} '{paragraphs[best_idx]['text'][:50]}'")
        del bge; torch.cuda.empty_cache()
    except Exception as e:
        print(f"[{FAIL}] BGE-M3: {e}")
else:
    print("\n=== 6. BGE-M3: skipped (no GPU) ===")

print("\n=== Done ===")
