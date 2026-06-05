"""Run the FINAL inference pipeline on the 50-query sample test and time every stage.
Drop reranker version: dense BGE-M3 only → no reranker calls.
Outputs predictions JSON + timing log."""
import os, json, time, re, sys, gc

t0 = time.time()
def log(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

TEST_PATH = "data/ชุดข้อมูล/ตัวอย่าง_test_set.json"
LLM_PATH  = "Qwen3-14B-SFT-v5-AWQ"
BGE_PATH  = "bge-m3"
RERANK    = os.environ.get("USE_RERANKER", "0") == "1"
RERANK_PATH = "bge-reranker-v2-m3"
TOP_K_RETRIEVE = int(os.environ.get("TOP_K_RETRIEVE", "15"))
TOP_K_FINAL    = int(os.environ.get("TOP_K_FINAL", "7"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "120"))
OUT_PATH       = os.environ.get("OUT_PATH", "sample_predictions.json")

log(f"config: RERANK={RERANK} TOP_K_RETRIEVE={TOP_K_RETRIEVE} TOP_K_FINAL={TOP_K_FINAL} MAX_NEW_TOKENS={MAX_NEW_TOKENS}")

log("import torch / FlagEmbedding / vLLM ...")
import torch
from FlagEmbedding import BGEM3FlagModel
if RERANK:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vllm import LLM, SamplingParams

log("load test set")
data = json.load(open(TEST_PATH))
docs = {d["doc_id"]: d for d in data["docs"]}
queries = data["queries"]
log(f"  docs={len(docs)}  queries={len(queries)}")

log("load BGE-M3 dense")
bge = BGEM3FlagModel(BGE_PATH, use_fp16=True)

log("embed corpus + retrieve")
para_text_map = {}
doc_embeds = {}
for doc_id, doc in docs.items():
    para_ids   = [p["para_id"] for p in doc["paragraphs"]]
    para_texts = [p["text"]    for p in doc["paragraphs"]]
    for pid, txt in zip(para_ids, para_texts):
        para_text_map[(doc_id,pid)] = txt
    emb = bge.encode(para_texts, batch_size=32, max_length=512)["dense_vecs"]
    doc_embeds[doc_id] = (para_ids, torch.tensor(emb))

q_embeds = bge.encode([q["query"] for q in queries], batch_size=32, max_length=128)["dense_vecs"]

retrieved_all = []
import numpy as np
for q, qe in zip(queries, q_embeds):
    pids, pe = doc_embeds[q["doc_id"]]
    sims = (pe @ torch.tensor(qe)).cpu().numpy()
    idx = sims.argsort()[::-1][:TOP_K_RETRIEVE]
    retrieved_all.append([(pids[i], float(sims[i])) for i in idx])

t_retrieve = time.time() - t0
log(f"retrieve done — {t_retrieve:.1f}s")

# Optional reranker
if RERANK:
    log("load reranker")
    tok = AutoTokenizer.from_pretrained(RERANK_PATH)
    rer = AutoModelForSequenceClassification.from_pretrained(RERANK_PATH, torch_dtype=torch.float16).cuda().eval()
    log("rerank")
    new_retrieved = []
    for q, cands in zip(queries, retrieved_all):
        pairs = [(q["query"], para_text_map[(q["doc_id"], pid)][:500]) for pid,_ in cands]
        enc = tok([p[0] for p in pairs], [p[1] for p in pairs],
                  padding=True, truncation=True, max_length=384, return_tensors="pt").to("cuda")
        with torch.no_grad():
            scores = rer(**enc).logits.squeeze(-1).cpu().tolist()
        ranked = sorted(zip([c[0] for c in cands], scores), key=lambda x: -x[1])[:TOP_K_FINAL]
        new_retrieved.append(ranked)
    retrieved_all = new_retrieved
    del rer, tok; gc.collect(); torch.cuda.empty_cache()
    t_rerank = time.time() - t0
    log(f"rerank done — {t_rerank:.1f}s")
else:
    retrieved_all = [r[:TOP_K_FINAL] for r in retrieved_all]
    t_rerank = t_retrieve

# Free BGE
del bge; gc.collect(); torch.cuda.empty_cache()
log("BGE freed")

log("load vLLM AWQ")
llm = LLM(model=LLM_PATH, quantization="awq", dtype="float16",
          max_model_len=6144, max_num_seqs=16, gpu_memory_utilization=0.85,
          enforce_eager=False, trust_remote_code=True)
t_llm_load = time.time() - t0
log(f"vLLM loaded — {t_llm_load:.1f}s")

SYSTEM = "คุณเป็นผู้ช่วยสรุปบันทึกการประชุมรัฐสภาภาษาไทยแบบ abstractive ตอบให้กระชับ"

def build_prompt(q, retrieved):
    pid_list = ", ".join(pid for pid,_ in retrieved)
    context = "\n".join(f"[{pid}] {para_text_map[(q['doc_id'], pid)]}" for pid,_ in retrieved)
    user = (
        f"คำถาม: {q['query']}\n\n"
        f"บริบท (ย่อหน้าที่เกี่ยวข้อง: {pid_list}):\n{context}\n\n"
        f"กรุณาสรุปคำตอบให้กระชับ (5-20 คำ) แบบ abstractive (เรียบเรียงใหม่ ไม่ลอกประโยคตรงๆ) "
        f"และระบุย่อหน้าทั้งหมดที่สนับสนุนคำตอบ (อาจมีหลายย่อหน้า)\n"
        f'ตอบเป็น JSON: {{"abstractive":"...", "refs":["P1","P2"]}}'
    )
    return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"

prompts = [build_prompt(q, r) for q, r in zip(queries, retrieved_all)]
log(f"prompts built — median len {sorted(len(p) for p in prompts)[len(prompts)//2]} chars")

params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS, stop=["<|im_end|>"])
t_gen0 = time.time()
outputs = llm.generate(prompts, params)
t_gen = time.time() - t_gen0
log(f"gen done — {t_gen:.1f}s  ({t_gen/len(queries)*1000:.0f}ms/query)")

# Parse + save
def parse_output(text, fallback):
    try:
        m = re.search(r'\{.*?\}', text, re.DOTALL)
        if m:
            obj = json.loads(m.group())
            a = str(obj.get("abstractive","")).strip()
            r = [str(x) for x in obj.get("refs",[])]
            if a and r: return a, r
    except Exception: pass
    am = re.search(r'"abstractive"\s*:\s*"((?:[^"\\]|\\.)*)"?', text, re.DOTALL)
    a = am.group(1).strip() if am else ""
    rm = re.search(r'"refs"\s*:\s*\[([^\]]*)', text, re.DOTALL)
    r = re.findall(r'"([^"]+)"', rm.group(1)) if rm else []
    return a or text.strip(), r or [fallback]

rows = []
for q, retrieved, out in zip(queries, retrieved_all, outputs):
    text = out.outputs[0].text
    fb = retrieved[0][0] if retrieved else "P1"
    a, r = parse_output(text, fb)
    rows.append({"ID": q["ID"], "abstractive": a, "refs": ",".join(r)})

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)

t_total = time.time() - t0
log("======== TIMING SUMMARY ========")
log(f"vLLM load     : {t_llm_load - t_rerank:6.1f}s")
log(f"BGE retrieve  : {t_retrieve:6.1f}s ({t_retrieve/len(queries)*1000:.0f}ms/q)")
log(f"reranker      : {t_rerank - t_retrieve:6.1f}s")
log(f"LLM generate  : {t_gen:6.1f}s ({t_gen/len(queries)*1000:.0f}ms/q)")
log(f"TOTAL         : {t_total:6.1f}s for {len(queries)} queries")
log(f"50q sample → est. 1239q full: {t_total * 1239/len(queries) / 60:.1f} min")
log(f"           → est.  500q hidden: {t_total *  500/len(queries) / 60:.1f} min")
log(f"           → est.  300q hidden: {t_total *  300/len(queries) / 60:.1f} min")
