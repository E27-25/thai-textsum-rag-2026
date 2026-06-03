"""
Generate DPO rejected samples v2 — FIXED:
  - Match SFT training chat template (no enable_thinking parameter)
  - Sampling generation (temp=0.7) for more realistic model distribution
  - Generate N=2 samples, pick worst as rejected for stronger DPO signal
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json, random, re
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from tqdm import tqdm
from rouge_score import rouge_scorer

BASE        = Path("/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther")
BGE_PATH    = str(BASE / "bge-m3")
LLM_PATH    = str(BASE / "Qwen3-32B-SFT-v1")
DATA_PATH   = BASE / "data" / "ชุดข้อมูล" / "train_set.json"
OUTPUT_FILE = str(BASE / "dpo_pairs_32b_v2.json")

N_DISTRACTORS  = 5     # match SFT v1 training context (7 distractors there but 5 here for speed)
N_SAMPLES      = 2     # generate 2 per query, pick worst
MAX_PARA_CHARS = 800
MAX_NEW_TOKENS = 180
SEED           = 42
BGE_BATCH      = 128
ROUGEL_SKIP    = 0.85   # only skip near-perfect outputs

SYSTEM_PROMPT = (
    "คุณเป็นผู้ช่วยตอบคำถามจากบันทึกการประชุมภาษาไทย "
    "ตอบให้ครบถ้วนและชัดเจน โดยใช้ถ้อยคำและสำนวนจากย่อหน้าที่ให้มาโดยตรง "
    "อ้างอิงเฉพาะข้อมูลจากย่อหน้าที่ให้มาเท่านั้น "
    "หากคำตอบกระจายอยู่หลายย่อหน้าให้ระบุทุก ID ที่เกี่ยวข้อง"
)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
def rougel(hyp, ref): return scorer.score(ref, hyp)["rougeL"].fmeasure

print("Loading training data...")
with open(DATA_PATH) as f:
    data = json.load(f)
docs_index = {d["doc_id"]: d["paragraphs"] for d in data["docs"]}
queries    = data["queries"]
print(f"  {len(queries)} queries")

print(f"Loading BGE-M3 on {device}...")
bge_tok   = AutoTokenizer.from_pretrained(BGE_PATH)
bge_model = AutoModel.from_pretrained(BGE_PATH, torch_dtype=torch.bfloat16).to(device)
bge_model.eval()

def encode_cls(texts, batch_size=BGE_BATCH):
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = bge_tok(batch, padding=True, truncation=True,
                         max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            out = bge_model(**inputs)
        emb = F.normalize(out.last_hidden_state[:, 0, :], p=2, dim=1)
        all_emb.append(emb.float().cpu().numpy())
    return np.vstack(all_emb)

print("Building embeddings...")
doc_embs = {}
for doc_id, paras in tqdm(docs_index.items(), desc="  indexing"):
    valid = [p for p in paras if p["text"]]
    embs  = encode_cls([p["text"] for p in valid])
    doc_embs[doc_id] = (valid, embs)

def get_hard_negatives(query_text, doc_id, correct_ids, n=N_DISTRACTORS):
    paras, embs = doc_embs[doc_id]
    scores = embs @ encode_cls([query_text])[0]
    negs = []
    for i in np.argsort(scores)[::-1]:
        if paras[i]["para_id"] not in correct_ids:
            negs.append(paras[i])
            if len(negs) >= n: break
    return negs

def build_prompt(query, context_paras):
    context = "\n".join(f"[{p['para_id']}] {p['text'][:MAX_PARA_CHARS]}" for p in context_paras)
    ids = ", ".join(f'"{p["para_id"]}"' for p in context_paras)
    return (
        f"ย่อหน้าจากบันทึกการประชุม:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"ให้ตอบคำถามโดยสรุปคำตอบจากย่อหน้าข้างต้น และระบุ ID ย่อหน้าที่เกี่ยวข้อง\n"
        f"ID ที่เลือกได้: {ids}\n\n"
        f"ตอบเป็น JSON เท่านั้น:\n"
        f'{{"abstractive": "สรุปคำตอบภาษาไทย", "refs": ["id1", "id2"]}}'
    )

def extract_abstractive(text):
    try:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            obj = json.loads(m.group())
            abst = str(obj.get("abstractive", "")).strip()
            return abst if abst else None
    except Exception:
        pass
    return None

print("Building contexts...")
llm_tok = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)
contexts = []
for q in tqdm(queries, desc="  contexts"):
    para_map = {p["para_id"]: p for p in docs_index[q["doc_id"]]}
    oracle = [para_map[r] for r in q["refs"] if r in para_map and para_map[r]["text"]]
    if not oracle:
        contexts.append(None); continue
    hard_negs = get_hard_negatives(q["query"], q["doc_id"], set(q["refs"]))
    random.shuffle(hard_negs)
    contexts.append(hard_negs + oracle)

del bge_model; torch.cuda.empty_cache()

print(f"Loading {LLM_PATH} bfloat16 across 4 GPUs ...")
gen_model = AutoModelForCausalLM.from_pretrained(
    LLM_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
)
gen_model.eval()

print(f"Generating N={N_SAMPLES} samples per query (sampling) ...")
pairs, skipped = [], 0

for q, context in tqdm(zip(queries, contexts), total=len(queries), desc="  generating"):
    if context is None:
        skipped += 1; continue

    prompt = build_prompt(q["query"], context)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    # MATCH SFT training template — no enable_thinking parameter
    prompt_text = llm_tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = llm_tok(prompt_text, return_tensors="pt").to("cuda:0")
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = gen_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=N_SAMPLES,
            pad_token_id=llm_tok.eos_token_id,
        )

    # Score all N, pick worst as rejected
    candidates = []
    for seq in out:
        gen_text = llm_tok.decode(seq[input_len:], skip_special_tokens=True)
        abst = extract_abstractive(gen_text)
        if not abst: continue
        rl = rougel(abst, q["abstractive"])
        candidates.append((rl, gen_text))

    if not candidates:
        skipped += 1; continue

    candidates.sort(key=lambda x: x[0])  # ascending: worst first
    worst_rl, worst_text = candidates[0]

    if worst_rl >= ROUGEL_SKIP:
        skipped += 1; continue  # all candidates too good

    gt_answer = json.dumps({"abstractive": q["abstractive"], "refs": q["refs"]}, ensure_ascii=False)
    pairs.append({
        "prompt":   prompt_text,
        "chosen":   gt_answer,
        "rejected": worst_text,
        "rougel":   round(worst_rl, 4),
    })

print(f"  {len(pairs)} pairs created, {skipped} skipped")
if pairs:
    print(f"  Mean RougeL of rejected: {np.mean([p['rougel'] for p in pairs]):.3f}")
    print(f"  Median: {np.median([p['rougel'] for p in pairs]):.3f}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(pairs, f, ensure_ascii=False, indent=2)
print(f"Saved to {OUTPUT_FILE}")
