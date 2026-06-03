"""
Generate DPO rejected samples using Qwen3-32B-SFT-v1.
Oracle context (2 distractors + oracle paras) — matches SFT training distribution.
chosen = ground-truth JSON, rejected = model's greedy output.
Saves to dpo_pairs_32b.json.
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
OUTPUT_FILE = str(BASE / "dpo_pairs_32b.json")

N_DISTRACTORS  = 2
MAX_PARA_CHARS = 800
MAX_NEW_TOKENS = 200
SEED           = 42
BGE_BATCH      = 128
ROUGEL_SKIP    = 0.80   # skip pair if model output already ≥ this quality

SYSTEM_PROMPT = (
    "คุณเป็นผู้ช่วยตอบคำถามจากบันทึกการประชุมภาษาไทย "
    "ตอบให้ครบถ้วนและชัดเจน โดยใช้ถ้อยคำและสำนวนจากย่อหน้าที่ให้มาโดยตรง "
    "อ้างอิงเฉพาะข้อมูลจากย่อหน้าที่ให้มาเท่านั้น "
    "หากคำตอบกระจายอยู่หลายย่อหน้าให้ระบุทุก ID ที่เกี่ยวข้อง"
)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

def rougel(hyp: str, ref: str) -> float:
    return scorer.score(ref, hyp)["rougeL"].fmeasure

print("Loading training data...")
with open(DATA_PATH) as f:
    data = json.load(f)
docs_index = {d["doc_id"]: d["paragraphs"] for d in data["docs"]}
queries    = data["queries"]
print(f"  {len(queries)} queries across {len(docs_index)} documents")

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
            if len(negs) >= n:
                break
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

def extract_abstractive(text: str):
    try:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            obj = json.loads(m.group())
            abst = str(obj.get("abstractive", "")).strip()
            return abst if abst else None
    except Exception:
        pass
    return None

# Pre-build all contexts while BGE-M3 is loaded
print("Building oracle contexts for all queries...")
llm_tok = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)
contexts = []
for q in tqdm(queries, desc="  contexts"):
    para_map = {p["para_id"]: p for p in docs_index[q["doc_id"]]}
    oracle = [para_map[r] for r in q["refs"] if r in para_map and para_map[r]["text"]]
    if not oracle:
        contexts.append(None)
        continue
    hard_negs = get_hard_negatives(q["query"], q["doc_id"], set(q["refs"]))
    random.shuffle(hard_negs)
    contexts.append(hard_negs + oracle)  # oracle last (most-relevant-last)

del bge_model; torch.cuda.empty_cache()

print(f"Loading {LLM_PATH} bfloat16 across all GPUs for generation...")
gen_model = AutoModelForCausalLM.from_pretrained(
    LLM_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
)
gen_model.eval()

print("Generating rejected samples (greedy)...")
pairs, skipped = [], 0

for q, context in tqdm(zip(queries, contexts), total=len(queries), desc="  generating"):
    if context is None:
        skipped += 1; continue

    prompt = build_prompt(q["query"], context)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    prompt_text = llm_tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = llm_tok(prompt_text, return_tensors="pt").to("cuda:0")
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = gen_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=llm_tok.eos_token_id,
        )

    gen_text = llm_tok.decode(out[0][input_len:], skip_special_tokens=True)
    abst = extract_abstractive(gen_text)
    if not abst:
        skipped += 1; continue

    rl = rougel(abst, q["abstractive"])
    if rl >= ROUGEL_SKIP:
        skipped += 1; continue  # model already good — don't penalize

    gt_answer = json.dumps({"abstractive": q["abstractive"], "refs": q["refs"]}, ensure_ascii=False)
    pairs.append({
        "prompt":   prompt_text,
        "chosen":   gt_answer,
        "rejected": gen_text,
        "rougel":   round(rl, 4),
    })

print(f"  {len(pairs)} pairs created, {skipped} skipped")
if pairs:
    print(f"  Mean RougeL of rejected: {np.mean([p['rougel'] for p in pairs]):.3f}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(pairs, f, ensure_ascii=False, indent=2)
print(f"Saved to {OUTPUT_FILE}")
