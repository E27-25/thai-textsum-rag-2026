#!/usr/bin/env python3
"""
Run 14B SFT-v3 inference on training data → submission_train.csv
Uses HF transformers (4-bit) instead of vLLM to avoid cu130 symbol mismatch.
"""

import json, re, csv, os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

BASE      = Path("/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther")
BGE_PATH  = str(BASE / "bge-m3")
LLM_PATH  = str(BASE / "Qwen3-14B-SFT-v3")
DATA_PATH = BASE / "data" / "ชุดข้อมูล" / "train_set.json"
OUT_CSV   = str(BASE / "submission_train.csv")

TOP_K_RETRIEVE = 20
TOP_K_FINAL    = 7
MAX_PARA_CHARS = 800
BGE_BATCH      = 64
GEN_BATCH      = 4
MAX_NEW_TOKENS = 512

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}", flush=True)

SYSTEM_PROMPT = (
    "คุณเป็นผู้ช่วยตอบคำถามจากบันทึกการประชุมภาษาไทย "
    "ตอบให้ครบถ้วน ชัดเจน และอ้างอิงเฉพาะข้อมูลจากย่อหน้าที่ให้มาเท่านั้น"
)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading training data...", flush=True)
with open(DATA_PATH) as f:
    data = json.load(f)
docs_index = {d["doc_id"]: d["paragraphs"] for d in data["docs"]}
queries    = data["queries"]
print(f"  {len(queries)} queries, {len(docs_index)} docs", flush=True)

# ── BGE-M3 retrieval ──────────────────────────────────────────────────────────
print("Loading BGE-M3...", flush=True)
bge_tok   = AutoTokenizer.from_pretrained(BGE_PATH)
bge_model = AutoModel.from_pretrained(BGE_PATH, torch_dtype=torch.bfloat16).to(device)
bge_model.eval()

def encode_cls(texts, batch_size=BGE_BATCH):
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inp = bge_tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            out = bge_model(**inp)
        emb = F.normalize(out.last_hidden_state[:, 0, :], p=2, dim=1)
        all_emb.append(emb.float().cpu().numpy())
    return np.vstack(all_emb)

def encode_tokens(text):
    inp = bge_tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = bge_model(**inp)
    return F.normalize(out.last_hidden_state[0], p=2, dim=-1)

print("Building paragraph embeddings...", flush=True)
doc_embs = {}
for doc_id, paras in tqdm(docs_index.items(), desc="  indexing"):
    valid = [p for p in paras if p["text"]]
    embs  = encode_cls([p["text"] for p in valid])
    doc_embs[doc_id] = (valid, embs)

def retrieve(query, doc_id):
    paras, para_embs = doc_embs[doc_id]
    q_emb   = encode_cls([query])[0]
    top_idx = np.argsort(para_embs @ q_emb)[::-1][:TOP_K_RETRIEVE].tolist()
    cands   = [paras[i] for i in top_idx]
    q_embs  = encode_tokens(query)
    scores  = [(q_embs @ encode_tokens(p["text"][:MAX_PARA_CHARS]).T).max(dim=1).values.sum().item()
               for p in cands]
    ranked  = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)
    return [cands[i] for i in ranked[:TOP_K_FINAL]]

print("Retrieving contexts...", flush=True)
contexts = []
for q in tqdm(queries, desc="  retrieve+rerank"):
    contexts.append(retrieve(q["query"], q["doc_id"]))

del bge_model
torch.cuda.empty_cache()
print(f"BGE-M3 freed | VRAM: {torch.cuda.memory_allocated(0)/1e9:.1f} GB", flush=True)

# ── Build prompts ─────────────────────────────────────────────────────────────
def build_prompt(query, retrieved):
    ordered = list(reversed(retrieved))
    context = "\n".join(f"[{p['para_id']}] {p['text'][:MAX_PARA_CHARS]}" for p in ordered)
    ids = ", ".join(f'"{p["para_id"]}"' for p in ordered)
    return (
        f"ย่อหน้าจากบันทึกการประชุม:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"ให้ตอบคำถามโดยสรุปคำตอบจากย่อหน้าข้างต้น และระบุ ID ย่อหน้าที่เกี่ยวข้อง\n"
        f"ID ที่เลือกได้: {ids}\n\n"
        f"ตอบเป็น JSON เท่านั้น:\n"
        f'{{"abstractive": "สรุปคำตอบภาษาไทย", "refs": ["id1", "id2"]}}'
    )

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
    result = set(refs)
    for i in range(len(nums) - 1):
        n1, pfx, _ = nums[i]
        n2, _, _   = nums[i+1]
        if n2 - n1 == 2:
            mid = f"{pfx}{n1+1}"
            if mid in valid_ids:
                result.add(mid)
    return sorted(result, key=lambda x: int(re.search(r'\d+', x).group()))

def parse_output(text, fallback, valid_ids):
    try:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            obj = json.loads(m.group())
            abst = str(obj.get("abstractive", "")).strip()
            refs = [str(r) for r in obj.get("refs", [])]
            if abst and refs:
                return abst, expand_contiguous(refs, valid_ids)
    except Exception:
        pass
    return text.strip(), [fallback]

print(f"Loading {LLM_PATH} (4-bit)...", flush=True)
tok = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
llm = AutoModelForCausalLM.from_pretrained(
    LLM_PATH,
    quantization_config=bnb_cfg,
    device_map="auto",
    trust_remote_code=True,
)
llm.eval()
print(f"Model loaded | VRAM: {torch.cuda.memory_allocated(0)/1e9:.1f} GB", flush=True)

# Build chat prompts
prompts = []
for q, retrieved in zip(queries, contexts):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_prompt(q["query"], retrieved)},
    ]
    prompts.append(tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    ))

# ── Batch inference ───────────────────────────────────────────────────────────
print(f"Generating {len(prompts)} answers (batch={GEN_BATCH})...", flush=True)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id

gen_texts = []
for i in tqdm(range(0, len(prompts), GEN_BATCH), desc="  generate"):
    batch = prompts[i : i + GEN_BATCH]
    enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
              max_length=4096).to(llm.device)
    input_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = llm.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tok.pad_token_id,
        )
    for seq in out:
        gen_texts.append(tok.decode(seq[input_len:], skip_special_tokens=True))

# ── Write CSV ─────────────────────────────────────────────────────────────────
rows = []
for q, retrieved, gen_text in zip(queries, contexts, gen_texts):
    fallback  = retrieved[0]["para_id"] if retrieved else "P1"
    valid_ids = {p["para_id"] for p in docs_index[q["doc_id"]]}
    abst, refs = parse_output(gen_text, fallback, valid_ids)
    rows.append({"ID": q["ID"], "abstractive": abst, "refs": ",".join(refs)})

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["ID", "abstractive", "refs"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} rows → {OUT_CSV}", flush=True)
print("Run eval_train.py next to score + see worst cases.", flush=True)
