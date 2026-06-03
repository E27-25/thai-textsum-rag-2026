"""
DPO training for Qwen3-14B on top of Oracle-v4.
Steps:
  1. For each training query, generate N=6 abstractive answers (temp=0.7)
  2. Score each with RougeL vs ground-truth abstractive
  3. Create pairs: chosen=best RougeL, rejected=worst (filter: diff > 0.08)
  4. Train DPO with trl.DPOTrainer
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
from tqdm import tqdm
from rouge_score import rouge_scorer

BASE         = Path("/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther")
BGE_PATH     = str(BASE / "bge-m3")
LLM_PATH     = str(BASE / "Qwen3-14B-SFT-v3")
DATA_PATH    = BASE / "data" / "ชุดข้อมูล" / "train_set.json"
OUTPUT_DIR   = str(BASE / "lora_output_dpo")

TOP_K_RETRIEVE = 20
TOP_K_FINAL    = 7
N_SAMPLES      = 6      # answers to generate per query
ROUGEL_DIFF    = 0.08   # min difference to create a pair
MAX_PARA_CHARS = 800
MAX_SEQ_LEN    = 3072
SEED           = 42
BGE_BATCH      = 128

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

def rougel(hyp: str, ref: str) -> float:
    return scorer.score(ref, hyp)["rougeL"].fmeasure

print("Loading training data...")
with open(DATA_PATH) as f:
    data = json.load(f)
docs_index = {d["doc_id"]: d["paragraphs"] for d in data["docs"]}
queries    = data["queries"]
print(f"  {len(queries)} queries")

# ── BGE-M3 for retrieval ──────────────────────────────────────────────────────
print(f"Loading BGE-M3 on {device}...")
bge_tok   = AutoTokenizer.from_pretrained(BGE_PATH)
bge_model = AutoModel.from_pretrained(BGE_PATH, torch_dtype=torch.bfloat16).to(device)
bge_model.eval()

def encode_cls(texts: list[str], batch_size: int = BGE_BATCH) -> np.ndarray:
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = bge_tok(batch, padding=True, truncation=True,
                         max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            out = bge_model(**inputs)
        emb = out.last_hidden_state[:, 0, :]
        emb = F.normalize(emb, p=2, dim=1)
        all_emb.append(emb.float().cpu().numpy())
    return np.vstack(all_emb)

def encode_tokens(text: str) -> torch.Tensor:
    inputs = bge_tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = bge_model(**inputs)
    return F.normalize(out.last_hidden_state[0], p=2, dim=-1)

def colbert_score(q_embs: torch.Tensor, passage: str) -> float:
    p_embs = encode_tokens(passage[:MAX_PARA_CHARS])
    return (q_embs @ p_embs.T).max(dim=1).values.sum().item()

print("Building paragraph embeddings...")
doc_embs = {}
for doc_id, paras in tqdm(docs_index.items(), desc="  indexing"):
    valid = [p for p in paras if len(p["text"]) > 0]
    embs  = encode_cls([p["text"] for p in valid])
    doc_embs[doc_id] = (valid, embs)

def retrieve_and_rerank(query: str, doc_id: str) -> list[dict]:
    paras, para_embs = doc_embs[doc_id]
    q_emb  = encode_cls([query])[0]
    top_idx = np.argsort(para_embs @ q_emb)[::-1][:TOP_K_RETRIEVE].tolist()
    candidates = [paras[i] for i in top_idx]
    q_embs = encode_tokens(query)
    scores = [colbert_score(q_embs, p["text"]) for p in candidates]
    ranked = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    return [candidates[i] for i in ranked[:TOP_K_FINAL]]

def build_prompt(query: str, retrieved: list[dict]) -> str:
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

# ── Prepare retrieval contexts ────────────────────────────────────────────────
print("Retrieving contexts for all queries...")
query_contexts = []
for q in tqdm(queries, desc="  retrieve+rerank"):
    retrieved = retrieve_and_rerank(q["query"], q["doc_id"])
    query_contexts.append(retrieved)

del bge_model
torch.cuda.empty_cache()
print("BGE-M3 freed")

# ── Load LLM for pair generation ──────────────────────────────────────────────
print(f"Loading {LLM_PATH} for pair generation...")
llm_tok = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)

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

SYSTEM_PROMPT = (
    "คุณเป็นผู้ช่วยตอบคำถามจากบันทึกการประชุมภาษาไทย "
    "ตอบให้ครบถ้วน ชัดเจน และอ้างอิงเฉพาะข้อมูลจากย่อหน้าที่ให้มาเท่านั้น"
)

def make_chat_prompt(query: str, retrieved: list[dict]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_prompt(query, retrieved)},
    ]
    return llm_tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

import re

def extract_abstractive(text: str) -> str | None:
    try:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            obj = json.loads(m.group())
            abst = str(obj.get("abstractive", "")).strip()
            return abst if abst else None
    except Exception:
        pass
    return None

# ── Generate N samples per query ──────────────────────────────────────────────
print(f"Generating {N_SAMPLES} answers per query...")
pairs = []
skipped = 0

for q, retrieved in tqdm(zip(queries, query_contexts), total=len(queries), desc="  generating"):
    prompt_text = make_chat_prompt(q["query"], retrieved)
    inputs = llm_tok(prompt_text, return_tensors="pt").to(llm.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = llm.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=N_SAMPLES,
            pad_token_id=llm_tok.eos_token_id,
        )

    candidates = []
    for seq in out:
        gen_text = llm_tok.decode(seq[input_len:], skip_special_tokens=True)
        abst = extract_abstractive(gen_text)
        if abst:
            score = rougel(abst, q["abstractive"])
            candidates.append((score, abst, gen_text))

    if len(candidates) < 2:
        skipped += 1
        continue

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score,  best_abst,  best_full  = candidates[0]
    worst_score, worst_abst, worst_full = candidates[-1]

    if best_score - worst_score < ROUGEL_DIFF:
        skipped += 1
        continue

    pairs.append({
        "prompt":   prompt_text,
        "chosen":   best_full,
        "rejected": worst_full,
    })

print(f"  {len(pairs)} pairs created, {skipped} skipped")

del llm
torch.cuda.empty_cache()

# ── DPO Training ──────────────────────────────────────────────────────────────
random.shuffle(pairs)
val_size = max(30, int(len(pairs) * 0.1))
train_ds = Dataset.from_list(pairs[val_size:])
val_ds   = Dataset.from_list(pairs[:val_size])
print(f"  DPO train={len(train_ds)}, val={len(val_ds)}")

print("Loading model for DPO...")
model = AutoModelForCausalLM.from_pretrained(
    LLM_PATH,
    quantization_config=bnb_cfg,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False

lora_cfg = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

dpo_cfg = DPOConfig(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs             = 1,
    per_device_train_batch_size  = 1,
    per_device_eval_batch_size   = 1,
    gradient_accumulation_steps  = 16,
    learning_rate                = 5e-5,
    lr_scheduler_type            = "cosine",
    warmup_ratio                 = 0.1,
    bf16                         = True,
    gradient_checkpointing       = True,
    logging_steps                = 10,
    save_strategy                = "epoch",
    eval_strategy                = "epoch",
    beta                         = 0.1,
    max_length                   = MAX_SEQ_LEN,
    report_to                    = "none",
    seed                         = SEED,
)

trainer = DPOTrainer(
    model      = model,
    args       = dpo_cfg,
    train_dataset = train_ds,
    eval_dataset  = val_ds,
    peft_config   = lora_cfg,
    processing_class = llm_tok,
)

print("DPO Training...")
trainer.train()
trainer.save_model(OUTPUT_DIR + "/final")
llm_tok.save_pretrained(OUTPUT_DIR + "/final")
print(f"DPO adapter saved to {OUTPUT_DIR}/final")
