"""
32B Oracle SFT v2:
  - Correct system prompt (matches inference)
  - Oracle refs last (most-relevant-last)
  - Only 2 distractors (light noise, cleaner style learning)
  - 4 epochs
  - Output: lora_output_32b_oracle → Qwen3-32B-Oracle-v1
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json, random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from tqdm import tqdm

BASE        = Path("/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther")
BGE_PATH    = str(BASE / "bge-m3")
LLM_PATH    = str(BASE / "Qwen3-32B")
DATA_PATH   = BASE / "data" / "ชุดข้อมูล" / "train_set.json"
OUTPUT_DIR  = str(BASE / "lora_output_32b_oracle")

N_DISTRACTORS  = 2
MAX_PARA_CHARS = 800
MAX_SEQ_LEN    = 4096
SEED           = 42
BGE_BATCH      = 128

SYSTEM_PROMPT = (
    "คุณเป็นผู้ช่วยตอบคำถามจากบันทึกการประชุมภาษาไทย "
    "ตอบให้ครบถ้วนและชัดเจน โดยใช้ถ้อยคำและสำนวนจากย่อหน้าที่ให้มาโดยตรง "
    "อ้างอิงเฉพาะข้อมูลจากย่อหน้าที่ให้มาเท่านั้น "
    "หากคำตอบกระจายอยู่หลายย่อหน้าให้ระบุทุก ID ที่เกี่ยวข้อง"
)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

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

print("Building embeddings for hard negative mining...")
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

print("Building training examples...")
llm_tok = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)
examples, skipped = [], 0
for q in tqdm(queries, desc="  preparing"):
    para_map = {p["para_id"]: p for p in docs_index[q["doc_id"]]}
    oracle = [para_map[r] for r in q["refs"] if r in para_map and para_map[r]["text"]]
    if not oracle: skipped += 1; continue
    hard_negs = get_hard_negatives(q["query"], q["doc_id"], set(q["refs"]))
    random.shuffle(hard_negs)
    context = hard_negs + oracle   # oracle last
    prompt = build_prompt(q["query"], context)
    answer = json.dumps({"abstractive": q["abstractive"], "refs": q["refs"]}, ensure_ascii=False)
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": prompt},
        {"role": "assistant", "content": answer},
    ]
    text = llm_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if len(llm_tok.encode(text)) > MAX_SEQ_LEN: skipped += 1; continue
    examples.append({"messages": messages})
print(f"  {len(examples)} kept, {skipped} skipped")

del bge_model; torch.cuda.empty_cache()

random.shuffle(examples)
val_size = max(50, int(len(examples) * 0.1))
train_ds = Dataset.from_list(examples[val_size:])
val_ds   = Dataset.from_list(examples[:val_size])
print(f"  train={len(train_ds)}, val={len(val_ds)}")

print("Loading 32B QLoRA 4-bit...")
bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(LLM_PATH, quantization_config=bnb_cfg,
                                              device_map="auto", trust_remote_code=True)
model.config.use_cache = False

lora_cfg = LoraConfig(r=32, lora_alpha=64,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)

sft_cfg = SFTConfig(
    output_dir=OUTPUT_DIR, num_train_epochs=4,
    per_device_train_batch_size=1, per_device_eval_batch_size=1,
    gradient_accumulation_steps=16, learning_rate=1e-4,
    lr_scheduler_type="cosine", warmup_ratio=0.1, bf16=True,
    gradient_checkpointing=True, logging_steps=10,
    save_strategy="epoch", eval_strategy="epoch",
    load_best_model_at_end=True, metric_for_best_model="eval_loss",
    greater_is_better=False, max_grad_norm=1.0,
    dataloader_num_workers=4, report_to="none", seed=SEED, max_length=MAX_SEQ_LEN,
)

trainer = SFTTrainer(model=model, args=sft_cfg, train_dataset=train_ds,
                     eval_dataset=val_ds, peft_config=lora_cfg, processing_class=llm_tok)
print("Training...")
trainer.train()
trainer.save_model(OUTPUT_DIR + "/final")
llm_tok.save_pretrained(OUTPUT_DIR + "/final")
print(f"Saved to {OUTPUT_DIR}/final")
