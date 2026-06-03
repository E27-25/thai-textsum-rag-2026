"""
QLoRA SFT v5 for Qwen3-14B on Thai meeting-minutes dataset.
Key changes from v4 Oracle:
  - Add system prompt to training messages (v4 had none → mismatch with inference)
  - Paragraph ordering: hard-negatives first, oracle refs last (matches inference "most-relevant-last")
  - Hard negatives: 3 → 7 (matches inference TOP_K_FINAL=10 better)
  - 4 epochs (up from 3)
  - Output: lora_output_v5 → Qwen3-14B-SFT-v5
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
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from tqdm import tqdm

BASE        = Path("/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther")
BGE_PATH    = str(BASE / "bge-m3")
LLM_PATH    = str(BASE / "Qwen3-14B")
DATA_PATH   = BASE / "data" / "ชุดข้อมูล" / "train_set.json"
OUTPUT_DIR  = str(BASE / "lora_output_v5")

N_DISTRACTORS  = 7     # was 3 — match inference TOP_K_FINAL=10
MAX_PARA_CHARS = 800
MAX_SEQ_LEN    = 4096
SEED           = 42
BGE_BATCH      = 128

# Must match run_vllm.py exactly
SYSTEM_PROMPT = (
    "คุณเป็นผู้ช่วยตอบคำถามจากบันทึกการประชุมภาษาไทย "
    "ตอบให้ครบถ้วนและชัดเจน โดยใช้ถ้อยคำและสำนวนจากย่อหน้าที่ให้มาโดยตรง "
    "อ้างอิงเฉพาะข้อมูลจากย่อหน้าที่ให้มาเท่านั้น "
    "หากคำตอบกระจายอยู่หลายย่อหน้าให้ระบุทุก ID ที่เกี่ยวข้อง"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

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

print("Building paragraph embeddings for hard negative mining...")
doc_embs = {}
for doc_id, paras in tqdm(docs_index.items(), desc="  indexing"):
    valid_paras = [p for p in paras if len(p["text"]) > 0]
    texts = [p["text"] for p in valid_paras]
    embs  = encode_cls(texts)
    doc_embs[doc_id] = (valid_paras, embs)
print("  done")

def get_hard_negatives(query_text: str, doc_id: str,
                       correct_ref_ids: set[str], n: int = N_DISTRACTORS) -> list[dict]:
    paras, para_embs = doc_embs[doc_id]
    q_emb  = encode_cls([query_text])[0]
    scores = para_embs @ q_emb
    sorted_idx = np.argsort(scores)[::-1]
    negatives = []
    for i in sorted_idx:
        if paras[i]["para_id"] not in correct_ref_ids:
            negatives.append(paras[i])
            if len(negatives) >= n:
                break
    return negatives

def build_prompt(query: str, context_paras: list[dict]) -> str:
    # Must exactly match run_vllm.py build_prompt()
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
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)

examples = []
skipped  = 0
for q in tqdm(queries, desc="  preparing"):
    doc_id   = q["doc_id"]
    para_map = {p["para_id"]: p for p in docs_index[doc_id]}

    oracle_paras = [para_map[r] for r in q["refs"] if r in para_map and len(para_map[r]["text"]) > 0]
    if not oracle_paras:
        skipped += 1
        continue

    correct_ref_ids = set(q["refs"])
    hard_negs = get_hard_negatives(q["query"], doc_id, correct_ref_ids)

    # Hard negatives first, oracle refs last (matches inference "most-relevant-last" order)
    # Shuffle hard negatives among themselves for variety
    random.shuffle(hard_negs)
    context_paras = hard_negs + oracle_paras

    prompt = build_prompt(q["query"], context_paras)
    answer = json.dumps(
        {"abstractive": q["abstractive"], "refs": q["refs"]},
        ensure_ascii=False
    )

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": prompt},
        {"role": "assistant", "content": answer},
    ]
    text = llm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    if len(llm_tokenizer.encode(text)) > MAX_SEQ_LEN:
        skipped += 1
        continue
    examples.append({"messages": messages})

print(f"  {len(examples)} examples kept, {skipped} skipped")

del bge_model
torch.cuda.empty_cache()

random.shuffle(examples)
val_size       = max(50, int(len(examples) * 0.1))
train_examples = examples[val_size:]
val_examples   = examples[:val_size]
train_ds = Dataset.from_list(train_examples)
val_ds   = Dataset.from_list(val_examples)
print(f"  train={len(train_ds)}, val={len(val_ds)}")

print("Loading model (QLoRA 4-bit)...")
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    LLM_PATH,
    quantization_config=bnb_cfg,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False

lora_cfg = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

sft_cfg = SFTConfig(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs             = 4,
    per_device_train_batch_size  = 1,
    per_device_eval_batch_size   = 1,
    eval_accumulation_steps      = 1,
    gradient_accumulation_steps  = 16,
    learning_rate                = 2e-4,
    lr_scheduler_type            = "cosine",
    warmup_ratio                 = 0.1,
    bf16                         = True,
    gradient_checkpointing       = True,
    logging_steps                = 10,
    save_strategy                = "epoch",
    eval_strategy                = "epoch",
    load_best_model_at_end       = True,
    metric_for_best_model        = "eval_loss",
    greater_is_better            = False,
    max_grad_norm                = 1.0,
    dataloader_num_workers       = 4,
    report_to                    = "none",
    seed                         = SEED,
    max_length                   = MAX_SEQ_LEN,
)

trainer = SFTTrainer(
    model            = model,
    args             = sft_cfg,
    train_dataset    = train_ds,
    eval_dataset     = val_ds,
    peft_config      = lora_cfg,
    processing_class = llm_tokenizer,
)

print("Training...")
trainer.train()
trainer.save_model(OUTPUT_DIR + "/final")
llm_tokenizer.save_pretrained(OUTPUT_DIR + "/final")
print(f"LoRA adapter saved to {OUTPUT_DIR}/final")
