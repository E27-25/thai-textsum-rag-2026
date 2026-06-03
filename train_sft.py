"""
QLoRA SFT v3 for Qwen3-14B on Thai meeting-minutes dataset.
Changes from v2:
  - Use AutoModel (same as inference) instead of SentenceTransformer
  - 2-stage retrieval: dense top-20 → ColBERT rerank → top-7 (matches inference)
  - Force-include missing refs into top-20 pool before reranking
  - Filter empty paragraphs (len==0) only — short paras like member names are valid refs
  - 3 epochs (up from 2)
  - Output: lora_output_v3 → Qwen3-14B-SFT-v3
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

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE        = Path("/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther")
BGE_PATH    = str(BASE / "bge-m3")
LLM_PATH    = str(BASE / "Qwen3-14B")
DATA_PATH   = BASE / "data" / "ชุดข้อมูล" / "train_set.json"
OUTPUT_DIR  = str(BASE / "lora_output_v3")

TOP_K_RETRIEVE = 20   # dense retrieval candidates
TOP_K_FINAL    = 7    # after ColBERT rerank (matches inference)
MAX_PARA_CHARS = 800
MAX_SEQ_LEN    = 4096
SEED           = 42
BGE_BATCH      = 128

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading training data...")
with open(DATA_PATH) as f:
    data = json.load(f)

docs_index = {d["doc_id"]: d["paragraphs"] for d in data["docs"]}
queries    = data["queries"]
print(f"  {len(queries)} queries across {len(docs_index)} documents")

# ── 2. Load BGE-M3 (AutoModel, same as inference) ────────────────────────────
print(f"Loading BGE-M3 on {device}...")
bge_tok   = AutoTokenizer.from_pretrained(BGE_PATH)
bge_model = AutoModel.from_pretrained(BGE_PATH, torch_dtype=torch.bfloat16).to(device)
bge_model.eval()

def encode_cls(texts: list[str], batch_size: int = BGE_BATCH) -> np.ndarray:
    """CLS-token dense embeddings (L2 normalized)."""
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
    """All token embeddings for ColBERT late-interaction (L2 normalized)."""
    inputs = bge_tok(text, return_tensors="pt", truncation=True,
                     max_length=512).to(device)
    with torch.no_grad():
        out = bge_model(**inputs)
    embs = out.last_hidden_state[0]
    return F.normalize(embs, p=2, dim=-1)

def colbert_score(q_embs: torch.Tensor, passage: str) -> float:
    """ColBERT late-interaction: sum of per-query-token max similarities."""
    p_embs = encode_tokens(passage[:MAX_PARA_CHARS])
    return (q_embs @ p_embs.T).max(dim=1).values.sum().item()

# ── 3. Build dense indexes per document ──────────────────────────────────────
print("Building paragraph embeddings...")
doc_embs = {}
for doc_id, paras in tqdm(docs_index.items(), desc="  indexing"):
    # filter truly empty paragraphs only
    valid_paras = [p for p in paras if len(p["text"]) > 0]
    texts = [p["text"] for p in valid_paras]
    embs  = encode_cls(texts)
    doc_embs[doc_id] = (valid_paras, embs)
print("  done")

# ── 4. Retrieve + rerank helper ───────────────────────────────────────────────
def retrieve_and_rerank(query_text: str, doc_id: str,
                        correct_refs: list[str]) -> list[dict]:
    paras, para_embs = doc_embs[doc_id]

    # Dense top-20
    q_emb  = encode_cls([query_text])[0]
    scores = para_embs @ q_emb
    top_idx = np.argsort(scores)[::-1][:TOP_K_RETRIEVE].tolist()
    candidates = [paras[i] for i in top_idx]

    # Force-include up to 2 missing correct refs into the candidate pool
    cand_ids = {p["para_id"] for p in candidates}
    missing  = [r for r in correct_refs if r not in cand_ids]
    random.shuffle(missing)
    for ref in missing[:2]:
        for p in paras:
            if p["para_id"] == ref:
                candidates[-1] = p  # replace weakest dense slot
                cand_ids.add(ref)
                break

    # ColBERT rerank → top-7
    q_embs = encode_tokens(query_text)
    col_scores = [colbert_score(q_embs, p["text"]) for p in candidates]
    ranked = sorted(range(len(candidates)), key=lambda i: col_scores[i], reverse=True)
    return [candidates[i] for i in ranked[:TOP_K_FINAL]]


def build_prompt(query: str, retrieved: list[dict]) -> str:
    context = "\n".join(
        f"[{p['para_id']}] {p['text'][:MAX_PARA_CHARS]}" for p in retrieved
    )
    ids = ", ".join(f'"{p["para_id"]}"' for p in retrieved)
    return (
        f"ย่อหน้าจากบันทึกการประชุม:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"ID ย่อหน้าที่มี: {ids}\n"
        f"ตอบเป็น JSON ภาษาไทย เลือก refs จาก ID ข้างต้นเท่านั้น:\n"
        f'{{"abstractive": "<สรุปคำตอบภาษาไทย>", "refs": ["id1", "id2"]}}'
    )

# ── 5. Build dataset ──────────────────────────────────────────────────────────
print("Building training examples (retrieve + rerank)...")
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)

examples = []
skipped  = 0
for q in tqdm(queries, desc="  preparing"):
    retrieved     = retrieve_and_rerank(q["query"], q["doc_id"], q["refs"])
    retrieved_ids = {p["para_id"] for p in retrieved}

    visible_refs = [r for r in q["refs"] if r in retrieved_ids]
    if not visible_refs:
        visible_refs = [retrieved[0]["para_id"]]

    prompt = build_prompt(q["query"], retrieved)
    answer = json.dumps(
        {"abstractive": q["abstractive"], "refs": visible_refs},
        ensure_ascii=False
    )

    messages = [
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

print(f"  {len(examples)} examples kept, {skipped} skipped (too long)")

# Free BGE-M3 VRAM before loading LLM
del bge_model
torch.cuda.empty_cache()

random.shuffle(examples)
val_size       = max(50, int(len(examples) * 0.1))
train_examples = examples[val_size:]
val_examples   = examples[:val_size]
train_ds = Dataset.from_list(train_examples)
val_ds   = Dataset.from_list(val_examples)
print(f"  train={len(train_ds)}, val={len(val_ds)}")

# ── 6. Load model (QLoRA 4-bit) ───────────────────────────────────────────────
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

# ── 7. LoRA config ────────────────────────────────────────────────────────────
lora_cfg = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# ── 8. Training ───────────────────────────────────────────────────────────────
sft_cfg = SFTConfig(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs             = 3,
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
