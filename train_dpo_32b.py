"""
32B DPO training on top of Qwen3-32B-SFT-v1.
chosen = ground-truth abstractive, rejected = model's greedy output.
Input: dpo_pairs_32b.json (from gen_dpo_rejected.py)
Output: lora_output_dpo_32b → Qwen3-32B-DPO-v1
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json, random
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

BASE       = Path("/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther")
LLM_PATH   = str(BASE / "Qwen3-32B-SFT-v1")
PAIRS_FILE = str(BASE / "dpo_pairs_32b.json")
OUTPUT_DIR = str(BASE / "lora_output_dpo_32b")
MAX_SEQ_LEN = 4096
SEED        = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

print(f"Loading pairs from {PAIRS_FILE}...")
with open(PAIRS_FILE) as f:
    pairs = json.load(f)
for p in pairs:
    p.pop("rougel", None)
print(f"  {len(pairs)} pairs loaded")

random.shuffle(pairs)
val_size = max(30, int(len(pairs) * 0.1))
train_ds = Dataset.from_list(pairs[val_size:])
val_ds   = Dataset.from_list(pairs[:val_size])
print(f"  train={len(train_ds)}, val={len(val_ds)}")

llm_tok = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)

print("Loading 32B QLoRA NF4 for DPO...")
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    LLM_PATH, quantization_config=bnb_cfg, device_map="auto", trust_remote_code=True,
)
model.config.use_cache = False

lora_cfg = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
)

dpo_cfg = DPOConfig(
    output_dir=OUTPUT_DIR, num_train_epochs=1,
    per_device_train_batch_size=1, per_device_eval_batch_size=1,
    gradient_accumulation_steps=16, learning_rate=5e-5,
    lr_scheduler_type="cosine", warmup_ratio=0.1, bf16=True,
    gradient_checkpointing=True, logging_steps=10,
    save_strategy="epoch", eval_strategy="epoch",
    beta=0.1, max_length=MAX_SEQ_LEN,
    report_to="none", seed=SEED,
)

trainer = DPOTrainer(
    model=model, args=dpo_cfg,
    train_dataset=train_ds, eval_dataset=val_ds,
    peft_config=lora_cfg, processing_class=llm_tok,
)
print("DPO Training...")
trainer.train()
trainer.save_model(OUTPUT_DIR + "/final")
llm_tok.save_pretrained(OUTPUT_DIR + "/final")
print(f"Saved to {OUTPUT_DIR}/final")
