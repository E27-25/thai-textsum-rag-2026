"""
Generate Thai paraphrases of gt_abst for SFT v3 data augmentation.
  - Uses base Qwen3-32B with transformers + 4-bit BitsAndBytes (vLLM broken in env)
  - 2 paraphrases per query, temperature=0.7
  - Batched generation for speed
  - Output: paraphrases_v3.json
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json, re
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

BASE       = Path("/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther")
LLM_PATH   = str(BASE / "Qwen3-32B")
DATA_PATH  = BASE / "data" / "ชุดข้อมูล" / "train_set.json"
OUT_PATH   = BASE / "paraphrases_v3.json"

N_PARAPHRASES = 2
TEMPERATURE   = 0.7
TOP_P         = 0.9
MAX_TOKENS    = 200
BATCH_SIZE    = 8
SEED          = 42

SYSTEM = (
    "คุณเป็นผู้ช่วยภาษาไทย หน้าที่ของคุณคือเปลี่ยนสำนวนของข้อความที่ให้มา "
    "โดยรักษาข้อเท็จจริง ตัวเลข ชื่อบุคคล ชื่อสถานที่ และ ID ต่าง ๆ ไว้ครบถ้วน "
    "แต่ใช้คำและโครงสร้างประโยคแตกต่างจากต้นฉบับ"
)

def build_prompt(abst, query):
    return (
        f"คำถามจากบันทึกการประชุม: {query}\n\n"
        f"คำตอบต้นฉบับ: {abst}\n\n"
        f"ให้เขียนคำตอบใหม่ที่มีความหมายเหมือนเดิม รักษาข้อเท็จจริงทั้งหมด "
        f"(ตัวเลข ชื่อ สถานที่ ID) แต่เปลี่ยนสำนวน ลำดับ และคำที่ใช้\n\n"
        f"คำตอบใหม่:"
    )

torch.manual_seed(SEED)

print("Loading training data...")
with open(DATA_PATH) as f:
    data = json.load(f)
queries = data["queries"]
print(f"  {len(queries)} queries")

print("Loading Qwen3-32B 4-bit (BitsAndBytes NF4)...")
bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    LLM_PATH, quantization_config=bnb_cfg, device_map="auto", trust_remote_code=True,
)
model.eval()

print("Building prompts...")
all_prompts, prompt_index = [], []
for q in queries:
    for k in range(N_PARAPHRASES):
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": build_prompt(q["abstractive"], q["query"])},
        ]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        all_prompts.append(text)
        prompt_index.append((q["ID"], k))
print(f"  {len(all_prompts)} prompts ({len(queries)} queries × {N_PARAPHRASES} variants)")

def clean(text):
    text = text.strip()
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text).strip()
    text = re.sub(r"^(คำตอบใหม่[:：]|คำตอบ[:：])\s*", "", text)
    text = text.split("\n")[0].strip()
    return text

def safe_save(result):
    tmp = OUT_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    tmp.replace(OUT_PATH)

result = {}
print(f"Generating (batch={BATCH_SIZE}, temperature={TEMPERATURE})...")
with torch.inference_mode():
    for i in tqdm(range(0, len(all_prompts), BATCH_SIZE), desc="paraphrase"):
        batch_prompts = all_prompts[i:i+BATCH_SIZE]
        batch_index   = prompt_index[i:i+BATCH_SIZE]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True,
                           max_length=1024).to(model.device)
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_TOKENS,
            do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
            pad_token_id=tokenizer.pad_token_id,
        )
        input_len = inputs["input_ids"].shape[1]
        for (qid, k), out in zip(batch_index, outputs):
            gen_text = tokenizer.decode(out[input_len:], skip_special_tokens=True)
            para = clean(gen_text)
            result.setdefault(qid, []).append(para)
        if (i // BATCH_SIZE) % 20 == 0:
            safe_save(result)

safe_save(result)
print(f"Wrote {len(result)} entries to {OUT_PATH}")
example_qid = next(iter(result))
print(f"  Example ({example_qid}): {result[example_qid]}")
