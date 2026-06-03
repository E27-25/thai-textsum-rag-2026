"""
Merge LoRA adapter into base Qwen3-14B weights.
Run AFTER train_sft.py completes.
Output: Qwen3-14B-SFT/ (same directory structure as Qwen3-14B/)
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE       = Path("/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther")
LLM_PATH   = str(BASE / "Qwen3-14B")
LORA_PATH  = str(BASE / "lora_output_v3" / "final")
MERGED_PATH = str(BASE / "Qwen3-14B-SFT-v3")

print(f"Loading base model from {LLM_PATH} (CPU, bfloat16)...")
model = AutoModelForCausalLM.from_pretrained(
    LLM_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
)

print(f"Loading LoRA adapter from {LORA_PATH}...")
model = PeftModel.from_pretrained(model, LORA_PATH)

print("Merging LoRA into base weights...")
model = model.merge_and_unload()

print(f"Saving merged model to {MERGED_PATH}...")
model.save_pretrained(MERGED_PATH, safe_serialization=True, max_shard_size="5GB")

tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)
tokenizer.save_pretrained(MERGED_PATH)

print(f"""
Done! Merged model saved to: {MERGED_PATH}

Next steps to rebuild Docker image:
  1. Update Dockerfile: change COPY line to use Qwen3-14B-SFT/
       COPY Qwen3-14B-SFT/ ./Qwen3-14B/
  2. Add Qwen3-14B-SFT to .dockerignore.35b
  3. bash lanta_build_push.sh > build_14b_sft.log 2>&1 &
""")
