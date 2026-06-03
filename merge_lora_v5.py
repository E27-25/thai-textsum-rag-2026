import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE        = Path("/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther")
LLM_PATH    = str(BASE / "Qwen3-14B")
LORA_PATH   = str(BASE / "lora_output_v5" / "final")
MERGED_PATH = str(BASE / "Qwen3-14B-SFT-v5")

print(f"Loading base model from {LLM_PATH} (CPU, bfloat16)...")
model = AutoModelForCausalLM.from_pretrained(
    LLM_PATH, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
)
print(f"Loading LoRA adapter from {LORA_PATH}...")
model = PeftModel.from_pretrained(model, LORA_PATH)
print("Merging and unloading...")
model = model.merge_and_unload()
print(f"Saving to {MERGED_PATH} ...")
model.save_pretrained(MERGED_PATH, safe_serialization=True, max_shard_size="5GB")
tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)
tokenizer.save_pretrained(MERGED_PATH)
print("Done!")
