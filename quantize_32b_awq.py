"""
AWQ 4-bit quantize Qwen3-32B-SFT-v1 → Qwen3-32B-SFT-v1-AWQ
Reduces ~64GB bfloat16 model to ~16GB for container fit (<60GB limit).
Uses local training data as calibration (no internet needed on compute node).
"""
import warnings, json
warnings.filterwarnings("ignore")

from pathlib import Path
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

BASE        = Path("/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther")
MODEL_PATH  = str(BASE / "Qwen3-32B-SFT-v1")
QUANT_PATH  = str(BASE / "Qwen3-32B-SFT-v1-AWQ")
DATA_PATH   = BASE / "data" / "ชุดข้อมูล" / "train_set.json"

quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
}

# Use local training data as calibration dataset (no HuggingFace download needed)
print("Loading calibration data from local training set...")
with open(DATA_PATH) as f:
    data = json.load(f)
calib_data = [q["abstractive"] for q in data["queries"][:128]]
print(f"  {len(calib_data)} calibration samples ready")

print(f"Loading {MODEL_PATH} for AWQ quantization...")
model = AutoAWQForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",   # spread across all 4 GPUs (4×40GB=160GB, model=62GB)
    safetensors=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("Quantizing (this takes ~30-60 min)...")
model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)

print(f"Saving AWQ model to {QUANT_PATH} ...")
model.save_quantized(QUANT_PATH)
# Use base model tokenizer files — AWQ saves a broken tokenizer_config.json
import shutil
base_model = str(Path("/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther") / "Qwen3-32B")
for fname in ["tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt"]:
    src = Path(base_model) / fname
    if src.exists():
        shutil.copy(src, Path(QUANT_PATH) / fname)
print("Done!")
