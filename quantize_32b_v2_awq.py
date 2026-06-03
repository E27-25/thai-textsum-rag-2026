"""AWQ 4-bit quantize Qwen3-32B-SFT-v2 → Qwen3-32B-SFT-v2-AWQ"""
import warnings, json, shutil
warnings.filterwarnings("ignore")

from pathlib import Path
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

BASE        = Path("/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther")
MODEL_PATH  = str(BASE / "Qwen3-32B-SFT-v2")
QUANT_PATH  = str(BASE / "Qwen3-32B-SFT-v2-AWQ")
DATA_PATH   = BASE / "data" / "ชุดข้อมูล" / "train_set.json"

quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

print("Loading calibration data...")
with open(DATA_PATH) as f:
    data = json.load(f)
calib_data = [q["abstractive"] for q in data["queries"][:128]]
print(f"  {len(calib_data)} calibration samples")

print(f"Loading {MODEL_PATH} for AWQ quantization...")
model = AutoAWQForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", safetensors=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("Quantizing...")
model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)

print(f"Saving to {QUANT_PATH} ...")
model.save_quantized(QUANT_PATH)

base_model = str(BASE / "Qwen3-32B")
for fname in ["tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt"]:
    src = Path(base_model) / fname
    if src.exists():
        shutil.copy(src, Path(QUANT_PATH) / fname)
print("Done!")
