#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -N 1 -c 16
#SBATCH -t 01:00:00
#SBATCH -A zz991016
#SBATCH -J vllm_test
#SBATCH -o vllm_test-%j.out

set -e
cd /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther

source env/bin/activate

# Install vllm if not already installed
python -c "import vllm" 2>/dev/null || pip install vllm --quiet

echo "=== [$(date)] GPU info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== [$(date)] Running vLLM inference on train set ==="

python - <<'EOF'
import os, json, re, csv, time, torch, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

BGE_PATH  = "./bge-m3"
LLM_PATH  = "./Qwen3-14B"
DATA_PATH = "data/ชุดข้อมูล/train_set.json"
OUT_CSV   = "submission_vllm.csv"
TOP_K     = 7
MAX_TOKENS = 400
GPU       = "cuda:0"
BGE_BATCH = 128

class BGERetriever:
    def __init__(self, path, device):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
        self.model.eval()
    def _encode(self, texts, bs, desc=""):
        embs = []
        for i in tqdm(range(0, len(texts), bs), desc=desc, leave=False, dynamic_ncols=False):
            inp = self.tok(texts[i:i+bs], padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**inp)
            embs.append(F.normalize(out.last_hidden_state[:,0,:], p=2, dim=1).cpu())
        return torch.cat(embs)
    def build_index(self, paras, doc_id=""):
        return self._encode([p["text"] for p in paras], BGE_BATCH, f"  idx {doc_id}")
    def retrieve(self, q, paras, index, k):
        qe = self._encode([q], 1)
        idx = (index @ qe.T).squeeze(1).topk(min(k, len(paras))).indices.tolist()
        return [paras[i] for i in idx]

def build_prompt(query, retrieved):
    ctx = "\n".join(f"[{p['para_id']}] {p['text']}" for p in retrieved)
    return (f"ย่อหน้าจากบันทึกการประชุม:\n{ctx}\n\nคำถาม: {query}\n\n"
            f'ตอบเป็น JSON เท่านั้น:\n{{"abstractive": "...", "refs": ["P1"]}}')

def parse_output(text, fallback):
    try:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            obj = json.loads(m.group())
            a = str(obj.get("abstractive","")).strip()
            r = [str(x) for x in obj.get("refs",[])]
            if a and r: return a, r
    except: pass
    return text.strip(), [fallback]

t0 = time.time()

with open(DATA_PATH, encoding="utf-8") as f:
    data = json.load(f)
doc_map = {d["doc_id"]: d for d in data["docs"]}
queries = data["queries"]
print(f"Loaded {len(queries)} queries, {len(doc_map)} docs", flush=True)

ret = BGERetriever(BGE_PATH, GPU)
idx_map = {}
for did, doc in tqdm(doc_map.items(), desc="BGE index", dynamic_ncols=False):
    idx_map[did] = (doc["paragraphs"], ret.build_index(doc["paragraphs"], did))

retrieved_all = []
for q in tqdm(queries, desc="Retrieve", dynamic_ncols=False):
    paras, idx = idx_map[q["doc_id"]]
    retrieved_all.append(ret.retrieve(q["query"], paras, idx, TOP_K))

del ret; del idx_map; torch.cuda.empty_cache()
print(f"BGE done | VRAM: {torch.cuda.memory_allocated(0)/1e9:.1f}GB", flush=True)

tok = AutoTokenizer.from_pretrained(LLM_PATH)
prompts = []
for q, retrieved in zip(queries, retrieved_all):
    msgs = [{"role": "user", "content": build_prompt(q["query"], retrieved)}]
    prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
del tok

llm = LLM(model=LLM_PATH, dtype="bfloat16", tensor_parallel_size=1,
          gpu_memory_utilization=0.88, max_model_len=4096)
sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)

t1 = time.time()
outputs = llm.generate(prompts, sp)
t2 = time.time()

rows = []
for q, retrieved, out in zip(queries, retrieved_all, outputs):
    gen = out.outputs[0].text
    fb = retrieved[0]["para_id"] if retrieved else "P1"
    a, r = parse_output(gen, fb)
    rows.append({"ID": q["ID"], "abstractive": a, "refs": ",".join(r)})

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["ID","abstractive","refs"])
    w.writeheader(); w.writerows(rows)

print(f"\nSaved {len(rows)} rows → {OUT_CSV}", flush=True)
print(f"Total time: {t2-t0:.0f}s | Generation only: {t2-t1:.0f}s "
      f"({len(queries)/(t2-t1):.1f} q/s)", flush=True)
EOF

echo "=== [$(date)] Evaluating ==="
python -c "
import eval_train
eval_train.PRED_CSV = 'submission_vllm.csv'
import importlib, runpy
runpy.run_module('eval_train', run_name='__main__')
" 2>/dev/null || python -c "
import sys, types
import eval_train as e
e.PRED_CSV = 'submission_vllm.csv'
sol_csv = e.build_sol_from_train(e.TRAIN_PATH)
sol  = e.load_csv(sol_csv)
pred = e.load_csv('submission_vllm.csv')
print(f'Ground truth: {len(sol)} | Predictions: {len(pred)}')
matrix = e.run_evaluation(sol, pred)
matrix['score'] = e.calculate_final_score(matrix)
print('='*40)
for k,v in matrix.items(): print(f'  {k:<12}: {v:.4f}')
print('='*40)
"

echo "=== [$(date)] Done ==="
