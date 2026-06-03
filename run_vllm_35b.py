#!/usr/bin/env python3
import time; time.sleep(10)  # wait for Loki to collect startup logs
import sys, os
print("[BOOT] Python starting", flush=True)
print(f"[BOOT] VLLM_USE_V1={os.environ.get('VLLM_USE_V1','<unset>')}", flush=True)
print(f"[BOOT] LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH','<unset>')[:300]}", flush=True)

import glob
import json
import re
import csv
import subprocess

try:
    import torch
    import torch.nn.functional as F
    print(f"[BOOT] torch {torch.__version__} OK | CUDA={torch.cuda.is_available()} | devices={torch.cuda.device_count()}", flush=True)
    if torch.cuda.is_available():
        print(f"[BOOT] {torch.cuda.get_device_name(0)} {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB", flush=True)
except Exception as _e:
    print(f"[BOOT] torch FAILED: {_e}", flush=True)
    sys.exit(1)

try:
    from transformers import AutoModel, AutoTokenizer
    print("[BOOT] transformers OK", flush=True)
except Exception as _e:
    print(f"[BOOT] transformers FAILED: {_e}", flush=True)
    sys.exit(1)

try:
    from vllm import LLM, SamplingParams
    print("[BOOT] vllm OK", flush=True)
except Exception as _e:
    print(f"[BOOT] vllm FAILED: {_e}", flush=True)
    sys.exit(1)

from tqdm import tqdm

BGE_PATH       = "/model/bge-m3"
LLM_PATH       = "/model/Qwen3.6-35B-A3B-FP8"
TEST_DIR       = "/model/test/"
RESULT_DIR     = "/result/"
PROGRESS_LIB   = "/benchmark_lib/progress"
OUTPUT_CSV     = os.path.join(RESULT_DIR, "submission.csv")
TOP_K_RETRIEVE = 7    # dense-only, no rerank → direct top-7
TOP_K_FINAL    = 7
MAX_NEW_TOKENS = 200  # JSON output fits in ~150 tokens
MAX_PARA_CHARS = 600  # shorter prompts → faster prefill
GPU            = "cuda:0"
BGE_BATCH      = 64

SYSTEM_PROMPT = (
    "คุณเป็นผู้ช่วยตอบคำถามจากบันทึกการประชุมภาษาไทย "
    "ตอบให้ครบถ้วน ชัดเจน และอ้างอิงเฉพาะข้อมูลจากย่อหน้าที่ให้มาเท่านั้น"
)


class BGERetriever:
    def __init__(self, model_path: str, device: str):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        ).to(device)
        self.model.eval()

    def _encode(self, texts: list[str], batch_size: int, desc: str = "") -> torch.Tensor:
        all_emb = []
        for i in tqdm(range(0, len(texts), batch_size), desc=desc, leave=False, dynamic_ncols=False):
            batch = texts[i : i + batch_size]
            inputs = self.tok(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**inputs)
            emb = out.last_hidden_state[:, 0, :]
            emb = F.normalize(emb, p=2, dim=1)
            all_emb.append(emb.cpu())
        return torch.cat(all_emb, dim=0)

    def _encode_tokens(self, text: str) -> torch.Tensor:
        """All-token embeddings for ColBERT late-interaction (L2 normalized)."""
        inputs = self.tok(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        embs = out.last_hidden_state[0]
        return F.normalize(embs, p=2, dim=-1)

    def build_index(self, paragraphs: list[dict], doc_id: str = "") -> torch.Tensor:
        return self._encode([p["text"] for p in paragraphs], BGE_BATCH, f"  encode {doc_id}")

    def retrieve(self, query: str, paragraphs: list[dict], index: torch.Tensor, top_k: int) -> list[dict]:
        q_emb  = self._encode([query], batch_size=1)
        scores = (index @ q_emb.T).squeeze(1)
        top_idx = scores.topk(min(top_k, len(paragraphs))).indices.tolist()
        return [paragraphs[i] for i in top_idx]

    def rerank(self, query: str, passages: list[dict], top_k: int) -> list[dict]:
        """ColBERT late-interaction: sum of per-query-token max similarities."""
        q_embs = self._encode_tokens(query)
        scores = []
        for p in passages:
            p_embs = self._encode_tokens(p["text"][:800])
            score = (q_embs @ p_embs.T).max(dim=1).values.sum().item()
            scores.append(score)
        ranked = sorted(range(len(passages)), key=lambda i: scores[i], reverse=True)
        return [passages[i] for i in ranked[:top_k]]


def build_prompt(query: str, retrieved: list[dict], max_para_chars: int = 800) -> str:
    # Most relevant last — LLM recency bias helps recall
    ordered = list(reversed(retrieved))
    context = "\n".join(f"[{p['para_id']}] {p['text'][:max_para_chars]}" for p in ordered)
    ids = ", ".join(f'"{p["para_id"]}"' for p in ordered)
    return (
        f"ย่อหน้าจากบันทึกการประชุม:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"ให้ตอบคำถามโดยสรุปคำตอบจากย่อหน้าข้างต้น และระบุ ID ย่อหน้าที่เกี่ยวข้อง\n"
        f"ID ที่เลือกได้: {ids}\n\n"
        f"ตอบเป็น JSON เท่านั้น:\n"
        f'{{"abstractive": "สรุปคำตอบภาษาไทย", "refs": ["id1", "id2"]}}'
    )


def expand_contiguous(refs: list[str], valid_ids: set[str]) -> list[str]:
    """Fill single-paragraph gaps: if P51+P53 both cited and P52 exists → add P52."""
    if len(refs) <= 1:
        return refs
    nums = []
    for r in refs:
        m = re.match(r'([A-Za-z]+)(\d+)$', r)
        if m:
            nums.append((int(m.group(2)), m.group(1), r))
    if not nums:
        return refs
    nums.sort()
    result_ids = set(refs)
    for i in range(len(nums) - 1):
        n1, pfx, _ = nums[i]
        n2, _, _   = nums[i + 1]
        if n2 - n1 == 2:
            mid = f"{pfx}{n1 + 1}"
            if mid in valid_ids:
                result_ids.add(mid)
    return sorted(result_ids, key=lambda x: int(re.search(r'\d+', x).group()))


def parse_output(text: str, fallback_para: str, valid_para_ids: set[str] | None = None) -> tuple[str, list[str]]:
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            obj = json.loads(match.group())
            abstractive = str(obj.get("abstractive", "")).strip()
            refs = [str(r) for r in obj.get("refs", [])]
            if abstractive and refs:
                if valid_para_ids:
                    refs = expand_contiguous(refs, valid_para_ids)
                return abstractive, refs
    except Exception:
        pass
    return text.strip(), [fallback_para]


def main():
    import platform as _pl
    print(f"[START] Python {sys.version} | {_pl.platform()}", flush=True)
    print(f"[ENV] LLM_PATH={LLM_PATH}", flush=True)
    print(f"[ENV] TEST_DIR={TEST_DIR} | RESULT_DIR={RESULT_DIR}", flush=True)
    print(f"[CUDA] available={torch.cuda.is_available()} | device_count={torch.cuda.device_count()}", flush=True)
    if torch.cuda.is_available():
        print(f"[CUDA] {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB", flush=True)

    print(f"[STEP 1] Scanning {TEST_DIR} ...", flush=True)
    json_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.json")))
    if not json_files:
        raise FileNotFoundError(f"No JSON file found in {TEST_DIR}")
    test_path = json_files[0]
    print(f"Test file: {test_path}", flush=True)

    os.makedirs(RESULT_DIR, exist_ok=True)

    with open(test_path, encoding="utf-8") as f:
        test_data = json.load(f)

    doc_map = {d["doc_id"]: d for d in test_data["docs"]}
    queries  = test_data["queries"]
    print(f"Loaded {len(queries)} queries across {len(doc_map)} docs", flush=True)

    print(f"[STEP 2] Loading BGE-M3 on {GPU} ...", flush=True)
    retriever = BGERetriever(BGE_PATH, GPU)

    index_map: dict[str, tuple[list[dict], torch.Tensor]] = {}
    for doc_id, doc in tqdm(doc_map.items(), desc="Building BGE-M3 indexes", dynamic_ncols=False):
        paras = doc["paragraphs"]
        index = retriever.build_index(paras, doc_id=doc_id)
        index_map[doc_id] = (paras, index)

    print("Retrieving paragraphs for all queries (dense only) ...", flush=True)
    retrieved_all = []
    for q in tqdm(queries, desc="Retrieve", dynamic_ncols=False):
        paras, index = index_map[q["doc_id"]]
        retrieved = retriever.retrieve(q["query"], paras, index, top_k=TOP_K_FINAL)
        retrieved_all.append(retrieved)

    del retriever
    del index_map
    torch.cuda.empty_cache()
    print(f"BGE-M3 freed | VRAM: {torch.cuda.memory_allocated(0)/1e9:.1f} GB", flush=True)

    print(f"[STEP 3] Building prompts ...", flush=True)
    tok = AutoTokenizer.from_pretrained(LLM_PATH)
    prompts = []
    for q, retrieved in zip(queries, retrieved_all):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(q["query"], retrieved)},
        ]
        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(prompt)
    del tok

    print(f"[STEP 4] Loading {LLM_PATH} via vLLM ...", flush=True)
    llm = LLM(
        model=LLM_PATH,
        dtype="auto",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        max_model_len=4096,
        max_num_seqs=8,
        trust_remote_code=True,
        enforce_eager=False,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=MAX_NEW_TOKENS,
    )

    print(f"Generating {len(prompts)} answers ...", flush=True)
    outputs = llm.generate(prompts, sampling_params)

    rows = []
    for q, retrieved, output in zip(queries, retrieved_all, outputs):
        gen_text = output.outputs[0].text
        fallback = retrieved[0]["para_id"] if retrieved else "P1"
        valid_ids = {p["para_id"] for p in doc_map[q["doc_id"]]["paragraphs"]}
        abstractive, refs = parse_output(gen_text, fallback, valid_ids)
        rows.append({"ID": q["ID"], "abstractive": abstractive, "refs": ",".join(refs)})
        tqdm.write(f"[{q['ID']}] refs={refs} | {abstractive[:70]}...")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "abstractive", "refs"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows → {OUTPUT_CSV}", flush=True)
    try:
        subprocess.run([PROGRESS_LIB, str(len(rows))], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"Progress update skipped: {e}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise
