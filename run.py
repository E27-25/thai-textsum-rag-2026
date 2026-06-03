#!/usr/bin/env python3
import os
import glob
import json
import re
import csv
import subprocess

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

BGE_PATH     = "/model/bge-m3"
LLM_PATH     = "/model/Qwen3-14B"
TEST_DIR     = "/model/test/"
RESULT_DIR   = "/result/"
PROGRESS_LIB = "/benchmark_lib/progress"
OUTPUT_CSV   = os.path.join(RESULT_DIR, "submission.csv")
TOP_K        = 7
MAX_NEW_TOKENS = 400
GPU          = "cuda:0"
BATCH_SIZE   = 8
BGE_BATCH    = 128


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

    def build_index(self, paragraphs: list[dict], doc_id: str = "") -> torch.Tensor:
        return self._encode([p["text"] for p in paragraphs], BGE_BATCH, f"  encode {doc_id}")

    def retrieve(self, query: str, paragraphs: list[dict], index: torch.Tensor, top_k: int) -> list[dict]:
        q_emb  = self._encode([query], batch_size=1)
        scores = (index @ q_emb.T).squeeze(1)
        top_idx = scores.topk(min(top_k, len(paragraphs))).indices.tolist()
        return [paragraphs[i] for i in top_idx]


def build_prompt(query: str, retrieved: list[dict]) -> str:
    context = "\n".join(f"[{p['para_id']}] {p['text']}" for p in retrieved)
    return (
        f"ย่อหน้าจากบันทึกการประชุม:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"ตอบเป็น JSON เท่านั้น:\n"
        f'{{\"abstractive\": \"...\", \"refs\": [\"P1\"]}}'
    )


def parse_output(text: str, fallback_para: str) -> tuple[str, list[str]]:
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            obj = json.loads(match.group())
            abstractive = str(obj.get("abstractive", "")).strip()
            refs = [str(r) for r in obj.get("refs", [])]
            if abstractive and refs:
                return abstractive, refs
    except Exception:
        pass
    return text.strip(), [fallback_para]


def generate_batch(prompts: list[str], llm, llm_tok, device: str) -> list[str]:
    inputs = llm_tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=3072,
    ).to(device)

    prompt_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = llm.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=llm_tok.pad_token_id,
        )

    results = []
    for out in output_ids:
        text = llm_tok.decode(out[prompt_len:], skip_special_tokens=True)
        results.append(text)
    return results


def main():
    # Find test JSON in mounted directory
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

    # BGE-M3: retrieve all queries, then free VRAM
    print(f"\nLoading BGE-M3 on {GPU} ...", flush=True)
    retriever = BGERetriever(BGE_PATH, GPU)

    index_map: dict[str, tuple[list[dict], torch.Tensor]] = {}
    for doc_id, doc in tqdm(doc_map.items(), desc="Building BGE-M3 indexes", dynamic_ncols=False):
        paras = doc["paragraphs"]
        index = retriever.build_index(paras, doc_id=doc_id)
        index_map[doc_id] = (paras, index)

    print("Retrieving paragraphs for all queries ...", flush=True)
    retrieved_all = []
    for q in tqdm(queries, desc="Retrieving", dynamic_ncols=False):
        paras, index = index_map[q["doc_id"]]
        retrieved = retriever.retrieve(q["query"], paras, index, top_k=TOP_K)
        retrieved_all.append(retrieved)

    del retriever
    del index_map
    torch.cuda.empty_cache()
    print(f"BGE-M3 freed | GPU VRAM: {torch.cuda.memory_allocated(0)/1e9:.1f} GB", flush=True)

    # Load LLM (FP8 on H100)
    llm_tok = AutoTokenizer.from_pretrained(LLM_PATH)
    llm_tok.padding_side = "left"
    if llm_tok.pad_token is None:
        llm_tok.pad_token = llm_tok.eos_token

    print(f"\nLoading {LLM_PATH} (FP8, single GPU) ...", flush=True)
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_PATH,
        torch_dtype=torch.bfloat16,
        device_map=GPU,
        attn_implementation="sdpa",
    )
    llm.eval()

    used_vram  = torch.cuda.memory_allocated(0) / 1e9
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Loaded | GPU0 VRAM: {used_vram:.1f}/{total_vram:.0f} GB\n", flush=True)

    # Batched generation
    rows = []
    n_batches = (len(queries) + BATCH_SIZE - 1) // BATCH_SIZE
    pbar = tqdm(range(0, len(queries), BATCH_SIZE), desc="Inference", unit="batch",
                total=n_batches, dynamic_ncols=False)

    for batch_start in pbar:
        batch_q   = queries[batch_start : batch_start + BATCH_SIZE]
        batch_ret = retrieved_all[batch_start : batch_start + BATCH_SIZE]

        pbar.set_postfix({
            "q": f"{batch_start+1}-{min(batch_start+BATCH_SIZE, len(queries))}/{len(queries)}"
        })

        prompts = []
        for q, retrieved in zip(batch_q, batch_ret):
            messages = [{"role": "user", "content": build_prompt(q["query"], retrieved)}]
            prompts.append(llm_tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            ))

        generated_texts = generate_batch(prompts, llm, llm_tok, GPU)

        for q, retrieved, gen_text in zip(batch_q, batch_ret, generated_texts):
            fallback = retrieved[0]["para_id"] if retrieved else "P1"
            abstractive, refs = parse_output(gen_text, fallback)
            rows.append({"ID": q["ID"], "abstractive": abstractive, "refs": ",".join(refs)})
            tqdm.write(f"[{q['ID']}] refs={refs} | {abstractive[:70]}...")

    # Save results
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "abstractive", "refs"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows → {OUTPUT_CSV}", flush=True)

    # Report progress to benchmark system
    subprocess.run([PROGRESS_LIB, str(len(rows))], check=True)


if __name__ == "__main__":
    main()
