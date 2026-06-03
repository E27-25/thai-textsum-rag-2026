#!/usr/bin/env python3
import json
import re
import csv

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────
BGE_PATH       = "./bge-m3"
LLM_PATH       = "./Qwen3-14B"
TEST_PATH      = "data/ชุดข้อมูล/ตัวอย่าง_test_set.json"
OUTPUT_CSV     = "submission.csv"
TOP_K          = 10
MAX_NEW_TOKENS = 512
BGE_DEVICE     = "cuda:0"
LLM_DEVICE     = "cuda:1"   # Qwen3-14B ~28GB fits on one A100 40GB
BATCH_SIZE     = 8          # queries per generation batch
BGE_BATCH      = 128        # paragraphs per bge-m3 encoding batch
# ───────────────────────────────────────────────────────────────────────────


# ── BGE-M3 Retriever ────────────────────────────────────────────────────────
class BGERetriever:
    def __init__(self, model_path: str, device: str):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(
            model_path, dtype=torch.bfloat16
        ).to(device)
        self.model.eval()

    def _encode(self, texts: list[str], batch_size: int, desc: str = "") -> torch.Tensor:
        all_emb = []
        batches = range(0, len(texts), batch_size)
        for i in tqdm(batches, desc=desc, leave=False, dynamic_ncols=False):
            batch = texts[i : i + batch_size]
            inputs = self.tok(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**inputs)
            emb = out.last_hidden_state[:, 0, :]          # CLS pooling
            emb = F.normalize(emb, p=2, dim=1)
            all_emb.append(emb.cpu())
        return torch.cat(all_emb, dim=0)

    def build_index(self, paragraphs: list[dict], doc_id: str = "") -> torch.Tensor:
        texts = [p["text"] for p in paragraphs]
        return self._encode(texts, batch_size=BGE_BATCH, desc=f"  encode {doc_id}")

    def retrieve(
        self,
        query: str,
        paragraphs: list[dict],
        index: torch.Tensor,
        top_k: int = 10,
    ) -> list[dict]:
        q_emb  = self._encode([query], batch_size=1)
        scores = (index @ q_emb.T).squeeze(1)
        top_idx = scores.topk(min(top_k, len(paragraphs))).indices.tolist()
        return [paragraphs[i] for i in top_idx]
# ───────────────────────────────────────────────────────────────────────────


def build_prompt(query: str, retrieved: list[dict]) -> str:
    context = "\n".join(f"[{p['para_id']}] {p['text']}" for p in retrieved)
    return f"""ต่อไปนี้คือย่อหน้าที่เกี่ยวข้องจากบันทึกการประชุม:

{context}

คำถาม: {query}

กรุณาตอบคำถามโดยสรุปเป็นภาษาไทย และระบุหมายเลขย่อหน้าที่ใช้อ้างอิง
ตอบในรูปแบบ JSON เท่านั้น ห้ามมีข้อความอื่น:
{{"abstractive": "...", "refs": ["P1", "P2"]}}"""


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


def generate_batch(
    prompts: list[str],
    llm,
    llm_tok,
    device: str,
) -> list[str]:
    inputs = llm_tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    ).to(device)

    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
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
    # ── Load data ──────────────────────────────────────────────────────────
    with open(TEST_PATH, encoding="utf-8") as f:
        test_data = json.load(f)

    doc_map = {d["doc_id"]: d for d in test_data["docs"]}
    queries  = test_data["queries"]
    print(f"Loaded {len(queries)} queries across {len(doc_map)} docs", flush=True)

    # ── Load BGE-M3 + build indexes (cuda:0) ───────────────────────────────
    print(f"\nLoading BGE-M3 on {BGE_DEVICE} ...", flush=True)
    retriever = BGERetriever(BGE_PATH, BGE_DEVICE)

    index_map: dict[str, tuple[list[dict], torch.Tensor]] = {}
    for doc_id, doc in tqdm(doc_map.items(), desc="Building BGE-M3 indexes", dynamic_ncols=False):
        paras = doc["paragraphs"]
        index = retriever.build_index(paras, doc_id=doc_id)
        index_map[doc_id] = (paras, index)
        tqdm.write(f"  Indexed {doc_id}: {len(paras)} paragraphs")

    # ── Load Qwen3-14B (cuda:1, single GPU) ───────────────────────────────
    print(f"\nLoading Qwen3-14B on {LLM_DEVICE} ...", flush=True)
    llm_tok = AutoTokenizer.from_pretrained(LLM_PATH)
    llm_tok.padding_side = "left"   # required for batched causal LM generation
    if llm_tok.pad_token is None:
        llm_tok.pad_token = llm_tok.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        LLM_PATH,
        dtype=torch.bfloat16,
        device_map=LLM_DEVICE,
        attn_implementation="sdpa",  # PyTorch built-in FlashAttention (no extra package)
    )
    llm.eval()
    print(f"Qwen3-14B loaded | VRAM used: {torch.cuda.memory_allocated(LLM_DEVICE)/1e9:.1f} GB\n", flush=True)

    # ── Retrieve all queries first ─────────────────────────────────────────
    print("Retrieving paragraphs for all queries ...", flush=True)
    retrieved_all = []
    for q in tqdm(queries, desc="Retrieving", dynamic_ncols=False):
        paras, index = index_map[q["doc_id"]]
        retrieved = retriever.retrieve(q["query"], paras, index, top_k=TOP_K)
        retrieved_all.append(retrieved)

    # ── Batched generation ─────────────────────────────────────────────────
    rows = []
    n_batches = (len(queries) + BATCH_SIZE - 1) // BATCH_SIZE
    pbar = tqdm(range(0, len(queries), BATCH_SIZE), desc="Inference", unit="batch",
                total=n_batches, dynamic_ncols=False)

    for batch_start in pbar:
        batch_q   = queries[batch_start : batch_start + BATCH_SIZE]
        batch_ret = retrieved_all[batch_start : batch_start + BATCH_SIZE]

        pbar.set_postfix({
            "queries": f"{batch_start+1}-{min(batch_start+BATCH_SIZE, len(queries))}/{len(queries)}"
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

        generated_texts = generate_batch(prompts, llm, llm_tok, LLM_DEVICE)

        for q, retrieved, gen_text in zip(batch_q, batch_ret, generated_texts):
            fallback = retrieved[0]["para_id"] if retrieved else "P1"
            abstractive, refs = parse_output(gen_text, fallback)
            rows.append({"ID": q["ID"], "abstractive": abstractive, "refs": ",".join(refs)})
            tqdm.write(f"[{q['ID']}] refs={refs} | {abstractive[:70]}...")

    # ── Save ───────────────────────────────────────────────────────────────
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "abstractive", "refs"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows → {OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
