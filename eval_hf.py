#!/usr/bin/env python3
"""Transformers-based eval (no vllm). Works on CUDA 12.x.
Usage:
  python eval_hf.py --model ./Qwen3-14B [--out submission_14b.csv] [--batch 4]
"""
import argparse, json, re, csv, time, os
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

BGE_PATH  = "./bge-m3"
DATA_PATH = "data/ชุดข้อมูล/train_set.json"
TOP_K     = 7
MAX_TOKENS = 400
BGE_BATCH = 128


class BGERetriever:
    def __init__(self, device):
        self.device = device
        self.tok   = AutoTokenizer.from_pretrained(BGE_PATH)
        self.model = AutoModel.from_pretrained(BGE_PATH, torch_dtype=torch.bfloat16).to(device)
        self.model.eval()

    def _encode(self, texts, bs, desc=""):
        embs = []
        for i in tqdm(range(0, len(texts), bs), desc=desc, leave=False, dynamic_ncols=False):
            inp = self.tok(texts[i:i+bs], padding=True, truncation=True,
                           max_length=512, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**inp)
            embs.append(F.normalize(out.last_hidden_state[:,0,:], p=2, dim=1).cpu())
        return torch.cat(embs)

    def build_index(self, paras, doc_id=""):
        return self._encode([p["text"] for p in paras], BGE_BATCH, f"  {doc_id}")

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
            a = str(obj.get("abstractive", "")).strip()
            r = [str(x) for x in obj.get("refs", [])]
            if a and r:
                return a, r
    except Exception:
        pass
    return text.strip(), [fallback]


def generate_batch(model, tok, prompts, batch_size, max_new_tokens):
    results = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating", dynamic_ncols=False):
        batch = prompts[i:i+batch_size]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=3800).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tok.eos_token_id,
            )
        for j, seq in enumerate(out):
            in_len = enc["input_ids"].shape[1]
            gen = tok.decode(seq[in_len:], skip_special_tokens=True)
            results.append(gen)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--load-4bit", action="store_true", help="Force 4-bit quantization")
    args = ap.parse_args()

    model_name = os.path.basename(args.model.rstrip("/"))
    out_csv = args.out or f"submission_{model_name}.csv"

    print(f"\n{'='*60}", flush=True)
    print(f"Model : {model_name}", flush=True)
    print(f"Output: {out_csv}", flush=True)
    print(f"{'='*60}\n", flush=True)

    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    doc_map = {d["doc_id"]: d for d in data["docs"]}
    queries = data["queries"]
    print(f"Loaded {len(queries)} queries, {len(doc_map)} docs", flush=True)

    # BGE
    print("Loading BGE-M3 ...", flush=True)
    ret = BGERetriever("cuda:0")
    idx_map = {}
    for did, doc in tqdm(doc_map.items(), desc="Build BGE index", dynamic_ncols=False):
        idx_map[did] = (doc["paragraphs"], ret.build_index(doc["paragraphs"], did))

    retrieved_all = []
    for q in tqdm(queries, desc="Retrieve", dynamic_ncols=False):
        paras, idx = idx_map[q["doc_id"]]
        retrieved_all.append(ret.retrieve(q["query"], paras, idx, TOP_K))

    del ret, idx_map
    torch.cuda.empty_cache()
    print(f"BGE freed | VRAM: {torch.cuda.memory_allocated(0)/1e9:.1f} GB", flush=True)

    # Build prompts
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = []
    for q, retrieved in zip(queries, retrieved_all):
        msgs = [{"role": "user", "content": build_prompt(q["query"], retrieved)}]
        prompts.append(tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))

    # Load LLM
    print(f"Loading {model_name} ...", flush=True)
    load_kwargs = dict(
        pretrained_model_name_or_path=args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        load_kwargs.pop("torch_dtype")

    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    model.eval()
    print(f"Model loaded | VRAM: {torch.cuda.memory_allocated(0)/1e9:.1f} GB", flush=True)

    t0 = time.time()
    outputs = generate_batch(model, tok, prompts, args.batch, MAX_TOKENS)
    elapsed = time.time() - t0
    print(f"\nGeneration: {elapsed:.0f}s | {len(queries)/elapsed:.2f} q/s", flush=True)

    rows = []
    for q, retrieved, gen in zip(queries, retrieved_all, outputs):
        fb = retrieved[0]["para_id"] if retrieved else "P1"
        a, r = parse_output(gen, fb)
        rows.append({"ID": q["ID"], "abstractive": a, "refs": ",".join(r)})

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ID", "abstractive", "refs"])
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {len(rows)} rows → {out_csv}", flush=True)

    # Eval
    print("\nRunning eval ...", flush=True)
    try:
        import eval_train as et
        sol_csv = et.build_sol_from_train(DATA_PATH)
        sol  = et.load_csv(sol_csv)
        pred = et.load_csv(out_csv)
        matrix = et.run_evaluation(sol, pred)
        matrix["score"] = et.calculate_final_score(matrix)
        print(f"\n{'='*40}")
        print(f"  {model_name}")
        print(f"{'='*40}")
        for k, v in matrix.items():
            print(f"  {k:<12}: {v:.4f}")
        print(f"{'='*40}\n")
    except Exception as e:
        print(f"Eval error: {e}", flush=True)


if __name__ == "__main__":
    main()
