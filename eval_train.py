#!/usr/bin/env python3
"""Eval submission_train.csv against train_set.json ground truth.
Logic copied from evaluate_sample/eval.py with local BGE-M3 path."""
import json
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from pythainlp.tokenize import word_tokenize
from rouge_score import rouge_scorer
from rouge_score.tokenizers import Tokenizer

TRAIN_PATH = "data/ชุดข้อมูล/train_set.json"
PRED_CSV   = "submission_train.csv"
BGE_PATH   = "./bge-m3"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Copied from evaluate_sample/eval.py ───────────────────────────────────

def tokenize_thai(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""
    tokens = word_tokenize(text, engine="newmm", keep_whitespace=False)
    return " ".join(tokens)

class ThaiSpaceTokenizer(Tokenizer):
    def tokenize(self, text):
        return text.split(" ")

def load_csv(file_path):
    df = pd.read_csv(file_path)

    def parse_para(x):
        if pd.isna(x) or str(x).strip() == "":
            return []
        try:
            return [i.strip() for i in str(x).split(",")]
        except ValueError:
            return []

    if 'refs' in df.columns:
        df['refs'] = df['refs'].apply(parse_para)

    return df

def calculate_iou(list_pred, list_sol):
    set_pred = set(list_pred) if isinstance(list_pred, list) else set()
    set_sol  = set(list_sol)  if isinstance(list_sol,  list) else set()
    if not set_sol:
        return 0.0
    return len(set_pred.intersection(set_sol)) / len(set_pred.union(set_sol))

def run_evaluation(sol: pd.DataFrame, pred: pd.DataFrame, merge='ID'):
    if len(sol) != len(pred):
        raise ValueError("จำนวนแถวของ sol และ pred ไม่เท่ากัน")

    df = pd.merge(sol, pred, on=merge, suffixes=('_sol', '_pred'))

    df['IoU'] = df.apply(lambda x: calculate_iou(x['refs_pred'], x['refs_sol']), axis=1)

    scorer = rouge_scorer.RougeScorer(['rougeL'],
                                      use_stemmer=False,
                                      tokenizer=ThaiSpaceTokenizer())

    sol_toks  = df['abstractive_sol'].apply(tokenize_thai)
    pred_toks = df['abstractive_pred'].apply(tokenize_thai)
    results   = [scorer.score(g, p) for g, p in zip(sol_toks, pred_toks)]
    df['rougeL'] = [r['rougeL'].fmeasure for r in results]

    # SS-score with local BGE-M3
    model = SentenceTransformer(BGE_PATH)

    texts = df['abstractive_sol'].fillna("").astype(str).tolist() + df['abstractive_pred'].fillna("").astype(str).tolist()
    embeddings = model.encode(texts, batch_size=32,
                              convert_to_tensor=True,
                              normalize_embeddings=True)

    ref_emb  = embeddings[0 : len(texts) // 2]
    pred_emb = embeddings[len(texts) // 2 :]

    scores = F.cosine_similarity(pred_emb, ref_emb, dim=1)
    df['SS-score'] = scores.cpu().numpy()

    metric_cols = ["rougeL", "SS-score", "IoU"]
    final_report = df[metric_cols].mean().to_dict()

    return final_report

def calculate_final_score(metrics_dict):
    wss, wrl, wj = 0.45, 0.35, 0.2
    ss = metrics_dict['SS-score']
    rl = metrics_dict['rougeL']
    j  = metrics_dict['IoU']
    return wss * ss + wrl * rl + wj * j

# ── Train-specific: build sol CSV from train_set.json ─────────────────────

def build_sol_from_train(train_path: str) -> pd.DataFrame:
    with open(train_path, encoding="utf-8") as f:
        data = json.load(f)
    rows = [
        {"ID": q["ID"], "abstractive": q["abstractive"], "refs": ",".join(q["refs"])}
        for q in data["queries"]
    ]
    sol = pd.DataFrame(rows)
    sol_csv = "sol_train.csv"
    sol.to_csv(sol_csv, index=False)
    print(f"Solution CSV saved → {sol_csv} ({len(sol)} rows)", flush=True)
    return sol_csv

# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sol_csv = build_sol_from_train(TRAIN_PATH)
    sol  = load_csv(sol_csv)
    pred = load_csv(PRED_CSV)

    print(f"Ground truth: {len(sol)} | Predictions: {len(pred)}", flush=True)

    matrix = run_evaluation(sol, pred)
    matrix['score'] = calculate_final_score(matrix)

    print("\n" + "=" * 40)
    print("  Train Eval Results")
    print("=" * 40)
    for k, v in matrix.items():
        print(f"  {k:<12}: {v:.4f}")
    print("=" * 40)

    # ── Per-question breakdown ────────────────────────────────────────────────
    import numpy as np

    scorer_rl = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False,
                                         tokenizer=ThaiSpaceTokenizer())

    with open(TRAIN_PATH, encoding="utf-8") as f:
        train_data = json.load(f)
    gt_map = {q["ID"]: q for q in train_data["queries"]}

    rows = []
    for _, row in pd.merge(sol, pred, on='ID', suffixes=('_sol','_pred')).iterrows():
        qid    = row["ID"]
        gt     = gt_map[qid]
        iou_sc = calculate_iou(row["refs_pred"], row["refs_sol"])
        rl_sc  = scorer_rl.score(
            tokenize_thai(row["abstractive_sol"]),
            tokenize_thai(row["abstractive_pred"])
        )["rougeL"].fmeasure
        rows.append({
            "ID":       qid,
            "query":    gt["query"],
            "doc_id":   gt["doc_id"],
            "rougeL":   round(rl_sc, 4),
            "iou":      round(iou_sc, 4),
            "partial":  round(0.35 * rl_sc + 0.20 * iou_sc, 4),
            "gt_refs":  gt["refs"],
            "pred_refs":row["refs_pred"],
            "gt_abst":  gt["abstractive"],
            "pred_abst":row["abstractive_pred"],
        })

    rows.sort(key=lambda x: x["partial"])

    # Save detail JSON
    detail_path = "eval_detail.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\nPer-question detail saved → {detail_path}")

    # Error pattern counts
    iou_zero    = [r for r in rows if r["iou"] == 0]
    iou_perfect = [r for r in rows if r["iou"] == 1.0]
    rouge_low   = [r for r in rows if r["rougeL"] < 0.1]

    print(f"\n--- Error patterns ---")
    print(f"  IoU = 0   (wrong refs entirely):  {len(iou_zero):4d}/{len(rows)}")
    print(f"  IoU = 1.0 (perfect refs):         {len(iou_perfect):4d}/{len(rows)}")
    print(f"  RougeL < 0.1 (very bad summary):  {len(rouge_low):4d}/{len(rows)}")

    print(f"\n--- Bottom 20 worst (RougeL+IoU weighted) ---")
    for r in rows[:20]:
        print(f"[{r['ID']}] RougeL={r['rougeL']:.3f}  IoU={r['iou']:.3f}  doc={r['doc_id']}")
        print(f"  Q:    {r['query'][:90]}")
        print(f"  GT:   {r['gt_abst'][:90]}")
        print(f"  Pred: {r['pred_abst'][:90]}")
        print(f"  GT refs:   {r['gt_refs']}")
        print(f"  Pred refs: {r['pred_refs']}")
        print()
