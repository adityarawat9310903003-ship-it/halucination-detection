"""Baseline comparison + per-category error analysis + correlation with human scores.

  python evaluate.py --data data/sample_adversarial.json

Compares: Pure NLI, Pure QA, Hybrid (fused 40/40/20) on standard vs adversarial splits.
Runs with short-circuiting off so every signal is computed for every claim.
"""
import argparse
import json
from collections import defaultdict

import numpy as np

from hybrid_detector import Config, Detector


def acc(pairs):
    return 100.0 * sum(p == g for p, g in pairs) / len(pairs) if pairs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sample_adversarial.json")
    ap.add_argument("--no-expert", action="store_true")
    ap.add_argument("--out", default="results/eval.json")
    args = ap.parse_args()

    cfg = Config()
    det = Detector(cfg)
    items = json.load(open(args.data, encoding="utf-8"))
    rows = []
    for it in items:
        r = det.verify_claim(it["claim"], it["evidence"], short_circuit=False, use_expert=not args.no_expert)
        gold = it["label"] == "supported"
        rows.append({
            "id": it["id"], "split": it["split"], "category": it["category"], "gold": gold,
            "nli": r.nli["entailment"] >= cfg.threshold,
            "qa": (r.qa_consistency or 0.0) >= cfg.threshold,
            "hybrid": r.fused >= cfg.threshold,
            "fused": r.fused, "human": it.get("human_score"), "label": r.label,
        })

    print(f"\n{'Method':<22}{'Standard':>10}{'Adversarial':>14}")
    for name in ("nli", "qa", "hybrid"):
        std = acc([(r[name], r["gold"]) for r in rows if r["split"] == "standard"])
        adv = acc([(r[name], r["gold"]) for r in rows if r["split"] == "adversarial"])
        print(f"{name.upper():<22}{std:>9.1f}%{adv:>13.1f}%")

    print(f"\n{'Category (hybrid)':<26}{'Accuracy':>10}")
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append((r["hybrid"], r["gold"]))
    for cat, pairs in by_cat.items():
        print(f"{cat:<26}{acc(pairs):>9.1f}%")

    pairs = [(r["fused"], r["human"]) for r in rows if r["human"] is not None]
    if len(pairs) > 2:
        x, y = zip(*pairs)
        print(f"\nPearson correlation with human judgment: {np.corrcoef(x, y)[0, 1]:.3f}")

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"Saved per-claim results to {args.out}")


if __name__ == "__main__":
    main()
