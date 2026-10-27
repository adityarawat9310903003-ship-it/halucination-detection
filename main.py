"""Command-line interface.

Examples:
  python main.py --claim "The Eiffel Tower is in Berlin." --source data/eiffel.txt
  python main.py --text "Paris is in France. It is the capital of Spain." --source data/eiffel.txt --json
"""
import argparse
import json
import re
import sys

from hybrid_detector import Config, Detector


def load_docs(paths):
    docs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            docs += [d.strip() for d in re.split(r"\n\s*\n", f.read()) if d.strip()]
    return docs


def main():
    ap = argparse.ArgumentParser(description="Hybrid 2D Hallucination Detector")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--claim", help="a single claim to verify")
    g.add_argument("--text", help="a passage; each sentence is verified")
    ap.add_argument("--source", nargs="+", required=True,
                    help="text file(s) with source documents (blank-line separated)")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = ap.parse_args()

    det = Detector(Config())
    if not det.cfg.groq_api_key:
        print("[warn] GROQ_API_KEY not set: self-consistency and expert layer disabled.", file=sys.stderr)
    docs = load_docs(args.source)
    results = ([det.verify_claim(args.claim, docs)] if args.claim else det.verify_text(args.text, docs))

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return
    for r in results:
        print(f"[{r.label:<12}] grounding={r.grounding:.2f} veracity={r.veracity:.2f}  {r.claim}")


if __name__ == "__main__":
    main()
