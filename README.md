# Hybrid 2D Hallucination Detector

Sentence-level factual verification for LLM output, built for **AI and Society (CSET485)**, Bennett University.

Every claim is scored on two independent axes and mapped to one of four labels:

|                | **Grounded** (source aligned) | **Not grounded** |
|----------------|-------------------------------|------------------|
| **True**       | Supported                     | Unsupported      |
| **False**      | Contradicted                  | Hallucinated     |

## Architecture

```
claim ──► Retriever (MiniLM embeddings + NLI rerank, k=5)
              │
              ▼
          NLI signal (RoBERTa-MNLI) ──► confident? ──yes──► short-circuit
              │ no
              ├──► QAGS signal  (T5 question generation + RoBERTa-SQuAD2 answering)
              └──► Self-consistency signal (sampled Llama-3 generations checked with NLI)
              │
              ▼
   Grounding = 0.7·NLI + 0.3·QA          Veracity = 0.5·agreement + 0.3·QA + 0.2·SC
              │
              ▼  (veracity ambiguous)
   Expert layer: Groq Llama-3.3-70B  ── "false" ──► veracity × 0.2
              │
              ▼
          2D label
```

Missing signals (e.g. no Groq key) are skipped and the remaining weights are renormalised, so the system degrades gracefully instead of failing at a single point.

## Project layout

```
hybrid_detector/
  config.py       models, weights, thresholds
  nli.py          NLI signal
  qa.py           QAGS signal (coverage + consistency)
  consistency.py  sampling-based self-consistency
  expert.py       Groq client: expert judge + sampler
  retrieval.py    embeddings + NLI reranking
  classifier.py   2D scoring and labelling (pure functions)
  pipeline.py     Detector: ties everything together
main.py           CLI
evaluate.py       baseline comparison, per-category accuracy, Pearson correlation
data/             sample adversarial dataset
tests/            unit tests (no model downloads needed)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export GROQ_API_KEY=your_key      # Windows PowerShell: $env:GROQ_API_KEY="your_key"
```

Models download automatically from Hugging Face on first run. The Groq key is optional: without it, self-consistency and the expert layer are skipped.

## Usage

```bash
# one claim against a source file (documents separated by blank lines)
python main.py --claim "The Eiffel Tower is in Berlin." --source data/eiffel.txt

# whole LLM response, sentence by sentence, JSON output
python main.py --text "Paris is in France. It is the capital of Spain." --source data/eiffel.txt --json

# evaluation: Pure NLI vs Pure QA vs Hybrid, per-category accuracy, Pearson r
python evaluate.py --data data/sample_adversarial.json
```

Python API:

```python
from hybrid_detector import Detector
det = Detector()
r = det.verify_claim("Marie Curie's husband won a Nobel Prize.",
                     ["Marie Curie was married to Pierre Curie.",
                      "Pierre Curie shared the 1903 Nobel Prize in Physics."])
print(r.label, r.grounding, r.veracity)
```

## Tests

```bash
pytest -q
```

## Implementation notes

The report gives the weights and the overall design but not every formula, so some choices here are interpretations. Adjust them in `config.py` / `classifier.py`:

- **Grounding** uses NLI *coverage* (1 − P(neutral): the source takes a stance on the claim) and QA *coverage* (the source can answer the generated questions).
- **Veracity** uses NLI *truth* (0.5 + 0.5·(P(entail) − P(contradict))), QA *consistency* (token-F1 between the claim's span and the source's answer) and self-consistency.
- The **fused 40/40/20 score** is used only for the baseline comparison in `evaluate.py`.
- `data/sample_adversarial.json` is a small illustrative set, **not** the dataset behind the report's reported numbers (81.8% adversarial accuracy, r = 0.563). Results will differ.

## Publishing to GitHub

```bash
git init
git add .
git commit -m "Hybrid 2D hallucination detector"
git branch -M main
git remote add origin https://github.com/<your-username>/hallucination-detector.git
git push -u origin main
```

Create the empty repository on github.com first (no README/.gitignore). Never commit your `.env` or API key.

## Team

Aditya Rawat 
