# Visual Chain-of-Thought Data Pipeline — Chart Reasoning

An end-to-end automated pipeline that takes raw ChartQA samples and produces a Supervised Fine-Tuning (SFT) dataset of structured Visual Chain-of-Thought reasoning traces. Built as the Problem 2 deliverable for the *Visual Chain-of-Thought Data Engineering for VLMs* assignment.

The pipeline is **CPU-friendly** — all heavy lifting is offloaded to the Gemini API (free tier sufficient for the demo). It runs end-to-end on ~100 chart samples in under 30 minutes on a standard laptop.

---

## What This Does

Modern Vision-Language Models can read a bar chart and answer "by how much did revenue grow from Q2 to Q4?" — but they often do it in a single opaque step that's prone to silent errors. This pipeline produces training data that teaches the model to *reason out loud* before answering, using a four-stage structured trace:

```
<chart_summary>   The chart type, axes, what it depicts.
<extraction>      The specific data values relevant to the question.
<computation>     The arithmetic / comparison performed step by step.
<answer>          The final answer in canonical form.
```

A model fine-tuned on this format is more accurate, more auditable, and less prone to silent miscalculation than one trained on direct question-answer pairs alone.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  ChartQA (HuggingFace)                                          │
│         │                                                       │
│         ▼                                                       │
│  [1] download_chartqa.py                                        │
│         │  Stratified sampling → 100 chart QA samples           │
│         ▼                                                       │
│  data/raw/chartqa_100.jsonl                                     │
│         │                                                       │
│         ▼                                                       │
│  [2] generate_traces.py                                         │
│         │  3 traces × 100 samples via Gemini 2.5 Flash          │
│         ▼                                                       │
│  data/synth/traces.jsonl                                        │
│         │                                                       │
│         ▼                                                       │
│  [3] filtering.py                                               │
│         │  Schema → Answer → Grounding → Computation →          │
│         │  Self-consistency gates                               │
│         ▼                                                       │
│  data/filtered/clean.jsonl                                      │
│         │                                                       │
│         ▼                                                       │
│  [4] formatting.py                                              │
│         │  Convert to chat-format SFT JSONL, 90/10 train/val    │
│         ▼                                                       │
│  data/final/sft_train.jsonl                                     │
│  data/final/sft_val.jsonl                                       │
│         │                                                       │
│         ▼                                                       │
│  [5] validate.py                                                │
│         │  Schema, image-existence, tag presence checks         │
│         ▼                                                       │
│  Validation report                                              │
└─────────────────────────────────────────────────────────────────┘
```

Each stage reads JSONL from the previous stage and writes JSONL to the next, so any stage can be run, debugged, or resumed independently.

---


## Step-by-Step Execution

If you prefer to run the stages individually (recommended on first run, so you can inspect intermediate outputs):

### Step 1 — Download and Stratify Samples

```bash
python scripts/download_chartqa.py
```

Streams the `lmms-lab/ChartQA` test split, buckets the first 2,000 examples by chart type and question type (heuristic — `bar_arithmetic`, `pie_comparison`, etc.), then samples ~100 examples evenly across buckets. Saves images to `data/raw/images/` and a JSONL manifest to `data/raw/chartqa_100.jsonl`.

**Output schema (one line):**
```json
{
  "id": "chartqa_136",
  "image_path": "data/raw/images/chartqa_136.png",
  "question": "By how much did revenue grow from Q2 to Q4?",
  "gold_answer": "12",
  "data_table": "..."
}
```

### Step 2 — Synthesize CoT Traces

```bash
python scripts/generate_traces.py
```

For each sample, calls Gemini 2.5 Flash three times at temperature 0.7 to generate three independent four-stage traces. Three traces enable downstream **self-consistency filtering** — if the teacher truly understands the chart, the three traces should converge on the same final answer.

The script is **resumable**: if interrupted, rerun and it picks up where it left off (it tracks `processed_ids` from the existing output file). Each call uses exponential-backoff retry through `tenacity` to handle rate limits and transient failures.

The system prompt explicitly instructs the teacher to ground every numeric value in the chart or data table and to match the gold answer. When ChartQA provides the underlying data table, it is included in the prompt — this dramatically reduces hallucination.

**Output schema:**
```json
{
  "id": "chartqa_136",
  "question": "...",
  "gold_answer": "12",
  "data_table": "...",
  "traces": ["<chart_summary>...</chart_summary>...", "...", "..."]
}
```

### Step 3 — Filter Through Quality Gates

```bash
python scripts/filtering.py
```

Each of the three traces per sample is sent through five gates in order. A trace is rejected the moment any gate fails. The sample as a whole is rejected if fewer than 2 of its 3 traces survive (self-consistency requirement).

| Gate | What it checks | Why it matters |
|---|---|---|
| **Schema** | All four tags present, in order, non-empty | Catches malformed teacher outputs |
| **Answer correctness** | Final `<answer>` matches gold within 1% | Rejects traces that hallucinated to a wrong conclusion |
| **Value grounding** | Every numeric value in `<extraction>` exists in the data table within 2% | The killer filter — catches numbers invented by the teacher |
| **Arithmetic verification** | Re-evaluates each `A op B = C` in `<computation>` | Catches "right answer, wrong work" — critical because training on this teaches confabulation |
| **Self-consistency** | At least 2 of 3 traces produce the same answer | Hallucinations rarely repeat consistently (Wang et al. 2023) |

For samples that pass, the **single best trace** is kept (highest grounding score, with computation length as tiebreaker). The script prints a yield report:

```
=== Filtering Pipeline Status ===
=== Filtering Pipeline Status ===
total_samples            : 100
rejected_schema          : 0
rejected_answer          : 5
rejected_grounding       : 0
rejected_computation     : 0
rejected_consistency     : 0
surviving_samples        : 95
Overall Yield: 95.0% (this is because the data_table was not availabe for any samples so grounding gate did not work)
```

### Step 4 — Format for SFT

```bash
python scripts/formatting.py
```

Converts each surviving sample into a chat-format JSONL record ready for SFT, then splits into 90% train / 10% validation:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "path": "data/raw/images/chartqa_136.png"},
        {"type": "text", "text": "By how much did revenue grow from Q2 to Q4?"}
      ]
    },
    {
      "role": "assistant",
      "content": "<chart_summary>...</chart_summary>\n<extraction>...</extraction>\n<computation>...</computation>\n<answer>12</answer>"
    }
  ],
  "metadata": {"source": "chartqa", "id": "chartqa_136"}
}
```



## Project Structure

```
visual-cot/
├── README.md                          # This file
├── LICENSE                            # MIT
├── requirements.txt                   # Python dependencies
├── .env.example                       # Template for GEMINI_API_KEY
├── .gitignore

│
├── scripts/
│   ├── download_chartqa.py            # Stage 1: load + stratify
│   ├── generate_traces.py             # Stage 2: teacher synthesis
│   ├── filtering.py                   # Stage 3: quality gates
│   ├── formatting.py                  # Stage 4: SFT formatting
│
└── data/
    ├── raw/
    │   ├── chartqa_100.jsonl          # Stage 1 output (tracked)
    │   └── images/                    # PNG images (gitignored)
    ├── synth/                         # Stage 2 output (gitignored)
    ├── filtered/                      # Stage 3 output (gitignored)
    └── final/
        ├── sft_train.jsonl            # Final deliverable (tracked)
        └── sft_val.jsonl              # Final deliverable (tracked)
```

---

## Worked Example

**Input** (from `data/raw/chartqa_100.jsonl`):
```json
{
  "id": "chartqa_136",
  "image_path": "data/raw/images/chartqa_136.png",
  "question": "What is the difference between the highest and lowest values?",
  "gold_answer": "18",
  "data_table": "Q1: 12, Q2: 18, Q3: 25, Q4: 30"
}
```

**Output** (assistant content from `data/final/sft_train.jsonl`):
```
<chart_summary>
The image is a vertical bar chart showing quarterly values across four
periods (Q1, Q2, Q3, Q4) on the x-axis, with values ranging from 0 to
30 on the y-axis.
</chart_summary>

<extraction>
- Highest value: Q4 = 30
- Lowest value: Q1 = 12
</extraction>

<computation>
Difference = highest - lowest
            = 30 - 12
            = 18
</computation>

<answer>
18
</answer>
```

This trace was generated by the teacher, passed all five filter gates (schema, answer correctness, value grounding against the data table, arithmetic verification, self-consistency with two sibling traces), and was selected as the highest-grounded survivor.

---


## Scaling to Production

The pipeline as shipped processes ~100 samples on a laptop. Scaling to the ~200k-sample target outlined in the design document requires the following changes:

**Sharded streaming.** Replace the upfront pool collection in `download_chartqa.py` with `datasets` streaming + sharding. Process ~10k samples per shard so the dataset never lives entirely in memory.

**Async / batched teacher calls.** The current synchronous loop in `generate_traces.py` makes ~5 calls per minute. Replace with an `asyncio` queue of API workers rate-limited to your provider quota — for Gemini Tier 1 this is ~360 RPM, lifting throughput to ~5-10 samples per second.

**Self-hosted teacher above ~500k samples.** API costs dominate at high volume. Switch to Qwen2.5-VL-72B served via vLLM on 4× A100s (or equivalent) for a roughly 5-10× cost reduction.

**Programmatic synthesis where possible.** ChartQA provides underlying data tables for ~70% of samples. For these, replace teacher distillation with deterministic template-based generation (the MAVIS approach, Zhang et al. 2024). Programmatic samples have zero hallucination floor, are effectively free in compute, and are easier to audit.

**Perceptual-hash dedup.** Add image-hash dedup against ShareGPT4V, ChartGemma, and other public CoT chart corpora before synthesis. Avoids paying to regenerate samples already in the public pool.

**Manual review loop.** Sample 5% of survivors for human spot-check; use disagreements between human and automated filters to retune thresholds quarterly.

---

## Configuration

All scripts read paths and parameters from constants at the top of each file. Key knobs:

| Script | Constant | Default | Purpose |
|---|---|---|---|
| download_chartqa.py | `SAMPLE_SIZE` | 100 | How many samples to draw |
| download_chartqa.py | `POOL_SIZE` | 2000 | How many to pool before stratifying |
| generate_traces.py | `MODEL_NAME` | `gemini-2.5-flash` | Teacher model |
| generate_traces.py | (in code) | temperature 0.7, n=3 | Sampling for self-consistency |
| filtering.py | `rel_tol` | 0.01-0.02 | Answer / grounding tolerances |


---






