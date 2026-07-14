# Chargeback Intelligence Agent

An automated pipeline that ingests raw, unstructured chargeback dispute narratives and produces structured, evidence-backed defense strategies — anchored to actual Visa/Mastercard network rule text rather than model recall.

## The Problem

When a cardholder disputes a charge, the merchant has a fixed, unforgiving window to produce compliant evidence citing the correct network rule. Miss the deadline or cite the wrong rule, and the dispute is lost — no appeal. Fraud and dispute teams currently do this by manually cross-referencing messy customer complaints against thousands of pages of Visa/Mastercard compliance manuals.

In a compliance-audited environment, an LLM that hallucinates a rule code or invents a timeline isn't a minor bug — it's a liability. This project's design goal was to reduce that risk as far as possible, and to measure — rather than assume — how well it succeeds.

## Architecture

This is a **deterministic, sequential pipeline**, not an autonomous multi-agent system — every case runs through the same fixed sequence of stages. The classes named `*Agent` in `agents.py` are single-purpose helpers (a classifier, a lookup function), not LLM-driven decision-makers with their own tool selection.

```
Raw dispute narrative
   │
   ▼
1. Query Expansion (LLM)         — rewrites the narrative into a precise regulatory search query
   │
   ▼
2. Hybrid Retrieval              — Pinecone dense search (metadata-filtered by network)
                                    + BM25 keyword scoring, blended and re-ranked
                                    by a FlashRank cross-encoder
   │
   ▼
3. Schema-Constrained Generation — OpenAI Structured Outputs, parsed directly into
                                    a Pydantic model (verdict, cited rule, rationale,
                                    evidentiary requirements)
   │
   ▼
4. LLM-as-Judge Evaluation       — RAG Triad scoring (context relevance, groundedness,
                                    answer relevance) against the retrieved context
   │
   ▼
5. Audit Trail                   — logged to production_audit_trail.jsonl, vaulted to S3
```

### Core components

| File | Role |
|---|---|
| `orchestrator.py` | `ChargebackOrchestrator` — runs the end-to-end pipeline per case. Exposes two generation paths: `generate_agent_strategy` (labeled sweep, told the correct verdict in advance) and `generate_live_strategy` / `process_live_case` (blind inference — used by the UI and the blind eval sweep, with no answer given) |
| `agents.py` | Supporting classifiers: intent detection, network-filtered rule retrieval, refund/delivery status lookups |
| `parse_rulebooks.py` | Extracts dispute-relevant pages from source network rulebook PDFs (via `pdfplumber`), detects `Dispute Condition X.X` headers and chunks **at those boundaries** (not blind fixed-word windows), so unrelated rules/tables sitting near each other on a page can't get fused into the same retrieval chunk. Tables are kept as separate chunks, tagged with a condition ID only when unambiguous |
| `seed_vector_db.py` | Embeds parsed rule chunks and upserts them into the Pinecone index (`chargeback-rules`). **Destructive** — prompts for explicit confirmation before clearing and re-seeding the live index |
| `sample_generator.py` | Builds `test_bench.json` from **real CFPB (Consumer Financial Protection Bureau) consumer complaint data** (~13,000 complaints), stratified-sampled so category proportions mirror the real dataset. `expected_verdict` ground-truth labels are assigned by a simple keyword match on category name (`fraud`/`unauthorized`/`stolen` → CHALLENGE, else ACCEPT) — see Limitations below |
| `blind_eval_sweep.py` | Runs all 400 test-bench cases through the **blind** inference path (no expected_verdict given) and scores both verdict accuracy against the labeled ground truth and the RAG Triad — a harder, more honest measure of system capability than the labeled sweep alone |
| `test_bench.json` | 400 real, stratified-sampled CFPB complaint narratives used for both evaluation sweeps |
| `chargeback-ui/app.py` | Streamlit UI, wired to the real `ChargebackOrchestrator` (blind inference path) — not a standalone LLM call. Includes a debug expander showing the actual retrieved context and expanded query behind each result |
| `Dockerfile` / `.github/workflows/deploy.yml` | Containerized via `uv`, deployed live on **AWS ECS Fargate** behind a **GitHub Actions CI/CD pipeline** that builds, pushes to ECR, and force-redeploys on every push to `main` |

## Tech Stack

- **Language:** Python
- **LLM:** OpenAI GPT-4o-mini, with native Structured Outputs (`beta.chat.completions.parse`)
- **Structured generation:** Pydantic
- **Retrieval:** Pinecone (dense, metadata-filtered) + `rank_bm25` (keyword) + FlashRank (`ms-marco-MiniLM-L-12-v2` cross-encoder re-ranking)
- **Evaluation:** LLM-as-judge RAG Triad (context relevance, groundedness, answer relevance)
- **Storage/Audit:** AWS S3 via Boto3
- **Deployment:** Docker (via `uv`), running live on AWS ECS Fargate, deployed through a GitHub Actions CI/CD pipeline (build → ECR → force-redeploy on every push)

## Evaluation Results

### Labeled sweep (400 cases, answer given in advance)

The primary sweep runs all 400 test cases through `generate_agent_strategy`, which is told the correct verdict up front and scored on whether it can produce a well-grounded justification for it:

| Metric | Score |
|---|---|
| Answer Relevance | **0.97** |
| Context Relevance | **0.74** |
| Groundedness | **0.75** |

**Interpretation:** answer relevance is strong — the system reliably targets the actual complaint. Context relevance and groundedness are lower, and the two are linked: in roughly a quarter of cases, retrieval doesn't surface the single tightest-matching rule clause, which gives the generation step less to anchor to and more room to extrapolate. This is the failure mode a compliance-grade RAG system genuinely needs to be evaluated against — surfacing it, rather than a clean scorecard, is the point of building the eval layer as a first-class part of the pipeline.

### Blind sweep (completed — results need one more debugging pass before they're trustworthy)

```
Total Cases Evaluated: 399 / 400
Verdict Accuracy (model vs. labeled ground truth): 0.0000
Cases REJECTED as out-of-scope: 16
Context Relevance : 0.6877
Groundedness      : 0.6989
Answer Relevance  : 0.9424
```

**The 0.0000 verdict accuracy is almost certainly a bug, not a real finding — do not report this number as-is.** A literal zero across 399 cases is statistically implausible for a model that visibly favors CHALLENGE in manual testing; it's far more likely a string-comparison bug in `blind_eval_sweep.py` (the schema's `verdict` field is a plain `str`, not a strict enum, so exact-match comparison can silently fail on formatting differences). Needs a manual read of a few raw lines in `blind_audit_trail.jsonl` to confirm and fix before trusting this metric.

**The context relevance / groundedness drop vs. the labeled sweep (0.74→0.69, 0.75→0.70) is a separate, real signal — but the comparison is confounded.** The labeled sweep was run before the header-aware chunking rewrite; the blind sweep ran after the Pinecone reseed. The drop could mean (a) blind inference is genuinely harder, (b) the new header-based chunks are more fragmented/shorter and losing surrounding context, or some mix of both. Re-running the labeled sweep against the current (reseeded) index would isolate the variable and answer this cleanly.

## ⚠️ Known Issues / Revisit Before Interviews

This project is paused as of the point described below — Project 3 is the current priority, but these are real, understood open threads worth fixing (or at least being able to explain fluently) before relying on this project's numbers in an interview:

1. **`blind_eval_sweep.py` verdict-accuracy bug** — see above. Likely a 10-minute fix (add a strict `Literal["CHALLENGE","ACCEPT","REJECTED"]` type to the schema, or debug the exact string mismatch) once revisited.
2. **Context relevance / groundedness dip after the chunking rewrite** — needs the labeled sweep re-run on the current index to properly isolate cause (see above).
3. **Ground-truth label quality** — the keyword-heuristic label (see below) is the weakest link in both evaluations. Worth checking whether the source CFPB CSV has a "Company response to consumer" column, which would give a real-outcome-based label instead of a name-matching guess.
4. **Live deployment is currently scaled to 0** (ECS desired count) to stop AWS billing — scale back to 1 before demoing live.

**Work already completed based on findings from manual live testing:**
- Fixed a network-mapping bug where American Express cases silently fell back to Mastercard's rule set.
- Rewrote `parse_rulebooks.py` to chunk at detected rule-condition headers instead of blind fixed-word windows, after finding a case where an unrelated rule's heading got fused into the same chunk as the actual applicable text purely due to page proximity.
- Added explicit out-of-scope (`REJECTED`) handling so non-dispute input (jokes, unrelated requests, prompt-injection attempts) isn't forced into a misleading `ACCEPT`/`CHALLENGE` verdict — while ensuring genuine disputes phrased as questions (e.g., eligibility/timing questions) are still correctly treated as in-scope.
- Found, via manual testing with the debug context view, at least one case where the model stated a specific timeframe that was not actually present in the retrieved rule text — a live example of the groundedness gap the metric above is designed to catch.

**Planned next iteration:** tightening the metadata taxonomy and retrieval window further; auditing the label-generation heuristic (see Limitations); strengthening generation-side grounding constraints so numeric/date claims must be traceable to retrieved text.

## Setup

```bash
git clone https://github.com/sbehu/chargeback-intel-agent.git
cd chargeback-intel-agent
pip install -r requirements.txt
```

Create a `.env` file with:
```
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

### 1. Stage the network rulebooks

Place the source PDF rulebooks in a `raw_rulebooks/` folder at the repo root:
```
raw_rulebooks/visa_rulebook.pdf
raw_rulebooks/mastercard_rulebook.pdf
raw_rulebooks/amex_rulebook.pdf
```
> These are proprietary network documents — don't commit them to the public repo. Keep `raw_rulebooks/` in `.gitignore`.

### 2. Parse the rulebooks into chunks

```bash
python parse_rulebooks.py
```
This extracts dispute-relevant pages (filtering out glossary/appendix noise), converts any embedded tables to Markdown, tags each chunk with network/section/page metadata, and writes the result to `raw_rulebooks/parsed_corpus.json`.

### 3. Seed the vector index

```bash
python seed_vector_db.py
```
Creates the `chargeback-rules` Pinecone index (1536-dim, cosine, AWS `us-east-1` serverless) if it doesn't already exist, embeds each chunk with `text-embedding-3-small`, and upserts in batches of 25.

> **Note:** `sample_generator.py` builds `test_bench.json` from a local CFPB complaints CSV (`data/complaints.csv`) — see Limitations for how ground-truth labels are assigned. The exact CLI usage for the UI (`chargeback-ui/`) is covered in its own section below.

Run the full labeled evaluation sweep:
```bash
python orchestrator.py
```
This processes every case in `test_bench.json`, writes results to `production_audit_trail.jsonl`, and uploads the log to the configured S3 bucket.

Run the blind evaluation sweep (harder — no answer given in advance):
```bash
python blind_eval_sweep.py
```
Writes to a separate `blind_audit_trail.jsonl`, so it never overwrites the labeled sweep's output.

## Limitations & Future Work

- **Ground-truth label quality (important):** `expected_verdict` in `test_bench.json` is assigned by a coarse keyword match on the CFPB category name (`fraud`/`unauthorized`/`stolen` → CHALLENGE, everything else → ACCEPT) — it was never manually verified against the actual narrative text. This means both evaluation sweeps are scored against a heuristic label, not a certified-correct answer. A cluster of consistent, category-wide mismatches in the blind sweep is as likely to indicate a bad label as a bad model output — worth auditing before treating verdict accuracy as ground truth.
- The `LogisticsAgent`/`PolicyAgent` imports in `orchestrator.py` and the corresponding class names in `agents.py` still don't match, so those specific helper classes fall back to lightweight stubs — worth reconciling if those checks are meant to be part of the production path.
- **Fargate public IP changes on every redeploy** (new task → new ENI → new public IP by design). The current setup requires re-checking the ECS console after every deploy; a stable address would require an Application Load Balancer in front of the service — not yet implemented.
- Retrieval precision (context relevance) improved with header-aware chunking but is not fully solved; Mastercard and Amex rulebooks produced far fewer chunks than Visa (16 and 12 vs. 374) — worth confirming whether this reflects genuinely shorter source documents or a heading-format mismatch with the current parser's regex.
- Planned: auditing the label heuristic above, tighter metadata taxonomy, refined retrieval window sizing, stricter generation-side grounding constraints on numeric/date claims.

## License

[Add your preferred license here]