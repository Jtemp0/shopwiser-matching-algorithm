# ShopWiser product-matching pipeline

Matches the same grocery product across UK supermarkets (Tesco, Sainsbury's,
ASDA, Morrisons) from scraped catalogue data, producing cross-retailer
clusters for price comparison.

The pipeline is deterministic up to the LLM filter stage, which uses Claude
Haiku to verify borderline clusters against the contract's clause 4.2
questions.

---

## Prerequisites

- Python 3.14 (see `.python-version`)
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- An Anthropic API key (only needed for the LLM filter and audit steps)

## Installation

```bash
uv sync
```

## Environment

The LLM filter and audit steps read `ANTHROPIC_API_KEY` from a `.env` file.

```bash
cp .env.example .env
# then edit .env and paste your key
```

Scripts that call the API expect the key in the environment. Source it first:

```bash
set -a && source .env && set +a
```

---

## Running the full pipeline

Run these in order from the repo root. Each step reads the previous step's
output from `data/outputs/`.

```bash
# 1. Raw CSV → normalised features
uv run python main.py normalise

# 2. Rule-based (fuzzy + hard-gate) matcher
uv run python main.py cluster

# 3. ML matcher (FAISS retrieval + LightGBM ranker)
uv run python main.py ml-match

# 4. Ensemble (union of the two matchers via Kruskal one-per-retailer)
uv run python main.py ensemble

# 5. LLM filter (Claude Haiku verifies borderline clusters, overwrites in place)
set -a && source .env && set +a
uv run python scripts/llm_ensemble_filter.py

# 6. Finalisation pass (brand canonicalisation, unit_value normalisation,
#    pack-size guard, confidence_score column)
uv run python scripts/improvements/finalise_deliverable.py
```

**Canonical deliverable:** `data/outputs/improvements/ensemble_clusters_final.csv`
(one row per product, grouped by `ensemble_cluster_id`, with a
`confidence_score` per cluster).

To work on a small sample instead of the full catalogue, add `--sample` to
steps 1–4.

---

## Validation & analysis

```bash
# Structural metrics + per-cluster confidence
uv run python scripts/review_metrics.py --clusters data/outputs/improvements/ensemble_clusters_final.csv

# Precision validation against clause 4.2 (Claude Haiku, needs API key)
set -a && source .env && set +a
uv run python scripts/contract_validate_haiku.py

# Precision / calibration / coverage charts
uv run python scripts/improvements/precision_coverage_rigorous.py

# Human review form (50-cluster stratified sample → HTML)
uv run python scripts/improvements/build_review_sheet.py

# Similarity unit tests
uv run python main.py test-similarity
```

---

## Project layout

```
main.py                         Unified CLI (steps 1–4 above)
src/shopwiser/
  preprocess/                   Normalisation, units, brand, attributes
  rule_matcher/                 Fuzzy + hard-gate matcher
  ml_matcher/                   FAISS retrieval + LightGBM ranker
  ensemble/                     Kruskal union of the two matchers
  conflict_tokens.py            Flavour / variant / tier conflict vocabulary
scripts/                        LLM filter, validation, audits
scripts/improvements/           Finalisation, charts, review sheet
data/
  raw/                          Input catalogue CSVs
  processed/                    Normalised features
  outputs/                      Pipeline outputs (clusters, ensemble, final)
```

---

## Adapting to new scraped data

The pipeline keys off column names defined in the preprocess layer. When the
incoming data has different feature naming:

- Column mapping and cleaning: `src/shopwiser/preprocess/normalise.py`
- Brand vocabulary: `src/shopwiser/preprocess/brand.py`
- Conflict / variant tokens: `src/shopwiser/conflict_tokens.py`
- Matcher thresholds: `src/shopwiser/rule_matcher/config.py` and
  `src/shopwiser/ml_matcher/config.py`

Point `data/raw/` at the new CSV and re-run from step 1.

---

## Notes

- The `ensemble-complete`, `ensemble-merge`, and `export-demo` CLI commands are
  optional LLM-completion experiments and are **not** part of the final
  deliverable pipeline. Ignore them for a standard run.
- Categorical confidence bands are intentionally omitted from the deliverable;
  the raw `confidence_score` is shipped so any cut-off can be chosen against the
  empirical calibration (see the reliability diagram from the charts step).
